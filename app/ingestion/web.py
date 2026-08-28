"""Web Page Ingestion (Phase 11.3).

Fetches web pages, canonicalizes URLs, extracts content with metadata,
and retains URI/date provenance.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlparse
from uuid import UUID

import requests
from bs4 import BeautifulSoup

from app.config import get_settings
from app.ingestion.chunking import TextSegment
from app.ingestion.multimodal import MultimodalType, WebPage
from app.logging_config import get_logger

logger = get_logger("argus.ingestion.web")


@dataclass(frozen=True)
class WebPageResult:
    """Result of web page ingestion."""
    url: str
    canonical_url: str
    title: str | None
    author: str | None
    published_date: datetime | None
    retrieved_date: datetime
    html_content: str
    text_content: str
    description: str | None
    keywords: list[str]
    metadata: dict[str, Any]


def _normalize_url(url: str) -> str:
    """Normalize URL for canonicalization."""
    parsed = urlparse(url)
    # Remove fragment, normalize scheme/host
    normalized = parsed._replace(fragment="", query="").geturl()
    # Ensure https
    if normalized.startswith("http://"):
        normalized = "https://" + normalized[7:]
    return normalized.rstrip("/")


def _extract_metadata(soup: BeautifulSoup, url: str) -> dict[str, Any]:
    """Extract metadata from HTML meta tags."""
    metadata = {}
    
    # Standard meta tags
    meta_tags = {
        "description": ["description", "og:description", "twitter:description"],
        "keywords": ["keywords", "news_keywords"],
        "author": ["author", "og:author", "article:author"],
        "published_date": ["article:published_time", "datePublished", "pubdate"],
        "modified_date": ["article:modified_time", "dateModified"],
        "title": ["og:title", "twitter:title"],
        "image": ["og:image", "twitter:image"],
        "site_name": ["og:site_name"],
        "type": ["og:type"],
    }
    
    for key, tag_names in meta_tags.items():
        for tag_name in tag_names:
            # Try property first, then name
            tag = soup.find("meta", property=tag_name) or soup.find("meta", attrs={"name": tag_name})
            if tag and tag.get("content"):
                metadata[key] = tag["content"]
                break
    
    # Extract JSON-LD structured data
    json_ld_scripts = soup.find_all("script", type="application/ld+json")
    for script in json_ld_scripts:
        try:
            import json
            data = json.loads(script.string)
            if isinstance(data, dict):
                metadata.setdefault("json_ld", []).append(data)
            elif isinstance(data, list):
                metadata.setdefault("json_ld", []).extend(data)
        except (json.JSONDecodeError, TypeError):
            pass
    
    return metadata


def _parse_date(date_str: str | None) -> datetime | None:
    """Parse date string from various formats."""
    if not date_str:
        return None
    
    # Try common formats
    formats = [
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M:%S",
    ]
    
    for fmt in formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            return dt
        except ValueError:
            continue
    
    # Try parsing with dateutil if available
    try:
        from dateutil import parser
        dt = parser.parse(date_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt
    except (ValueError, TypeError, ImportError):
        pass
    
    return None


def _extract_main_content(soup: BeautifulSoup) -> str:
    """Extract main content from HTML, removing navigation, ads, etc."""
    # Remove script and style elements
    for elem in soup(["script", "style", "nav", "header", "footer", "aside", "noscript"]):
        elem.decompose()
    
    # Try to find main content area
    main_selectors = [
        "main",
        "article",
        '[role="main"]',
        ".content",
        ".main-content",
        ".post-content",
        ".entry-content",
        "#content",
        "#main",
    ]
    
    content_elem = None
    for selector in main_selectors:
        content_elem = soup.select_one(selector)
        if content_elem:
            break
    
    if not content_elem:
        content_elem = soup.body or soup
    
    # Get text with reasonable spacing
    text = content_elem.get_text(separator="\n", strip=True)
    
    # Clean up excessive whitespace
    lines = [line.strip() for line in text.split("\n")]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def fetch_web_page(url: str, timeout: int = 30) -> WebPageResult:
    """Fetch and parse a web page with full metadata extraction.

    .. note::

       This function uses synchronous ``requests.get``.  The rest of the
       codebase is async (httpx in the LLM Gateway).  This is acceptable
       for now because web ingestion is invoked from the sync ingestion
       pipeline.  A future migration to httpx should be considered when
       async pipeline support is added.

    Returns WebPageResult with canonical URL, content, and metadata.
    """
    settings = get_settings()
    
    if not settings.multimodal_enabled or not settings.multimodal_web_ingestion_enabled:
        raise RuntimeError("Web ingestion disabled via configuration")
    
    # Normalize URL
    canonical_url = _normalize_url(url)
    
    # Fetch page
    headers = {
        "User-Agent": "ARGUS/1.0 (Research Agent; +https://argus.example.com/bot)",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }
    
    response = requests.get(canonical_url, headers=headers, timeout=timeout)
    response.raise_for_status()
    
    # Check content type
    content_type = response.headers.get("Content-Type", "")
    if "text/html" not in content_type.lower():
        raise ValueError(f"URL does not return HTML content: {content_type}")
    
    html_content = response.text
    retrieved_date = datetime.now(UTC)
    
    # Parse HTML
    soup = BeautifulSoup(html_content, "lxml")
    
    # Extract metadata
    metadata = _extract_metadata(soup, canonical_url)
    
    # Title
    title = None
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    elif "title" in metadata:
        title = metadata["title"]
    
    # Author
    author = metadata.get("author")
    
    # Published date
    published_date = _parse_date(metadata.get("published_date"))
    
    # Description
    description = metadata.get("description")
    
    # Keywords
    keywords = []
    if "keywords" in metadata:
        kw = metadata["keywords"]
        if isinstance(kw, str):
            keywords = [k.strip() for k in kw.split(",") if k.strip()]
        elif isinstance(kw, list):
            keywords = [str(k).strip() for k in kw if str(k).strip()]
    
    # Extract main text content
    text_content = _extract_main_content(soup)
    
    return WebPageResult(
        url=url,
        canonical_url=canonical_url,
        title=title,
        author=author,
        published_date=published_date,
        retrieved_date=retrieved_date,
        html_content=html_content,
        text_content=text_content,
        description=description,
        keywords=keywords,
        metadata=metadata,
    )


def web_page_to_multimodal(
    result: WebPageResult,
    document_id: UUID,
    source_chunk_ids: list[UUID] | None = None,
) -> WebPage:
    """Convert WebPageResult to Multimodal WebPage object for storage."""
    return WebPage(
        id=UUID(int=0),  # Will be assigned by store
        source_path=result.canonical_url,
        content_type=MultimodalType.WEB_PAGE,
        source_uri=result.canonical_url,
        source_chunk_ids=source_chunk_ids or [],
        url=result.url,
        canonical_url=result.canonical_url,
        title=result.title,
        author=result.author,
        published_date=result.published_date,
        retrieved_date=result.retrieved_date,
        html_content=result.html_content,
        text_content=result.text_content,
        description=result.description,
        keywords=result.keywords,
        metadata=result.metadata,
    )


def web_page_to_text_segments(result: WebPageResult) -> list[TextSegment]:
    """Convert web page content to text segments for chunking pipeline."""
    segments = []
    
    if not result.text_content.strip():
        return segments
    
    # For web pages, we typically treat the whole page as one segment
    # with the URL as section_path for provenance
    segments.append(TextSegment(
        text=result.text_content,
        page_start=1,
        page_end=1,
        char_start=0,
        char_end=len(result.text_content),
        section_path=result.canonical_url,
    ))
    
    return segments


def compute_web_page_checksum(url: str, html_content: str) -> str:
    """Compute checksum for web page based on canonical URL and content."""
    canonical = _normalize_url(url)
    combined = f"{canonical}\n{html_content}"
    return hashlib.sha256(combined.encode()).hexdigest()


def is_valid_web_url(url: str) -> bool:
    """Validate URL format and scheme."""
    try:
        parsed = urlparse(url)
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)
    except (ValueError, AttributeError):
        return False


__all__ = [
    "WebPageResult",
    "compute_web_page_checksum",
    "fetch_web_page",
    "is_valid_web_url",
    "web_page_to_multimodal",
    "web_page_to_text_segments",
]