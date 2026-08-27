"""Retrieval API endpoint (Phase 01).

Provides query → ranked evidence with citations.
"""

from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field

from app.config import Settings, get_settings
from app.evidence.models import EvidenceRef
from app.reranking import get_reranker
from app.retrieval.hybrid import get_hybrid_retriever

router = APIRouter(prefix="/api/v1", tags=["retrieval"])


class RetrievalRequest(BaseModel):
    """Request for retrieval endpoint."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, max_length=2000, description="Search query")
    top_k: int | None = Field(default=None, ge=1, le=50, description="Number of results to return")
    bm25_weight: float = Field(default=0.5, ge=0.0, le=1.0, description="BM25 weight in hybrid fusion")
    vector_weight: float = Field(default=0.5, ge=0.0, le=1.0, description="Vector weight in hybrid fusion")
    use_reranker: bool = Field(default=True, description="Whether to apply cross-encoder reranking")
    mode: str = Field(default="hybrid", pattern="^(hybrid|bm25|vector)$", description="Retrieval mode")


class Citation(BaseModel):
    """Citation for a retrieved evidence chunk."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: UUID
    document_id: UUID
    source_id: UUID
    source_path: str
    source_type: str
    text: str
    page_start: int | None = None
    page_end: int | None = None
    section_path: str | None = None
    score: float
    rank: int
    metadata: dict = Field(default_factory=dict)


class RetrievalResponse(BaseModel):
    """Response for retrieval endpoint."""

    model_config = ConfigDict(extra="forbid")

    query: str
    citations: list[Citation]
    total_candidates: int
    mode: str
    reranked: bool


def _evidence_ref_to_citation(ref: EvidenceRef) -> Citation:
    return Citation(
        chunk_id=ref.chunk_id,
        document_id=ref.document_id,
        source_id=ref.source_id,
        source_path=ref.source_path,
        source_type=ref.source_type.value,
        text=ref.text,
        page_start=ref.page_start,
        page_end=ref.page_end,
        section_path=ref.section_path,
        score=ref.score,
        rank=ref.rank,
        metadata=ref.metadata,
    )


@router.post("/retrieve", response_model=RetrievalResponse)
async def retrieve(
    request: RetrievalRequest,
    settings: Settings = Depends(get_settings),
) -> RetrievalResponse:
    """Retrieve evidence chunks with citations for a query.

    Supports hybrid (BM25 + vector), BM25-only, and vector-only modes.
    Optionally applies cross-encoder reranking.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    retriever = get_hybrid_retriever()

    # Ensure indexes are built
    retriever.ensure_indexes()

    # Perform retrieval
    if request.mode == "bm25":
        results = retriever.search_bm25_only(request.query, top_k=request.top_k)
    elif request.mode == "vector":
        results = retriever.search_vector_only(request.query, top_k=request.top_k)
    else:
        results = retriever.search(
            request.query,
            top_k=request.top_k,
            bm25_weight=request.bm25_weight,
            vector_weight=request.vector_weight,
        )

    # Apply reranking if requested
    reranked = False
    if request.use_reranker and results:
        reranker = get_reranker()
        results = reranker.rerank(request.query, results, top_k=request.top_k)
        reranked = True

    citations = [_evidence_ref_to_citation(ref) for ref in results]

    return RetrievalResponse(
        query=request.query,
        citations=citations,
        total_candidates=len(citations),
        mode=request.mode,
        reranked=reranked,
    )


@router.get("/retrieve", response_model=RetrievalResponse)
async def retrieve_get(
    query: str = Query(..., min_length=1, max_length=2000),
    top_k: int | None = Query(default=None, ge=1, le=50),
    bm25_weight: float = Query(default=0.5, ge=0.0, le=1.0),
    vector_weight: float = Query(default=0.5, ge=0.0, le=1.0),
    use_reranker: bool = Query(default=True),
    mode: str = Query(default="hybrid", pattern="^(hybrid|bm25|vector)$"),
    settings: Settings = Depends(get_settings),
) -> RetrievalResponse:
    """GET endpoint for retrieval (convenience for simple queries)."""
    request = RetrievalRequest(
        query=query,
        top_k=top_k,
        bm25_weight=bm25_weight,
        vector_weight=vector_weight,
        use_reranker=use_reranker,
        mode=mode,
    )
    return await retrieve(request, settings)