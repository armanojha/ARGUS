"""Request ID middleware (Phase 00.2).

Generates or propagates an `X-Request-ID` header for every request, binds
it into `structlog`'s contextvars so every log emitted while handling the
request automatically includes `request_id`, and stashes it on
`request.state.request_id` so exception handlers can include it in the
error envelope.
"""

from __future__ import annotations

import time
import uuid
from collections import defaultdict

import structlog
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

REQUEST_ID_HEADER = "X-Request-ID"


def generate_request_id() -> str:
    return f"req_{uuid.uuid4().hex}"


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Ensure every request has a request ID, in logs, state, and response headers."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        incoming = request.headers.get(REQUEST_ID_HEADER)
        if incoming and len(incoming) <= 128:
            request_id = incoming
        else:
            request_id = generate_request_id()
        request.state.request_id = request_id

        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(request_id=request_id)

        logger = structlog.get_logger("argus.request")
        logger.info("request_started", method=request.method, path=request.url.path)

        try:
            response = await call_next(request)
        except Exception:
            logger.exception("request_failed", method=request.method, path=request.url.path)
            raise

        response.headers[REQUEST_ID_HEADER] = request_id
        logger.info(
            "request_finished",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
        )
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiter for expensive endpoints.

    Limits requests per client IP to prevent abuse. Uses a sliding window
    counter with automatic cleanup of expired entries.
    """

    def __init__(
        self,
        app: any,
        requests_per_minute: int = 30,
        burst_limit: int = 10,
    ) -> None:
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.burst_limit = burst_limit
        self._requests: dict[str, list[float]] = defaultdict(list)
        self._last_cleanup = time.time()

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from request, respecting X-Forwarded-For."""
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def _cleanup_old_entries(self) -> None:
        """Remove entries older than 1 minute."""
        now = time.time()
        if now - self._last_cleanup < 60:
            return

        cutoff = now - 60
        for ip in list(self._requests.keys()):
            self._requests[ip] = [t for t in self._requests[ip] if t > cutoff]
            if not self._requests[ip]:
                del self._requests[ip]
        self._last_cleanup = now

    def _is_rate_limited(self, client_ip: str) -> bool:
        """Check if client is rate limited."""
        now = time.time()
        cutoff = now - 60

        # Cleanup old entries periodically
        self._cleanup_old_entries()

        # Get recent requests for this IP
        recent = self._requests[client_ip]
        self._requests[client_ip] = [t for t in recent if t > cutoff]

        # Check burst limit (short window)
        burst_window = now - 5
        burst_count = sum(1 for t in self._requests[client_ip] if t > burst_window)
        if burst_count >= self.burst_limit:
            return True

        # Check rate limit (per minute)
        if len(self._requests[client_ip]) >= self.requests_per_minute:
            return True

        # Record this request
        self._requests[client_ip].append(now)
        return False

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        # Only rate-limit expensive endpoints
        if request.url.path in ("/api/v1/query", "/api/v1/query/stream"):
            client_ip = self._get_client_ip(request)
            if self._is_rate_limited(client_ip):
                logger = structlog.get_logger("argus.ratelimit")
                logger.warning("rate_limited", client_ip=client_ip, path=request.url.path)
                from app.api.errors import ApplicationHTTPException, ErrorCode
                raise ApplicationHTTPException(
                    status_code=429,
                    code=ErrorCode.RATE_LIMITED,
                    message="Rate limit exceeded. Please wait before making another request.",
                )

        return await call_next(request)


class TimeoutMiddleware(BaseHTTPMiddleware):
    """Request timeout middleware for long-running queries.

    Applies a timeout to expensive endpoints to prevent hung requests.
    Uses asyncio.wait_for to enforce the timeout.
    """

    def __init__(self, app: any, timeout_seconds: float = 120.0) -> None:
        super().__init__(app)
        self.timeout_seconds = timeout_seconds

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        import asyncio

        # Only apply timeout to expensive endpoints
        if request.url.path in ("/api/v1/query", "/api/v1/query/stream"):
            try:
                response = await asyncio.wait_for(
                    call_next(request),
                    timeout=self.timeout_seconds,
                )
                return response
            except TimeoutError:
                logger = structlog.get_logger("argus.timeout")
                logger.warning("request_timeout", path=request.url.path, timeout=self.timeout_seconds)
                from app.api.errors import ApplicationHTTPException, ErrorCode
                raise ApplicationHTTPException(
                    status_code=504,
                    code=ErrorCode.TIMEOUT,
                    message=f"Request timed out after {self.timeout_seconds} seconds",
                )

        return await call_next(request)
