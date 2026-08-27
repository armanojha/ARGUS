"""Request ID middleware (Phase 00.2).

Generates or propagates an `X-Request-ID` header for every request, binds
it into `structlog`'s contextvars so every log emitted while handling the
request automatically includes `request_id`, and stashes it on
`request.state.request_id` so exception handlers can include it in the
error envelope.
"""

from __future__ import annotations

import uuid

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
