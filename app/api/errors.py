"""Consistent JSON error envelope + exception handlers (Phase 00.2).

Every 4xx/5xx response returned by the API uses the same shape:

    {
      "error": {
        "code": "VALIDATION_ERROR",
        "message": "Request validation failed",
        "details": [...],
        "request_id": "req_abc123"
      }
    }

The `X-Request-ID` response header is set on error responses too (in
addition to the header added by `RequestIDMiddleware` on success paths),
since exception handlers registered for a broad `Exception` type run
inside Starlette's `ServerErrorMiddleware`, which sits outside our
custom middleware in the stack.
"""

from __future__ import annotations

from typing import Any

import structlog
from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from starlette.exceptions import HTTPException as StarletteHTTPException

# NOTE: we register against Starlette's base HTTPException, not
# `fastapi.HTTPException`. FastAPI's router raises the *Starlette* base
# class directly for automatic 404 / 405 responses, and `fastapi.HTTPException`
# is a subclass of it. Starlette's exception-handler lookup matches by
# walking the raised exception's MRO, so registering on the base class
# catches both the base-class instances (auto 404/405) and any
# `fastapi.HTTPException` raised explicitly by application code.

logger = structlog.get_logger("argus.errors")


class ErrorDetail(BaseModel):
    code: str
    message: str
    details: Any | None = None
    request_id: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail


def _request_id(request: Request) -> str | None:
    return getattr(request.state, "request_id", None)


def _error_response(
    request: Request,
    *,
    status_code: int,
    code: str,
    message: str,
    details: Any | None = None,
) -> JSONResponse:
    request_id = _request_id(request)
    body = ErrorResponse(
        error=ErrorDetail(code=code, message=message, details=details, request_id=request_id)
    )
    response = JSONResponse(status_code=status_code, content=body.model_dump())
    if request_id:
        response.headers["X-Request-ID"] = request_id
    return response


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    logger.warning("validation_error", errors=exc.errors())
    return _error_response(
        request,
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        code="VALIDATION_ERROR",
        message="Request validation failed",
        details=exc.errors(),
    )


async def http_exception_handler(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    logger.warning("http_exception", status_code=exc.status_code, detail=exc.detail)
    return _error_response(
        request,
        status_code=exc.status_code,
        code="HTTP_ERROR",
        message=str(exc.detail),
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("unhandled_exception")
    return _error_response(
        request,
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        code="INTERNAL_ERROR",
        message="An unexpected error occurred",
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Register the three exception handlers that produce the error envelope."""
    # mypy note: Starlette's add_exception_handler stub is invariant on
    # `Exception`, but Starlette actually dispatches by isinstance at
    # runtime, so a handler typed to a narrower exception subclass is
    # correct behavior that the stub can't express.
    app.add_exception_handler(RequestValidationError, validation_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)  # type: ignore[arg-type]
    app.add_exception_handler(Exception, unhandled_exception_handler)
