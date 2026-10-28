"""Typed errors: every error response tells a client what to do next.

    {"detail": ..., "error": "<type>", "retryable": <bool>, ...}

`detail` is unchanged (FastAPI's field, which existing clients read). `error`
is a stable machine-readable type and `retryable` says whether the SAME
request may be sent again to the SAME server:

    retry          429 overloaded, 503 unavailable / write_failed /
                   namespace_loading / namespace_held (honour Retry-After)
    go elsewhere   421 wrong_shard (body names owner_shard)
    your fault     every other 4xx: sending it again cannot succeed
    our fault      500 internal, 501 not_implemented, 502 upstream_error

`retryable` describes the server's state, not the request's safety: a client
retries only requests that are idempotent (GET, PUT, DELETE, a read-only POST
such as /search, or any write carrying an Idempotency-Key). Blindly retrying a
POST /vectors on 503 can apply it twice; blindly retrying anything on a 500
turns one bad request into a retry storm.
"""
from typing import Optional

from fastapi import HTTPException

# Default type per status. A raise site can be more specific (APIError).
_TYPES = {
    400: "invalid_request", 401: "unauthorized", 403: "forbidden", 404: "not_found",
    405: "method_not_allowed", 409: "conflict", 412: "precondition_failed",
    413: "too_large", 415: "unsupported_media_type", 421: "wrong_shard",
    422: "invalid_request", 429: "overloaded", 500: "internal",
    501: "not_implemented", 502: "upstream_error", 503: "unavailable", 504: "timeout",
}
RETRYABLE = {429, 503}


class APIError(HTTPException):
    """An HTTPException with an explicit error type (and extra body fields)."""

    def __init__(self, status_code: int, detail, error: Optional[str] = None,
                 retryable: Optional[bool] = None, headers: Optional[dict] = None, **extra):
        super().__init__(status_code=status_code, detail=detail, headers=headers)
        self.error = error
        self.retryable = retryable
        self.extra = extra


def error_type(status: int) -> str:
    return _TYPES.get(status, "client_error" if 400 <= status < 500 else "server_error")


def body(status: int, detail, error: Optional[str] = None,
         retryable: Optional[bool] = None, **extra) -> dict:
    return {"detail": detail, "error": error or error_type(status),
            "retryable": status in RETRYABLE if retryable is None else retryable, **extra}


def of(exc: HTTPException) -> dict:
    return body(exc.status_code, exc.detail, getattr(exc, "error", None),
                getattr(exc, "retryable", None), **getattr(exc, "extra", {}))
