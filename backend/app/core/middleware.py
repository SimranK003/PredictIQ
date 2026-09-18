"""Request-scoped context: request IDs, structured access logging, and
generic HTTP metrics — applied to every request regardless of endpoint.

Deliberately does not log request/response bodies: prediction inputs are
customer attributes, not secrets, but there's no reason to duplicate them
into logs when they're already persisted in the predictions table for
exactly the callers who need them (audit, debugging, drift analysis).
"""

import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from app.core.logging import get_logger
from monitoring.metrics import HTTP_REQUEST_LATENCY_SECONDS, HTTP_REQUESTS_TOTAL

logger = get_logger("access")


class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        start = time.perf_counter()

        try:
            response = await call_next(request)
        except Exception:
            latency_ms = (time.perf_counter() - start) * 1000
            logger.exception(
                "request_failed",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "latency_ms": round(latency_ms, 2),
                },
            )
            raise

        latency_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Request-ID"] = request_id

        route = request.scope.get("route")
        endpoint_label = route.path if route is not None else request.url.path

        HTTP_REQUESTS_TOTAL.labels(
            method=request.method, endpoint=endpoint_label, status_code=str(response.status_code)
        ).inc()
        HTTP_REQUEST_LATENCY_SECONDS.labels(method=request.method, endpoint=endpoint_label).observe(
            latency_ms / 1000
        )

        logger.info(
            "request_completed",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "latency_ms": round(latency_ms, 2),
            },
        )
        return response


def get_request_id(request: Request) -> uuid.UUID:
    return uuid.UUID(request.state.request_id)
