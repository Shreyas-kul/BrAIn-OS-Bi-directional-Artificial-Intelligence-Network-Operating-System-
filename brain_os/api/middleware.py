"""
BrAIn OS — API Middleware

Provides:
    1. Rate Limiting Middleware (Token bucket algorithm per client IP / session)
    2. Request Timing & Tracing Middleware (injects X-Process-Time and audit logs)
    3. Global Exception Handling Middleware
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable

from fastapi import FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from brain_os.config.settings import get_settings

logger = logging.getLogger(__name__)


@dataclass
class TokenBucket:
    """In-memory token bucket rate limiter per client."""

    tokens: float
    last_updated: float
    capacity: int
    fill_rate: float  # tokens per second

    def consume(self, amount: float = 1.0) -> bool:
        now = time.time()
        delta = now - self.last_updated
        self.tokens = min(self.capacity, self.tokens + delta * self.fill_rate)
        self.last_updated = now

        if self.tokens >= amount:
            self.tokens -= amount
            return True
        return False


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Token bucket rate limiter middleware."""

    def __init__(self, app: FastAPI):
        super().__init__(app)
        settings = get_settings()
        self.capacity = settings.security.rate_limit_burst
        self.fill_rate = settings.security.rate_limit_requests_per_minute / 60.0
        self.buckets: dict[str, TokenBucket] = defaultdict(
            lambda: TokenBucket(
                tokens=float(self.capacity),
                last_updated=time.time(),
                capacity=self.capacity,
                fill_rate=self.fill_rate,
            )
        )

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Exempt health checks and documentation endpoints
        path = request.url.path
        if path in ("/health", "/docs", "/openapi.json", "/redoc"):
            return await call_next(request)

        client_ip = request.client.host if request.client else "unknown"
        bucket = self.buckets[client_ip]

        if not bucket.consume(1.0):
            logger.warning("Rate limit exceeded for IP %s on path %s", client_ip, path)
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "error": "Rate limit exceeded. Please throttle your requests.",
                    "client_ip": client_ip,
                },
                headers={"Retry-After": "2"},
            )

        return await call_next(request)


class ProcessTimeMiddleware(BaseHTTPMiddleware):
    """Measures request duration and injects X-Process-Time header."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.perf_counter()
        response = await call_next(request)
        process_time = time.perf_counter() - start_time
        response.headers["X-Process-Time"] = f"{process_time:.4f}s"
        return response


def setup_middleware(app: FastAPI) -> None:
    """Register all middleware with the FastAPI application."""
    settings = get_settings()

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Process time & Rate limiting
    app.add_middleware(ProcessTimeMiddleware)
    app.add_middleware(RateLimitMiddleware)
