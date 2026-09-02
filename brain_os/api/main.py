"""
BrAIn OS — FastAPI Application Entry Point & CLI Runner

Starts the FastAPI server hosting REST & WebSocket gateways.
Entry point for CLI: `brain-os`
"""

from __future__ import annotations

import contextlib
import logging
import uvicorn
from fastapi import FastAPI

from brain_os.api.middleware import setup_middleware
from brain_os.api.routes import router
from brain_os.config.settings import get_settings
from brain_os.observability import get_logger, setup_logging, setup_tracer

logger = get_logger("brain_os.api")


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for BrAIn OS server startup & shutdown."""
    setup_logging()
    setup_tracer()
    settings = get_settings()
    logger.info(
        "🧠 BrAIn OS Kernel Gateway starting up on %s:%s (LLM: %s)",
        settings.api.host,
        settings.api.port,
        settings.llm.default_backend,
    )
    yield
    logger.info("🧠 BrAIn OS Kernel Gateway shutting down cleanly...")


def create_app() -> FastAPI:
    """Application factory for BrAIn OS."""
    settings = get_settings()

    app = FastAPI(
        title="BrAIn OS Gateway",
        description="Bi-directional Artificial Intelligence Network Operating System API",
        version="0.1.0",
        lifespan=lifespan,
    )

    # Attach middleware (CORS, Rate Limiting, Request Timing)
    setup_middleware(app)

    # Attach routes
    app.include_router(router)

    return app


app = create_app()


def run() -> None:
    """CLI entry point executed by `brain-os` command."""
    settings = get_settings()
    uvicorn.run(
        "brain_os.api.main:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=False,
        log_level=settings.observability.log_level.lower(),
    )


if __name__ == "__main__":
    run()
