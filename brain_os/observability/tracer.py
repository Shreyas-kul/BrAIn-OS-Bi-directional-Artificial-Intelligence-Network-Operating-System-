"""
BrAIn OS — OpenTelemetry Distributed Tracing

Provides tracing across the request lifecycle:
    User Request → Input Guard → MoE Router → Scheduler → Agent Execution → Output Guard
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any, Iterator

from brain_os.config.settings import get_settings

logger = logging.getLogger(__name__)

_tracer = None
_initialized = False


def setup_tracer(service_name: str = "brain-os") -> Any:
    """Initialize OpenTelemetry tracer provider."""
    global _tracer, _initialized
    if _initialized:
        return _tracer

    settings = get_settings()

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        resource = Resource.create({"service.name": settings.observability.otel_service_name or service_name})
        provider = TracerProvider(resource=resource)

        if settings.observability.otel_exporter_endpoint:
            try:
                import socket
                from urllib.parse import urlparse
                parsed = urlparse(settings.observability.otel_exporter_endpoint)
                host = parsed.hostname or "localhost"
                port = parsed.port or 4317
                with socket.create_connection((host, port), timeout=0.1):
                    pass

                exporter = OTLPSpanExporter(
                    endpoint=settings.observability.otel_exporter_endpoint,
                    insecure=True,
                )
                provider.add_span_processor(BatchSpanProcessor(exporter))
            except Exception as e:
                logger.debug("OTLP collector not reachable (%s); continuing in local mode", e)

        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer("brain-os-tracer")
        logger.info("OpenTelemetry tracing initialized for service %s", service_name)

    except ImportError:
        logger.debug("OpenTelemetry not installed; tracing running in no-op mode")
        _tracer = None

    _initialized = True
    return _tracer


def get_tracer() -> Any:
    """Return the global tracer instance."""
    global _tracer
    if not _initialized:
        setup_tracer()
    return _tracer


@contextlib.contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[Any]:
    """Context manager for tracing operations with optional attributes."""
    tracer = get_tracer()
    if tracer is not None:
        with tracer.start_as_current_span(name) as span:
            if attributes:
                for k, v in attributes.items():
                    span.set_attribute(k, str(v))
            yield span
    else:
        # No-op dummy span
        class NoOpSpan:
            def set_attribute(self, key: str, value: Any) -> None:
                pass

        yield NoOpSpan()
