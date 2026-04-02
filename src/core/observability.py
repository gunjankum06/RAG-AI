"""Optional OpenTelemetry tracing for Arize/Phoenix observability."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractContextManager, nullcontext
from typing import Any

from src.core.config import settings
from src.core.logging import logger

_provider = None
_tracer = None


def setup_observability() -> bool:
    """Configure OTLP trace exporter when Arize observability is enabled."""
    global _provider, _tracer

    if not settings.arize_enabled:
        logger.info("Arize observability disabled")
        return False

    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        logger.warning(
            "Arize observability enabled but OpenTelemetry packages are missing. "
            "Install opentelemetry-sdk and opentelemetry-exporter-otlp."
        )
        return False

    headers: dict[str, str] = {}
    if settings.arize_api_key:
        headers["api_key"] = settings.arize_api_key
    if settings.arize_space_key:
        headers["space_key"] = settings.arize_space_key

    resource = Resource.create(
        {
            "service.name": settings.arize_service_name,
            "service.version": "2.2.0",
            "deployment.environment": settings.environment,
            "arize.project_name": settings.arize_project_name,
        }
    )

    exporter = OTLPSpanExporter(
        endpoint=settings.arize_otlp_endpoint,
        headers=headers or None,
    )
    _provider = TracerProvider(resource=resource)
    _provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(_provider)
    _tracer = trace.get_tracer("rag_ai")

    logger.info(
        "Arize observability enabled (endpoint=%s, project=%s)",
        settings.arize_otlp_endpoint,
        settings.arize_project_name,
    )
    return True


def start_span(
    name: str,
    attributes: Mapping[str, Any] | None = None,
) -> AbstractContextManager[Any]:
    """Start a tracing span if observability is configured, else no-op."""
    if _tracer is None:
        return nullcontext(None)
    return _tracer.start_as_current_span(name, attributes=dict(attributes or {}))


def shutdown_observability() -> None:
    """Flush and shutdown the configured tracer provider, if any."""
    global _provider
    if _provider is None:
        return
    try:
        _provider.shutdown()
    except Exception:
        logger.warning("Failed to shutdown observability provider cleanly")
