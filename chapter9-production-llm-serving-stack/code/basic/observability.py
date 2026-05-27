"""
Observability for Production LLM Serving

Provides OpenTelemetry tracing, Prometheus metrics collection, and structured logging.

Usage:
    from observability import setup_tracing, MetricsCollector, StructuredLogger
"""

import time
import logging
import json
from datetime import datetime
from typing import Dict, Optional

# OpenTelemetry imports
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.resources import Resource
from opentelemetry.propagate import inject, extract

# Prometheus imports
from prometheus_client import Counter, Histogram, Gauge, start_http_server


def setup_tracing(service_name: str = "llm-serving"):
    """Setup OpenTelemetry tracing"""
    resource = Resource.create({"service.name": service_name})

    provider = TracerProvider(resource=resource)
    processor = BatchSpanProcessor(
        OTLPSpanExporter(endpoint="http://localhost:4317", insecure=True)
    )
    provider.add_span_processor(processor)

    trace.set_tracer_provider(provider)

    return trace.get_tracer(__name__)


class MetricsCollector:
    """Prometheus metrics collector for LLM serving"""

    def __init__(self):
        # Request metrics
        self.request_count = Counter(
            "llm_requests_total",
            "Total number of requests",
            ["model", "status"],
        )

        self.request_latency = Histogram(
            "llm_request_latency_seconds",
            "Request latency in seconds",
            ["model"],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
        )

        self.active_requests = Gauge(
            "llm_active_requests",
            "Number of active requests",
            ["model"],
        )

        # Resource metrics
        self.gpu_utilization = Gauge(
            "llm_gpu_utilization_percent",
            "GPU utilization percentage",
            ["gpu_id"],
        )

    def record_request_start(self, model: str):
        """Record request start"""
        self.active_requests.labels(model=model).inc()

    def record_request_end(self, model: str, latency: float, success: bool = True):
        """Record request completion"""
        status = "success" if success else "error"
        self.request_count.labels(model=model, status=status).inc()
        self.request_latency.labels(model=model).observe(latency)
        self.active_requests.labels(model=model).dec()

    def update_gpu_utilization(self, gpu_id: int, utilization: float):
        """Update GPU utilization metric"""
        self.gpu_utilization.labels(gpu_id=str(gpu_id)).set(utilization)

    def start_server(self, port: int = 9090):
        """Start Prometheus metrics server"""
        start_http_server(port)


class StructuredLogger:
    """Structured JSON logger for LLM serving"""

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)

        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(message)s")
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def log_request(
        self, request_id: str, model: str, latency_ms: float, error: bool = False
    ):
        """Log request with structured format"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": "ERROR" if error else "INFO",
            "request_id": request_id,
            "model": model,
            "latency_ms": latency_ms,
            "error": error,
        }
        self.logger.info(json.dumps(log_entry))

    def log_metric(self, metric_name: str, value: float, tags: Optional[Dict] = None):
        """Log metric"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "type": "metric",
            "metric": metric_name,
            "value": value,
            "tags": tags or {},
        }
        self.logger.info(json.dumps(log_entry))


if __name__ == "__main__":
    # Test structured logger
    logger = StructuredLogger("llm-serving")
    logger.log_request("req-123", "llama-2-7b", 150.5)
    logger.log_metric("gpu_memory_used", 0.85, {"gpu_id": "0"})

    # Test metrics collector
    metrics = MetricsCollector()
    metrics.record_request_start("llama-2-7b")
    time.sleep(0.1)
    metrics.record_request_end("llama-2-7b", 0.1, success=True)
    print("Metrics recorded successfully")
