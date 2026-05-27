# Basic Production LLM Serving Components

This directory contains the fundamental building blocks for a production LLM serving system.

## Files

| File | Description |
|------|-------------|
| `tokenizer_service.py` | Stateless tokenizer service with FastAPI |
| `model_runner.py` | GPU-backed model inference with vLLM |
| `api_gateway.py` | API gateway with routing and rate limiting |
| `routing.py` | Multi-model routing strategies (feature-based, A/B, dynamic) |
| `load_balancer.py` | Load balancing algorithms (round-robin, least connections, weighted) |
| `health_check.py` | Endpoint health monitoring |
| `canary.py` | Canary deployments and A/B testing framework |
| `observability.py` | OpenTelemetry tracing, Prometheus metrics, structured logging |
| `fault_tolerance.py` | Cold start handling, autoscaling, request queuing, cost optimization |

## Architecture

```
┌─────────────┐
│   Clients   │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ API Gateway │ (api_gateway.py)
│  - Routing  │ (routing.py)
│  - Rate Limiting
│  - Load Balancing (load_balancer.py)
└──────┬──────┘
       │
       ├──────────────┬──────────────┐
       ▼              ▼              ▼
┌──────────┐   ┌──────────┐   ┌──────────┐
│Tokenizer │   │ Model    │   │ Model    │
│ Service  │   │ Runner 1 │   │ Runner 2 │
└──────────┘   └──────────┘   └──────────┘
       │              │              │
       └──────────────┴──────────────┘
                      │
                      ▼
              ┌──────────────┐
              │ Observability│ (observability.py)
              │ & Monitoring │
              └──────────────┘
```

## Quick Start

### 1. Start Tokenizer Service
```bash
uvicorn tokenizer_service:app --host 0.0.0.0 --port 8001
```

### 2. Start Model Runner
```bash
python model_runner.py
```

### 3. Start API Gateway
```bash
uvicorn api_gateway:app --host 0.0.0.0 --port 8000
```

## Dependencies

```
fastapi
uvicorn
transformers
vllm
httpx
prometheus-client
opentelemetry-api
opentelemetry-sdk
opentelemetry-exporter-otlp
opentelemetry-instrumentation-fastapi
opentelemetry-instrumentation-httpx
```

## Usage Examples

### Routing
```python
from routing import FeatureBasedRouter, ABRouter

router = FeatureBasedRouter()
model = router.route({"prompt": "Write a function to sort"})
# Returns: "code-llama-7b"
```

### Load Balancing
```python
from load_balancer import RoundRobinBalancer

endpoints = ["http://localhost:8001", "http://localhost:8002"]
balancer = RoundRobinBalancer(endpoints)
endpoint = balancer.get_endpoint()
```

### Canary Deployment
```python
from canary import CanaryDeployment

canary = CanaryDeployment("llama-2-7b", "llama-2-13b", traffic_percent=0.1)
model = canary.route({})
canary.record_metrics(model, latency=0.15)
```

### Observability
```python
from observability import MetricsCollector, StructuredLogger

metrics = MetricsCollector()
metrics.record_request_start("llama-2-7b")
# ... process request ...
metrics.record_request_end("llama-2-7b", latency=0.15, success=True)

logger = StructuredLogger("llm-serving")
logger.log_request("req-123", "llama-2-7b", 150.5)
```
