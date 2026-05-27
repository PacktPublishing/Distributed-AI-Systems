## Hands-On Examples: LLM serving in Kubernetes with k3d

This section demonstrates two practical examples of deploying LLM serving systems in Kubernetes with API Gateway routing:

1. **Example 1**: Deploying different models using the same inference engine (vLLM)
2. **Example 2**: Deploying the same model using different inference engines (vLLM vs SGLang)

### How Routing Works

This is the final routing solution that supports routing based on both model and inference engine. The gateway routes requests using both the `model` field and the `owned_by` field.

1. **Client sends request** to API Gateway with both `model` and `owned_by` fields:
   ```json
   {
     "model": "meta-llama/Llama-3.2-1B-Instruct",
     "owned_by": "sglang",
     "messages": [...]
   }
   ```

2. **Gateway parses request** and extracts both the `model` field and the `owned_by` field from the JSON body.

3. **Gateway looks up service** using both fields:
   - `("meta-llama/Llama-3.2-1B-Instruct", "vllm")` → `vllm-llama-32-1b`
   - `("meta-llama/Llama-3.2-1B-Instruct", "sglang")` → `sglang-llama-32-1b`
   - `("meta-llama/Llama-3.2-1B-Instruct", None)` → `vllm-llama-32-1b` (default)

4. **Gateway forwards request** to the target service based on the inference engine specified.

5. **Gateway returns response** to client (with streaming support if requested).

### Example 1: Different Models, Same Inference Engine (vLLM)

This example demonstrates how to deploy multiple vLLM models in a Kubernetes cluster and use an API Gateway to provide a unified interface that automatically routes requests to the correct model based on the `model` field in the request.

**Architecture:**

```
┌─────────────────────────────────────────────────────────────┐
│                    Client Applications                      │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            │ HTTP Request with 'model' field
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              API Gateway (Unified Entry Point)              │
│  - Parses 'model' field from request body                   │
│  - Routes to appropriate vLLM service                       │
│  - Returns response to client                               │
└───────────────┬───────────────────────────┬─────────────────┘
                │                           │
                │                           │
    ┌───────────▼──────────┐    ┌───────────▼─────────┐
    │  vLLM Service 1      │    │  vLLM Service 2     │
    │  (Llama-3.2-1B)      │    │  (Phi-tiny-MoE)     │
    │                      │    │                     │
    │ Pod: vllm-llama-32-1b│    │  Pod: vllm-phi-tiny │
    │ Service: vllm-llama- │    │  Service: vllm-phi- │
    │    32-1b:8000        │    │    tiny-moe:8000    │
    └──────────────────────┘    └─────────────────────┘
```

This diagram shows two vLLM services serving different models:
- **vLLM Service 1**: Serves Llama-3.2-1B-Instruct (owned_by: "vllm")
- **vLLM Service 2**: Serves Phi-tiny-MoE-instruct (owned_by: "vllm")

Both use the same inference engine (vLLM) but serve different models. The gateway routes based on the `model` field in the request.

#### Step 1: Deploy First Model (Llama-3.2-1B-Instruct)

**File:** `code/vllm/llama-3.2-1b.yaml`

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: vllm-llama-32-1b
  labels:
    app: vllm
    model: llama-32-1b
spec:
  runtimeClassName: nvidia
  containers:
  - name: vllm-server
    image: vllm/vllm-openai:v0.12.0
    command:
    - python3
    - -m
    - vllm.entrypoints.openai.api_server
    args:
    - --model
    - meta-llama/Llama-3.2-1B-Instruct
    - --host
    - "0.0.0.0"
    - --port
    - "8000"
    - --tensor-parallel-size
    - "1"
    - --gpu-memory-utilization
    - "0.9"
    env:
    - name: HF_TOKEN
      valueFrom:
        secretKeyRef:
          name: hf-token-secret
          key: token
    resources:
      limits:
        nvidia.com/gpu: 1
        memory: 8Gi
      requests:
        nvidia.com/gpu: 1
        memory: 6Gi
---
apiVersion: v1
kind: Service
metadata:
  name: vllm-llama-32-1b
spec:
  type: ClusterIP
  selector:
    app: vllm
    model: llama-32-1b
  ports:
  - port: 8000
    targetPort: 8000
```

**Deploy:**

```bash
# Create HuggingFace token secret
kubectl create secret generic hf-token-secret \
  --from-literal=token="$HF_TOKEN"

# Deploy model
kubectl apply -f code/vllm/llama-3.2-1b.yaml

# Wait for pod to be ready
kubectl wait --for=condition=Ready pod/vllm-llama-32-1b --timeout=300s
```

#### Step 2: Deploy Second Model (Phi-tiny-MoE-instruct)

**File:** `code/vllm/phi-tiny-moe.yaml`

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: vllm-phi-tiny-moe
  labels:
    app: vllm
    model: phi-tiny-moe
spec:
  replicas: 1
  selector:
    matchLabels:
      app: vllm
      model: phi-tiny-moe
  template:
    metadata:
      labels:
        app: vllm
        model: phi-tiny-moe
    spec:
      runtimeClassName: nvidia
      containers:
      - name: vllm-server
        image: vllm/vllm-openai:v0.12.0
        command:
        - python3
        - -m
        - vllm.entrypoints.openai.api_server
        args:
        - --model
        - /models/Phi-tiny-MoE-instruct
        - --host
        - "0.0.0.0"
        - --port
        - "8000"
        - --tensor-parallel-size
        - "1"
        - --gpu-memory-utilization
        - "0.9"
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: 32Gi
          requests:
            nvidia.com/gpu: 1
            memory: 16Gi
        volumeMounts:
        - name: models
          mountPath: /models
      volumes:
      - name: models
        hostPath:
          path: /models
---
apiVersion: v1
kind: Service
metadata:
  name: vllm-phi-tiny-moe-service
spec:
  type: ClusterIP
  selector:
    app: vllm
    model: phi-tiny-moe
  ports:
  - port: 8000
    targetPort: 8000
```

**Deploy:**

```bash
# Deploy model
kubectl apply -f code/vllm/phi-tiny-moe.yaml

# Wait for pod to be ready
kubectl wait --for=condition=Ready pod -l app=vllm,model=phi-tiny-moe --timeout=600s
```

#### Step 3: Deploy API Gateway

The API Gateway provides a unified entry point that automatically routes requests to the correct vLLM service based on the `model` field in the request body.

**File:** `code/api-gateway.yaml`

**Key Features:**
- **Automatic Routing**: Routes requests based on `model` field in request body
- **Unified Interface**: Single endpoint for all models (`/v1/chat/completions`)
- **Model Discovery**: Aggregates model lists from all services (`/v1/models`)
- **Streaming Support**: Supports streaming responses

**Deploy:**

```bash
# Deploy API Gateway
kubectl apply -f code/api-gateway.yaml

# Wait for gateway to be ready
kubectl wait --for=condition=Ready pod/vllm-api-gateway --timeout=60s
```

#### Step 4: Using the Unified Interface

Once deployed, all requests go through the API Gateway, which automatically routes to the correct model service.

**1. List Available Models:**

```bash
# Port forward gateway
kubectl port-forward svc/vllm-api-gateway 8000:8000

# List all models
curl http://localhost:8000/v1/models
```

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "meta-llama/Llama-3.2-1B-Instruct",
      "object": "model",
      "created": 0,
      "owned_by": "vllm"
    },
    {
      "id": "/models/Phi-tiny-MoE-instruct",
      "object": "model",
      "created": 0,
      "owned_by": "vllm"
    }
  ]
}
```

**2. Chat Completion with Llama Model:**

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "messages": [
      {"role": "user", "content": "What is machine learning?"}
    ],
    "max_tokens": 100
  }'
```

**3. Chat Completion with Phi-tiny-MoE Model:**

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/Phi-tiny-MoE-instruct",
    "messages": [
      {"role": "user", "content": "What is a mixture of experts?"}
    ],
    "max_tokens": 100
  }'
```

**4. Streaming Response:**

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "messages": [
      {"role": "user", "content": "Explain quantum computing"}
    ],
    "stream": true
  }'
```

### Example 2: Same Model, Different Inference Engines (vLLM vs SGLang)

This example demonstrates how to deploy the same model (Llama-3.2-1B-Instruct) using different inference engines (vLLM and SGLang) and use an API Gateway to route requests based on both the `model` field and the `owned_by` field.

**Architecture:**

```
┌─────────────────────────────────────────────────────────────┐
│                    Client Applications                      │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            │ HTTP Request with 'model' and 
                            │ 'owned_by' fields
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              API Gateway (Unified Entry Point)              │
│  - Parses 'model' field from request body                   │
│  - Parses 'owned_by' field                                  │
│  - Routes to appropriate service based on both fields       │
│  - Returns response to client                               │
└───────────────┬───────────────────────────┬─────────────────┘
                │                           │
                │                           │
    ┌───────────▼──────────┐    ┌───────────▼─────────┐
    │  vLLM Service        │    │  SGLang Service     │
    │  (Llama-3.2-1B)      │    │  (Llama-3.2-1B)     │
    │                      │    │                     │
    │ Pod: vllm-llama-32-1b│    │  Pod: sglang-llama- │
    │ Service: vllm-llama- │    │    32-1b            │
    │    32-1b:8000        │    │  Service: sglang-   │
    │                      │    │    llama-32-1b:8000 │
    │ owned_by: "vllm"     │    │  owned_by: "sglang" │
    │                      │    │                     │
    │ Image: vllm/vllm-    │    │  Image: lmsysorg/   │
    │   openai:v0.12.0     │    │    sglang:v0.5.6    │
    └──────────────────────┘    └─────────────────────┘
```

**Routing Logic:**

1. **Request with vLLM engine:**
   ```json
   {
     "model": "meta-llama/Llama-3.2-1B-Instruct",
     "owned_by": "vllm",
     "messages": [...]
   }
   ```
   → Routes to `vllm-llama-32-1b:8000`

2. **Request with SGLang engine:**
   ```json
   {
     "model": "meta-llama/Llama-3.2-1B-Instruct",
     "owned_by": "sglang",
     "messages": [...]
   }
   ```
   → Routes to `sglang-llama-32-1b:8000`

3. **Request without owned_by (defaults to vLLM):**
   ```json
   {
     "model": "meta-llama/Llama-3.2-1B-Instruct",
     "messages": [...]
   }
   ```
   → Routes to `vllm-llama-32-1b:8000` (default)

**Key Differences:**
- **Same Model**: Both services serve `meta-llama/Llama-3.2-1B-Instruct`
- **Different Engines**: vLLM vs SGLang (different inference engines)
- **Different owned_by**: "vllm" vs "sglang" (used for routing)
- **Different Images**: Different container images for each engine
- **Different Performance**: Each engine has different characteristics (latency, throughput, etc.)

This setup allows clients to:
- Compare performance between inference engines
- A/B test different engines
- Route based on workload characteristics (e.g., use SGLang for structured generation, vLLM for chat)

#### Step 1: Deploy vLLM Service for Llama-3.2-1B-Instruct

Deploy the vLLM service serving Llama-3.2-1B-Instruct. This is the same deployment as in Example 1, Step 1.

**File:** `code/vllm/llama-3.2-1b.yaml`

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: vllm-llama-32-1b
  labels:
    app: vllm
    model: llama-32-1b
spec:
  runtimeClassName: nvidia
  containers:
  - name: vllm-server
    image: vllm/vllm-openai:v0.12.0
    command:
    - python3
    - -m
    - vllm.entrypoints.openai.api_server
    args:
    - --model
    - meta-llama/Llama-3.2-1B-Instruct
    - --host
    - "0.0.0.0"
    - --port
    - "8000"
    - --tensor-parallel-size
    - "1"
    - --gpu-memory-utilization
    - "0.9"
    env:
    - name: HF_TOKEN
      valueFrom:
        secretKeyRef:
          name: hf-token-secret
          key: token
    resources:
      limits:
        nvidia.com/gpu: 1
        memory: 8Gi
      requests:
        nvidia.com/gpu: 1
        memory: 6Gi
---
apiVersion: v1
kind: Service
metadata:
  name: vllm-llama-32-1b
spec:
  type: ClusterIP
  selector:
    app: vllm
    model: llama-32-1b
  ports:
  - port: 8000
    targetPort: 8000
```

**Deploy:**

```bash
# Create HuggingFace token secret (if not already created)
kubectl create secret generic hf-token-secret \
  --from-literal=token="$HF_TOKEN"

# Deploy model
kubectl apply -f code/vllm/llama-3.2-1b.yaml

# Wait for pod to be ready
kubectl wait --for=condition=Ready pod/vllm-llama-32-1b --timeout=300s
```

#### Step 2: Deploy SGLang Service for Llama-3.2-1B-Instruct

Deploy the SGLang service serving the same model (Llama-3.2-1B-Instruct) but using a different inference engine.

**File:** `code/sglang/llama-3.2-1b.yaml`

```yaml
apiVersion: v1
kind: Pod
metadata:
  name: sglang-llama-32-1b
  labels:
    app: sglang
    model: llama-32-1b
spec:
  runtimeClassName: nvidia
  containers:
  - name: sglang-server
    image: lmsysorg/sglang:v0.5.6.post2-runtime
    command:
    - python3
    - -m
    - sglang.launch_server
    args:
    - --model-path
    - meta-llama/Llama-3.2-1B-Instruct
    - --host
    - "0.0.0.0"
    - --port
    - "8000"
    - --trust-remote-code
    env:
    - name: HF_TOKEN
      valueFrom:
        secretKeyRef:
          name: hf-token-secret
          key: token
    resources:
      limits:
        nvidia.com/gpu: 1
        memory: 8Gi
      requests:
        nvidia.com/gpu: 1
        memory: 6Gi
---
apiVersion: v1
kind: Service
metadata:
  name: sglang-llama-32-1b
spec:
  type: ClusterIP
  selector:
    app: sglang
    model: llama-32-1b
  ports:
  - port: 8000
    targetPort: 8000
```

**Deploy:**

```bash
# Deploy SGLang service
kubectl apply -f code/sglang/llama-3.2-1b.yaml

# Wait for pod to be ready
kubectl wait --for=condition=Ready pod/sglang-llama-32-1b --timeout=300s
```

#### Step 3: Update API Gateway Configuration

The API Gateway needs to be configured to route based on both `model` and `owned_by` fields. The gateway code already supports this routing logic (see `code/api-gateway.py`).

**Routing Configuration:**

The gateway uses the following routing logic:
- `("meta-llama/Llama-3.2-1B-Instruct", "vllm")` → `vllm-llama-32-1b`
- `("meta-llama/Llama-3.2-1B-Instruct", "sglang")` → `sglang-llama-32-1b`
- `("meta-llama/Llama-3.2-1B-Instruct", None)` → `vllm-llama-32-1b` (default)

**Deploy/Update Gateway:**

```bash
# Deploy or update API Gateway
kubectl apply -f code/api-gateway.yaml

# Wait for gateway to be ready
kubectl wait --for=condition=Ready pod/vllm-api-gateway --timeout=60s
```

#### Step 4: Test the Unified Interface

Test routing to both inference engines using the `owned_by` field.

**1. List Available Models:**

```bash
# Port forward gateway
kubectl port-forward svc/vllm-api-gateway 8000:8000

# List all models (should show both vllm and sglang versions)
curl http://localhost:8000/v1/models
```

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "meta-llama/Llama-3.2-1B-Instruct",
      "object": "model",
      "created": 0,
      "owned_by": "vllm"
    },
    {
      "id": "meta-llama/Llama-3.2-1B-Instruct",
      "object": "model",
      "created": 0,
      "owned_by": "sglang"
    }
  ]
}
```

**2. Request with vLLM Engine:**

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "owned_by": "vllm",
    "messages": [
      {"role": "user", "content": "What is machine learning?"}
    ],
    "max_tokens": 100
  }'
```

**3. Request with SGLang Engine:**

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "owned_by": "sglang",
    "messages": [
      {"role": "user", "content": "What is machine learning?"}
    ],
    "max_tokens": 100
  }'
```

**4. Request without owned_by (defaults to vLLM):**

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "messages": [
      {"role": "user", "content": "What is machine learning?"}
    ],
    "max_tokens": 100
  }'
```

### Production Considerations

**1. Expose Gateway Externally:**

For production, expose the gateway using Ingress:

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: vllm-api-gateway-ingress
spec:
  rules:
  - host: api.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: vllm-api-gateway
            port:
              number: 8000
```

**2. Add Authentication:**

Add API key authentication to the gateway:

```python
API_KEYS = {"client-1": "key-abc123", "client-2": "key-def456"}

@app.middleware("http")
async def authenticate(request: Request, call_next):
    api_key = request.headers.get("X-API-Key")
    if not api_key or api_key not in API_KEYS.values():
        raise HTTPException(401, "Invalid API key")
    return await call_next(request)
```

**3. Add Rate Limiting:**

Implement per-client rate limiting:

```python
from collections import defaultdict
import time

rate_limiter = defaultdict(list)

@app.middleware("http")
async def rate_limit(request: Request, call_next):
    client_id = request.headers.get("X-Client-ID", "default")
    now = time.time()
    # Clean old requests
    rate_limiter[client_id] = [t for t in rate_limiter[client_id] if now - t < 60]
    if len(rate_limiter[client_id]) >= 100:
        raise HTTPException(429, "Rate limit exceeded")
    rate_limiter[client_id].append(now)
    return await call_next(request)
```

**4. Add Monitoring:**

Add Prometheus metrics:

```python
from prometheus_client import Counter, Histogram

request_count = Counter('gateway_requests_total', 'Total requests', ['model', 'status'])
request_latency = Histogram('gateway_request_duration_seconds', 'Request latency')

@app.middleware("http")
async def metrics(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = time.time() - start
    model = extract_model_from_request(request)
    request_count.labels(model=model, status=response.status_code).inc()
    request_latency.observe(duration)
    return response
```

