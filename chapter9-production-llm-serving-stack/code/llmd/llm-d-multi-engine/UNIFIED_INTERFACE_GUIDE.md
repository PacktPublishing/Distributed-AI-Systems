# Unified Interface Guide for llm-d-multi-engine

## Overview

The `llm-d-multi-engine` cluster provides a **unified interface** that routes requests to different inference engines (vLLM and SGLang) based on the `model` and `owned_by` fields in the request.

## Architecture

The cluster includes:
- **vLLM ModelService**: Serves Qwen2.5-0.5B-Instruct using vLLM engine
- **SGLang ModelService**: Serves Qwen2.5-0.5B-Instruct using SGLang engine
- **Custom API Gateway**: Routes requests based on both `model` and `owned_by` fields
- **InferencePool Gateway**: llm-d's built-in gateway (routes by model only)

## Using the Unified Interface

### Option 1: Custom API Gateway (Recommended - Supports owned_by routing)

The Custom API Gateway (`engine-comparison-gateway`) provides advanced routing that supports both `model` and `owned_by` fields.

#### Access via Custom API Gateway

```bash
export NAMESPACE=llm-d-multi-engine

# Port-forward to Custom API Gateway
GATEWAY_POD=$(kubectl get pods -n ${NAMESPACE} -l app=engine-comparison-gateway | grep Running | tail -1 | awk '{print $1}')
kubectl port-forward -n ${NAMESPACE} ${GATEWAY_POD} 54321:8000

# Or use the service (if port-forward works)
# kubectl port-forward -n ${NAMESPACE} svc/engine-comparison-gateway 54321:8000
```

#### 1. List All Available Models

```bash
curl http://localhost:54321/v1/models
```

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "Qwen/Qwen2.5-0.5B-Instruct",
      "object": "model",
      "created": 1766189481,
      "owned_by": "sglang",
      "root": "Qwen/Qwen2.5-0.5B-Instruct",
      "parent": null,
      "max_model_len": 32768
    },
    {
      "id": "Qwen/Qwen2.5-0.5B-Instruct",
      "object": "model",
      "created": 1766189481,
      "owned_by": "vllm",
      "root": "Qwen/Qwen2.5-0.5B-Instruct",
      "parent": null,
      "max_model_len": 32768
    }
  ]
}
```

#### 2. Chat Completion with vLLM (using owned_by in request body)

```bash
curl http://localhost:54321/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "owned_by": "vllm",
    "messages": [
      {"role": "user", "content": "Hello! Say hi"}
    ],
    "max_tokens": 30
  }'
```

**Response:**
```json
{
  "id": "chatcmpl-c1df477cb37142f9851a7758866db839",
  "object": "chat.completion",
  "created": 1766189945,
  "model": "Qwen/Qwen2.5-0.5B-Instruct",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I assist you today?",
        "refusal": null,
        "annotations": null,
        "audio": null,
        "function_call": null,
        "tool_calls": [],
        "reasoning": null,
        "reasoning_content": null
      },
      "logprobs": null,
      "finish_reason": "stop",
      "stop_reason": null,
      "token_ids": null
    }
  ],
  "service_tier": null,
  "system_fingerprint": null,
  "usage": {
    "prompt_tokens": 33,
    "total_tokens": 43,
    "completion_tokens": 10,
    "prompt_tokens_details": null
  },
  "prompt_logprobs": null,
  "prompt_token_ids": null,
  "kv_transfer_params": null
}
```

#### 3. Chat Completion with SGLang (using owned_by in request body)

```bash
curl http://localhost:54321/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "owned_by": "sglang",
    "messages": [
      {"role": "user", "content": "Hello! Say hi"}
    ],
    "max_tokens": 30
  }'
```

Response:
```json
{
  "id": "4945821cb2a44aed85d989cf8ea1325f",
  "object": "chat.completion",
  "created": 1766189957,
  "model": "Qwen/Qwen2.5-0.5B-Instruct",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I assist you today?",
        "reasoning_content": null,
        "tool_calls": null
      },
      "logprobs": null,
      "finish_reason": "stop",
      "matched_stop": 151645
    }
  ],
  "usage": {
    "prompt_tokens": 33,
    "total_tokens": 43,
    "completion_tokens": 10,
    "prompt_tokens_details": null,
    "reasoning_tokens": 0
  },
  "metadata": {
    "weight_version": "default"
  }
}

```


#### 4. Chat Completion with owned_by in HTTP Header

```bash
curl http://localhost:54321/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "x-owned-by: vllm" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [
      {"role": "user", "content": "What is machine learning?"}
    ],
    "max_tokens": 50
  }'
```

```json
{
  "id": "chatcmpl-731d20cf13f74b1fa31b9d1928e747a1",
  "object": "chat.completion",
  "created": 1766190096,
  "model": "Qwen/Qwen2.5-0.5B-Instruct",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Machine learning is a subset of artificial intelligence (AI) that involves the development and application of algorithms to enable computer systems to learn from data without being explicitly programmed. The goal of machine learning is to enable computers to make predictions or decisions based on patterns and",
        "refusal": null,
        "annotations": null,
        "audio": null,
        "function_call": null,
        "tool_calls": [],
        "reasoning": null,
        "reasoning_content": null
      },
      "logprobs": null,
      "finish_reason": "length",
      "stop_reason": null,
      "token_ids": null
    }
  ],
  "service_tier": null,
  "system_fingerprint": null,
  "usage": {
    "prompt_tokens": 34,
    "total_tokens": 84,
    "completion_tokens": 50,
    "prompt_tokens_details": null
  },
  "prompt_logprobs": null,
  "prompt_token_ids": null,
  "kv_transfer_params": null
}
```


#### 5. Admin API - Service Discovery

```bash
# Discover available services (requires POST method)
curl -X POST http://localhost:54321/admin/discover
```

**Response:**
```json
{
  "message": "Service discovery completed",
  "discovered": 2,
  "services": {
    "Qwen/Qwen2.5-0.5B-Instruct::sglang": "infra-sglang-qwen2-5-0-5b-inference-gateway-istio",
    "Qwen/Qwen2.5-0.5B-Instruct::vllm": "infra-vllm-qwen2-5-0-5b-inference-gateway-istio"
  }
}
```

**Key Features:**
- ✅ **Unified Interface**: Single endpoint (`/v1/chat/completions`) for all models and engines
- ✅ **Advanced Routing**: Routes based on both `model` and `owned_by` fields
- ✅ **Multiple Input Sources**: Reads `owned_by` from request body, HTTP header (`x-owned-by`), or query parameters
- ✅ **Model Discovery**: Aggregates model lists from all ModelServices (`/v1/models`)
- ✅ **Admin API**: Dynamic routing configuration management (`/admin/discover`)
- ✅ **Fast Fail**: No fallback - fails fast if gateway service is not available

### Option 2: InferencePool Gateway (Basic Unified Interface)

The InferencePool Gateway provides a unified interface that routes based on `model` field only (does not support `owned_by` routing).

#### Access via InferencePool Gateway

```bash
export NAMESPACE=llm-d-multi-engine

# Port-forward to vLLM InferencePool Gateway
kubectl port-forward -n ${NAMESPACE} svc/infra-vllm-qwen2-5-0-5b-inference-gateway-istio 54321:80

# Or SGLang InferencePool Gateway
# kubectl port-forward -n ${NAMESPACE} svc/infra-sglang-qwen2-5-0-5b-inference-gateway-istio 54322:80
```

#### List Available Models

```bash
curl http://localhost:54321/v1/models
```

**⚠️ Known Issue:** The InferencePool Gateway returns 404 "route_not_found" because it requires two key resources that are not automatically created:

1. **InferencePool CRD instance** - Defines which ModelService pods belong to the pool
2. **HTTPRoute resource** - Configures how to route requests to the InferencePool

**Why InferencePool Gateway is llm-d's strength:**
- ✅ Intelligent routing and load balancing
- ✅ Prefix-cache awareness
- ✅ Automatic service discovery
- ✅ Production-grade reliability

However, it requires proper configuration of InferencePool and HTTPRoute resources to work.

**Current Status:** The helmfile deployment creates the InferencePool Gateway and Extension, but doesn't automatically create the InferencePool CRD instance and HTTPRoute. These need to be created manually or configured in the helmfile values.

**Recommended Solution:** Use the **Custom API Gateway (Option 1)** instead, which:
- Provides direct pod access with automatic fallback
- Works even when InferencePool Gateway services are unavailable
- Supports routing by both `model` and `owned_by` fields
- Doesn't require InferencePool/HTTPRoute configuration

#### Chat Completion (routes to vLLM by default)

```bash
curl http://localhost:54321/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [
      {"role": "user", "content": "What is machine learning?"}
    ],
    "max_tokens": 100
  }'
```

**Key Features:**
- ✅ **Unified Interface**: Single endpoint (`/v1/chat/completions`) for all models
- ✅ **Automatic Routing**: Routes based on `model` field in request body
- ✅ **Model Discovery**: Aggregates model lists from all ModelServices (`/v1/models`)
- ⚠️ **Limitation**: Only routes by `model` field, not by `owned_by` (inference engine type)

## Comparison: Unified Interface Options

| Feature | InferencePool Gateway | Custom API Gateway |
|---------|----------------------|-------------------|
| **Unified Interface** | ✅ | ✅ |
| **Single Endpoint** | ✅ | ✅ |
| **Route by `model`** | ✅ | ✅ |
| **Route by `owned_by`** | ❌ | ✅ |
| **Header-based routing** | ❌ | ✅ |
| **Admin API** | ❌ | ✅ |
| **Auto Service Discovery** | ✅ | ✅ |
| **Fast Fail** | N/A | ✅ |

**Recommendation:**
- Use **InferencePool Gateway** if you only need routing by `model` field (simpler, production-ready)
- Use **Custom API Gateway** if you need routing by both `model` and `owned_by` fields (for engine comparison/testing)

## Testing Examples

### Example 1: Compare vLLM vs SGLang Performance

```bash
# Test vLLM
time curl -s -X POST http://localhost:54321/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "owned_by": "vllm",
    "messages": [{"role": "user", "content": "Explain quantum computing"}],
    "max_tokens": 100
  }' > vllm_response.json

# Test SGLang
time curl -s -X POST http://localhost:54321/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "owned_by": "sglang",
    "messages": [{"role": "user", "content": "Explain quantum computing"}],
    "max_tokens": 100
  }' > sglang_response.json
```

### Example 2: Using HTTP Header for Routing

```bash
# Route to vLLM using header
curl http://localhost:54321/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "x-owned-by: vllm" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 20
  }'

# Route to SGLang using header
curl http://localhost:54321/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "x-owned-by: sglang" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 20
  }'
```

### Example 3: Direct Access from Within Cluster

```bash
export NAMESPACE=llm-d-multi-engine

# Get Custom API Gateway IP
GATEWAY_IP=$(kubectl get svc -n ${NAMESPACE} engine-comparison-gateway -o jsonpath='{.spec.clusterIP}')
echo "Custom API Gateway IP: ${GATEWAY_IP}"

# List Available Models
curl http://${GATEWAY_IP}:8000/v1/models

# Request to vLLM
curl http://${GATEWAY_IP}:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "owned_by": "vllm",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 20
  }'

# Request to SGLang
curl http://${GATEWAY_IP}:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "owned_by": "sglang",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 20
  }'
```

## Troubleshooting

### Port-forward Connection Issues

**Note:** Port-forward to localhost may sometimes return "Empty reply from server". If this happens, use cluster-internal access instead (see "Direct Access from Within Cluster" section below).

If port-forward fails, try:
1. Use pod port-forward instead of service:
   ```bash
   GATEWAY_POD=$(kubectl get pods -n llm-d-multi-engine -l app=engine-comparison-gateway | grep Running | tail -1 | awk '{print $1}')
   kubectl port-forward -n llm-d-multi-engine ${GATEWAY_POD} 54321:8000
   ```

2. **Alternative: Use cluster-internal access** (recommended if port-forward doesn't work):
   ```bash
   export NAMESPACE=llm-d-multi-engine
   GATEWAY_IP=$(kubectl get svc -n ${NAMESPACE} engine-comparison-gateway -o jsonpath='{.spec.clusterIP}')
   # Access from within cluster using a temporary pod
   kubectl run -n ${NAMESPACE} --rm -i --restart=Never test-curl --image=curlimages/curl:latest -- curl http://${GATEWAY_IP}:8000/v1/models
   ```

3. Check Gateway pod logs:
   ```bash
   kubectl logs -n llm-d-multi-engine -l app=engine-comparison-gateway --tail=50
   ```

4. Verify Gateway pod is running:
   ```bash
   kubectl get pods -n llm-d-multi-engine -l app=engine-comparison-gateway
   ```

### Service Discovery Issues

If routing fails, check service discovery:
```bash
# Note: /admin/discover requires POST method
curl -X POST http://localhost:54321/admin/discover
```

If services are not discovered, verify:
1. ModelService pods are running:
   ```bash
   kubectl get pods -n llm-d-multi-engine -l llm-d.ai/role=decode
   ```

2. InferencePool Gateway services exist:
   ```bash
   kubectl get svc -n llm-d-multi-engine | grep inference-gateway-istio
   ```

## Notes

- The Custom API Gateway uses **fast fail** mode: if the InferencePool Gateway service is not available, it will return an error instead of falling back to direct pod access.
- Metrics collection frequency is set to **60 minutes** to reduce system load.
- Both vLLM and SGLang serve the same model (`Qwen/Qwen2.5-0.5B-Instruct`) but use different inference engines for performance comparison.
- **Port-forward limitations**: Port-forward to localhost may sometimes return "Empty reply from server". If this happens, use cluster-internal access methods (see "Direct Access from Within Cluster" section) or ensure port-forward is properly established before making requests.
- **Admin API**: The `/admin/discover` endpoint requires a **POST** request, not GET.
- **Model Listing**: The `/v1/models` endpoint now correctly returns all models from all inference engines (vLLM and SGLang), even if InferencePool Gateway services are unavailable. Models are distinguished by their `owned_by` field.
