# Testing vLLM API Gateway

## Quick Start

### 1. Port Forward (if not already running)

```bash
# Check if port-forward is running
lsof -i :8000

# Start port-forward (background)
kubectl port-forward svc/vllm-api-gateway 8000:8000 > /tmp/port-forward.log 2>&1 &

# Or run in foreground (Ctrl+C to stop)
kubectl port-forward svc/vllm-api-gateway 8000:8000
```

### 2. Test Health Endpoint

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
    "status": "healthy",
    "service": "llm-api-gateway"
}
```

### 3. List Available Models

```bash
curl http://localhost:8000/v1/models | jq
```

Expected response:
```json
[
    {
        "id": "meta-llama/Llama-3.2-1B-Instruct",
        "object": "model",
        "created": 1766430923,
        "owned_by": "vllm",
        "root": "meta-llama/Llama-3.2-1B-Instruct",
        "parent": null,
        "max_model_len": 131072
    },
    {
        "id": "meta-llama/Llama-3.2-1B-Instruct",
        "object": "model",
        "created": 1766430923,
        "owned_by": "sglang",
        "root": "meta-llama/Llama-3.2-1B-Instruct",
        "parent": null,
        "max_model_len": 131072
    }
]
```

### 4. Test Chat Completion (vLLM)

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "owned_by": "vllm",
    "messages": [
      {"role": "user", "content": "Hello! Say hi"}
    ],
    "max_tokens": 30
  }' | jq
```

**Note:** The `owned_by` field routes to the specific inference engine:
- `"owned_by": "vllm"` → routes to `vllm-llama-32-1b` service
- `"owned_by": "sglang"` → routes to `sglang-llama-32-1b` service
- If `owned_by` is omitted, defaults to `vllm`

### 5. Test Chat Completion (SGLang)

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "owned_by": "sglang",
    "messages": [
      {"role": "user", "content": "Hello! Say hi"}
    ],
    "max_tokens": 30
  }' | jq
```

### 6. Test Default Routing (no owned_by)

```bash
# Without owned_by, defaults to vLLM
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "max_tokens": 30
  }' | jq
```

### 7. View Routing Configuration

```bash
curl http://localhost:8000/admin/api/routing | jq
```

Expected response:
```json
[
    {
        "model": "meta-llama/Llama-3.2-1B-Instruct",
        "inference_server": "vllm",
        "service_name": "vllm-llama-32-1b"
    },
    {
        "model": "meta-llama/Llama-3.2-1B-Instruct",
        "inference_server": null,
        "service_name": "vllm-llama-32-1b"
    },
    {
        "model": "meta-llama/Llama-3.2-1B-Instruct",
        "inference_server": "sglang",
        "service_name": "sglang-llama-32-1b"
    }
]
```

## Using the Test Script

You can also use the provided test script:

```bash
cd /home/wukong/workspace/coderepo/09/code/k3d
./test-api.sh http://localhost:8000
```

## Routing Logic

The Gateway routes requests based on:
1. **`model`** field (required): The model identifier
2. **`owned_by`** field (optional, preferred): `"vllm"` or `"sglang"`
   - Also supports `engine` or `inference_server` for backward compatibility
   - If omitted, defaults to `vllm`

## Troubleshooting

### Port 8000 already in use

```bash
# Find the process
lsof -i :8000

# Kill it
kill <PID>

# Or use a different port
kubectl port-forward svc/vllm-api-gateway 8001:8000
# Then use http://localhost:8001
```

### Pod not ready

```bash
# Check pod status
kubectl get pods

# Check logs
kubectl logs vllm-api-gateway
kubectl logs vllm-llama-32-1b
kubectl logs sglang-llama-32-1b
```

### Gateway returns 404 or 502

- Ensure the target service (vllm-llama-32-1b or sglang-llama-32-1b) is running and ready
- Check that the `model` field matches the routing configuration
- Verify the `owned_by` field is correct (`"vllm"` or `"sglang"`)
