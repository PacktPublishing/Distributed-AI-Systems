# Routing and Comparison Testing Guide

## 1. Resource Contention and Race Condition Analysis

### ✅ Resource isolation in the current configuration

**GPU isolation (fully isolated)**:
- vLLM: `NVIDIA_VISIBLE_DEVICES=0` -> uses GPU 0
- SGLang: `NVIDIA_VISIBLE_DEVICES=1` -> uses GPU 1
- ✅ **No GPU contention**: each service uses a dedicated GPU

**Memory isolation (partially isolated)**:
- vLLM: `--gpu-memory-utilization=0.1` -> uses 10% of GPU 0 memory
- SGLang: `--mem-fraction-static=0.1` -> uses 10% of GPU 1 memory
- ✅ **GPU memory is fully isolated**: different GPUs, no contention
- ⚠️ **System memory is shared**: both pods share node system memory (but each pod has limits: 8Gi)

**CPU resources (shared but limited)**:
- ⚠️ **CPU is shared**: both pods share node CPU
- ✅ **Resource limits exist**: each pod has CPU requests/limits if configured
- 💡 **Suggestion**: add CPU limits if you see CPU contention

**Disk I/O (shared)**:
- ⚠️ **Model storage is shared**: both pods access the `/models` directory
- ✅ **Read-only access**: model files are read-only, so there are no write conflicts
- ⚠️ **Cache writes**: HuggingFace cache writes to `/models/hub`, which may cause slight contention

**Network bandwidth (shared)**:
- ⚠️ **Network is shared**: both services share node network bandwidth
- 💡 **Impact is small**: for inference services, network bandwidth is usually not the bottleneck

### Race Condition Analysis

**❌ There will be no race condition**:
- The two pods are **independent processes** and do not share memory space
- Each service runs in its own container with a separate process space
- The model files are **read-only**, so there are no write conflicts
- GPU memory is fully isolated, so there is no memory contention

**Possible contention points**:
1. **CPU contention**: if both services are heavily loaded at the same time
2. **System memory contention**: if both services load large amounts of data into system memory
3. **Disk I/O contention**: if both services read model files at the same time (though models are usually already loaded into GPU memory)

### Fairness of the Comparison Test

**✅ The current setup is suitable for comparison testing**:
- Same hardware environment (same node)
- GPU isolation is complete (different GPUs)
- Same model (Qwen2.5-0.5B-Instruct)
- Same GPU memory utilization (both at 10%)

**⚠️ Variables to watch**:
- CPU contention may affect results (but can be controlled with CPU limits)
- System memory contention (usually a minor effect)
- Network latency (same node, so the effect is very small)

## 2. Routing Options

### Option 1: Direct Service Access (Simplest)

Each LLMInferenceService automatically creates a Kubernetes Service:

```bash
# Access vLLM
kubectl port-forward svc/vllm-qwen2-5-0-5b 8001:8000
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [{"role": "user", "content": "Hello"}]
  }'

# Access SGLang
kubectl port-forward svc/sglang-qwen2-5-0-5b 8002:8000
curl http://localhost:8002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

### Option 2: Route Through an API Gateway (Recommended for Comparison Tests)

Create an API Gateway that routes to different services based on the `inference_server` field in the request.

**Advantages**:
- A single entry point
- Engine selection through request parameters
- Easy A/B testing

**Implementation**: see `api-gateway.yaml` below

### Option 3: Use llm-d InferencePool (If Using Full llm-d Features)

If you deploy llm-d InferencePool, routing can be done through the `owned_by` label.

## 3. Example Comparison Test Script

```bash
#!/bin/bash
# benchmark-comparison.sh

# Test parameters
MODEL="Qwen/Qwen2.5-0.5B-Instruct"
NUM_REQUESTS=100
CONCURRENT=10

# Test vLLM
echo "Testing vLLM..."
kubectl port-forward svc/vllm-qwen2-5-0-5b 8001:8000 &
VLLM_PF=$!
sleep 2

time for i in $(seq 1 $NUM_REQUESTS); do
  curl -s http://localhost:8001/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"$MODEL\", \"messages\": [{\"role\": \"user\", \"content\": \"Test $i\"}]}" \
    > /dev/null &
  if [ $((i % $CONCURRENT)) -eq 0 ]; then
    wait
  fi
done
wait

kill $VLLM_PF

# Test SGLang
echo "Testing SGLang..."
kubectl port-forward svc/sglang-qwen2-5-0-5b 8002:8000 &
SGLANG_PF=$!
sleep 2

time for i in $(seq 1 $NUM_REQUESTS); do
  curl -s http://localhost:8002/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"$MODEL\", \"messages\": [{\"role\": \"user\", \"content\": \"Test $i\"}]}" \
    > /dev/null &
  if [ $((i % $CONCURRENT)) -eq 0 ]; then
    wait
  fi
done
wait

kill $SGLANG_PF
```

## 4. Optimization Recommendations

### If You See CPU Contention

Add CPU limits in LLMInferenceService:

```yaml
resources:
  requests:
    nvidia.com/gpu: 1
    memory: 6Gi
    cpu: "2"  # CPU request
  limits:
    nvidia.com/gpu: 1
    memory: 8Gi
    cpu: "4"  # CPU limit
```

### If You See System Memory Contention

Increase memory limits or reduce request concurrency.

### Monitor Resource Usage

```bash
# Monitor node resources
kubectl top node

# Monitor pod resources
kubectl top pod -l app=vllm
kubectl top pod -l app=sglang

# Monitor GPU usage
nvidia-smi
```
