# vLLM Deployment Guide

This directory contains Kubernetes deployment configurations for vLLM models.

## Available Models

### 1. Llama-3.2-1B-Instruct
- **File**: `llama-3.2-1b.yaml`
- **Deployment Script**: `deploy-llama-3.2-1b.sh`
- **Status**: ✅ Deployed and running
- **Node**: `k3d-mycluster-gpu-agent-0`

### 2. Phi-tiny-MoE-instruct
- **File**: `phi-tiny-moe.yaml`
- **Deployment Script**: `deploy-phi-tiny-moe.sh`
- **Status**: ✅ Deployed and running
- **Node**: `k3d-agent-1-0-56098497` (separate agent node for isolation)

## Quick Start

### Deploy Llama-3.2-1B-Instruct

```bash
cd /path/to/coderepo/09/code/k3d/vllm
./deploy-llama-3.2-1b.sh
```

Or from the `k3d/` directory:
```bash
cd vllm/
./deploy-llama-3.2-1b.sh
```

### Deploy Phi-tiny-MoE-instruct

```bash
cd /path/to/coderepo/09/code/k3d/vllm
./deploy-phi-tiny-moe.sh
```

Or from the `k3d/` directory:
```bash
cd vllm/
./deploy-phi-tiny-moe.sh
```

## Deployment Details

### Phi-tiny-MoE-instruct Configuration

**Key Features:**
- Uses standard `vllm/vllm-openai:v0.12.0` image (same as llama)
- Configured for GPU support with `runtimeClassName: nvidia`
- Model mounted from `/raid/models/Phi-tiny-MoE-instruct` (host) → `/models/Phi-tiny-MoE-instruct` (container)
- Deployed to dedicated agent node `k3d-agent-1-0-56098497` via `nodeSelector`
- HF cache configured to use `/models/hub`

**Resource Requirements:**
- GPU: 1 GPU (with sufficient memory)
- Memory: 16Gi request, 32Gi limit
- GPU Memory Utilization: 90% (configurable)

**Prerequisites:**
- Model directory exists at `/raid/models/Phi-tiny-MoE-instruct` on host
- Additional agent node created with GPU support (see main README step 4)
- GPU available on target node

## Manual Deployment

### Phi-tiny-MoE-instruct

```bash
# Deploy model
kubectl apply -f phi-tiny-moe.yaml

# Wait for pod to be ready (model loading may take several minutes)
kubectl wait --for=condition=Ready pod -l app=vllm,model=phi-tiny-moe --timeout=600s

# Check status
kubectl get pod -l app=vllm,model=phi-tiny-moe -o wide
```

## Testing

### Direct Service Access

```bash
# Port forward
kubectl port-forward svc/vllm-phi-tiny-moe-service 9876:8000

# Test health
curl http://localhost:9876/health

# Test chat completion
curl http://localhost:9876/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/Phi-tiny-MoE-instruct",
    "messages": [{"role": "user", "content": "Which is more beautiful—the Gaussian integral or Euler formula, and why, in one sentence?"}],
    "max_tokens": 100
  }'
```

### Via API Gateway (Recommended)

If API Gateway is deployed, you can access all models through a single endpoint:

```bash
# Port forward Gateway
kubectl port-forward svc/vllm-api-gateway 8000:8000

# Test phi-tiny-moe (automatically routed)
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/Phi-tiny-MoE-instruct",
    "messages": [{"role": "user", "content": "Which is more beautiful—the Gaussian integral or Euler formula, and why, in one sentence?"}]
  }'
```

The Gateway automatically routes requests based on the `model` and `owned_by` fields in the request body.

**To deploy the Gateway:**
```bash
cd ../gateway
./deploy-gateway.sh
```

See `../gateway/README.md` for more details on Gateway deployment and configuration.

## Troubleshooting

### Pod Fails to Start

1. **Check GPU memory availability**:
   ```bash
   kubectl describe node k3d-agent-1-0-56098497 | grep -A 5 nvidia.com/gpu
   ```
   If GPU memory is insufficient, reduce `--gpu-memory-utilization` in `phi-tiny-moe.yaml` (default is 0.9).

2. **Check model path**:
   ```bash
   # Verify model exists in k3d container
   docker exec k3d-agent-1-0-56098497 ls -la /models/Phi-tiny-MoE-instruct
   ```

3. **Check pod logs**:
   ```bash
   kubectl logs -l app=vllm,model=phi-tiny-moe --tail=50
   ```

4. **Check pod events**:
   ```bash
   kubectl describe pod -l app=vllm,model=phi-tiny-moe | grep -A 10 Events
   ```

### Common Issues

- **GPU memory insufficient**: Error message shows "Free memory on device is less than desired GPU memory utilization". Solution: Reduce `--gpu-memory-utilization` or free up GPU memory.
- **Model path not found**: Ensure model is at `/raid/models/Phi-tiny-MoE-instruct` on host.
- **Node not ready**: Check if the agent node has GPU support and is ready: `kubectl get nodes`.

## File Structure

```
k3d/
├── vllm/                      # vLLM model deployments
│   ├── llama-3.2-1b.yaml          # Llama-3.2-1B-Instruct deployment
│   ├── deploy-llama-3.2-1b.sh     # Llama deployment script
│   ├── phi-tiny-moe.yaml          # Phi-tiny-MoE-instruct deployment
│   ├── deploy-phi-tiny-moe.sh     # Phi-tiny-MoE deployment script
│   ├── mistral-7b.yaml            # Mistral-7B deployment (if available)
│   └── README.md                  # This file
├── sglang/                    # SGLang model deployments
│   ├── llama-3.2-1b.yaml
│   └── deploy-llama-3.2-1b.sh
├── gateway/                   # API Gateway (supports both vLLM and SGLang)
│   ├── api-gateway.py             # Gateway source code
│   ├── api-gateway.yaml           # Gateway Pod/Service definitions
│   ├── deploy-gateway.sh          # Gateway deployment script
│   ├── access-gateway.sh          # Gateway access helper
│   ├── ingress*.yaml              # Ingress configurations
│   └── README.md                  # Gateway documentation
├── README.md                  # Main setup guide
└── TEST_GATEWAY.md            # Gateway testing guide
```

**Note:** API Gateway files are in the `gateway/` directory as they are generic and support both vLLM and SGLang models.

## Additional Resources

- **Main README**: `../README.md` - Complete setup guide from k3d cluster creation to vLLM deployment
- **API Gateway**: `../gateway/README.md` - Gateway deployment, configuration, and usage
- **Gateway Testing**: `../TEST_GATEWAY.md` - Comprehensive testing guide for the unified API Gateway
- **SGLang Deployments**: `../sglang/` - SGLang model deployments (similar structure to vLLM)
