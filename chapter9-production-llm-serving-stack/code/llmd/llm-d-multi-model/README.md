# llm-d Multi-Model Deployment

This directory demonstrates multi-model serving with llm-d, deploying two models on vLLM:
- **Llama-3.2-1B-Instruct** (gated model, requires HF_TOKEN)
- **Qwen2.5-0.5B-Instruct** (open model)

## Quick Start

```bash
# Set HuggingFace token (required for Llama)
export HF_TOKEN='your_token_here'

# Deploy both models
./deploy.sh

# Test the deployment
./test.sh
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    k3d Cluster                          │
├─────────────────────────────────────────────────────────┤
│                                                         │
│  ┌──────────────────┐    ┌──────────────────┐           │
│  │  vLLM            │    │  vLLM            │           │
│  │  Llama-3.2-1B    │    │  Qwen2.5-0.5B    │           │
│  │  (Port 8000)     │    │  (Port 8000)     │           │
│  └────────┬─────────┘    └────────┬─────────┘           │
│           │                       │                     │
│           └───────────┬───────────┘                     │
│                       │                                 │
│              ┌────────▼────────┐                        │
│              │  K8s Services   │                        │
│              └─────────────────┘                        │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

## Version Information

This example uses **vLLM v0.14.1**, which is the version bundled with **llm-d v0.5.0** (released February 2026). vLLM and llm-d are actively developed projects with frequent releases. You may need to update the image versions in `deploy.sh` and `vllm-pod.yaml` to match newer releases.

To use a different vLLM version:
```bash
export VLLM_IMAGE="vllm/vllm-openai:v0.15.0"  # or newer
./deploy.sh
```

Check for latest versions:
- vLLM: https://hub.docker.com/r/vllm/vllm-openai/tags
- llm-d: https://github.com/llm-d/llm-d/releases

## Prerequisites

- k3d installed
- kubectl configured
- Docker with GPU support
- HuggingFace token (for Llama model)

## Manual Deployment

If you prefer to deploy step by step:

```bash
# 1. Create cluster
k3d cluster create llmd-cluster --gpus=all

# 2. Install NVIDIA device plugin
kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.1/nvidia-device-plugin.yml

# 3. Create HF token secret
kubectl create secret generic hf-token-secret --from-literal=token="$HF_TOKEN"

# 4. Deploy Llama
kubectl apply -f vllm-pod.yaml

# 5. Deploy Qwen (inline or create qwen-pod.yaml)
# See deploy.sh for the pod definition
```

## Testing

```bash
# Port forward to each service
kubectl port-forward svc/vllm-llama-32-1b 8001:8000 &
kubectl port-forward svc/vllm-qwen2-5-0-5b 8002:8000 &

# Test Llama
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "meta-llama/Llama-3.2-1B-Instruct",
       "messages": [{"role": "user", "content": "Hello!"}]}'

# Test Qwen
curl http://localhost:8002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen2.5-0.5B-Instruct",
       "messages": [{"role": "user", "content": "Hello!"}]}'
```

## Files

- `deploy.sh` - Automated deployment script
- `test.sh` - Test script for both models
- `vllm-pod.yaml` - Llama-3.2-1B pod definition
- `llama-3.2-1b-values.yaml` - Helm values for Llama (if using llm-d ModelService)
- `qwen2.5-0.5b-values.yaml` - Helm values for Qwen (if using llm-d ModelService)

## Cleanup

```bash
k3d cluster delete llmd-cluster
```
