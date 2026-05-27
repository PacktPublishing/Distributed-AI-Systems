# API Gateway Deployment

## Overview

The API Gateway provides a unified entry point for routing requests to different inference engines (vLLM, SGLang) based on the `model` and `owned_by` fields in the request.

## Architecture

- **Single Source of Truth**: The Python code lives in `api-gateway.py`
- **Dynamic Routing Config**: Routing configuration is loaded from `routing-config.yaml`
- **ConfigMap Generation**: Both files are automatically generated into ConfigMap during deployment
- **No Code Duplication**: The YAML file no longer contains hardcoded Python code

## Configuration

### Routing Configuration File

The routing configuration is defined in `routing-config.yaml`:

```yaml
routing:
  - model: "meta-llama/Llama-3.2-1B-Instruct"
    inference_server: "vllm"
    service_name: "vllm-llama-32-1b"
  
  - model: "/models/Phi-tiny-MoE-instruct"
    inference_server: "vllm"
    service_name: "vllm-phi-tiny-moe-service"
```

**Format:**
- `model`: The model identifier (as it appears in requests)
- `inference_server`: The inference engine type (`"vllm"`, `"sglang"`, or `null` for default)
- `service_name`: The Kubernetes Service name to route to

**Benefits:**
- ✅ Easy to add new models without modifying code
- ✅ No need to restart Gateway code for routing changes
- ✅ Configuration is version-controlled and reviewable
- ✅ Falls back to hardcoded defaults if config file is missing

## Deployment

### Automatic Deployment (Recommended)

```bash
cd /home/wukong/workspace/coderepo/09/code/k3d/gateway
./deploy-gateway.sh
```

This script will:
1. Generate the ConfigMap from `api-gateway.py` and `routing-config.yaml`
2. Deploy the Pod and Service
3. Wait for the Gateway to be ready

### Manual Deployment

If you prefer to deploy manually:

```bash
# 1. Generate ConfigMap from Python file and routing config
kubectl create configmap vllm-api-gateway-code \
  --from-file=api-gateway.py=api-gateway.py \
  --from-file=routing-config.yaml=routing-config.yaml \
  --dry-run=client -o yaml | kubectl apply -f -

# 2. Deploy Pod and Service
kubectl apply -f api-gateway.yaml
```

## Updating Routing Configuration

1. Edit `routing-config.yaml`
2. Run `./deploy-gateway.sh` to regenerate ConfigMap and redeploy
3. The Pod will automatically restart with the new configuration

**Note:** If you only change `routing-config.yaml`, you can update just the ConfigMap:

```bash
kubectl create configmap vllm-api-gateway-code \
  --from-file=api-gateway.py=api-gateway.py \
  --from-file=routing-config.yaml=routing-config.yaml \
  --dry-run=client -o yaml | kubectl apply -f -

# Restart pod to pick up new config
kubectl delete pod vllm-api-gateway
kubectl apply -f api-gateway.yaml
```

## File Structure

```
gateway/
├── api-gateway.py          # Source code (single source of truth)
├── routing-config.yaml     # Routing configuration (dynamic)
├── api-gateway.yaml        # Pod, Service definitions (no hardcoded code)
├── deploy-gateway.sh       # Deployment script (generates ConfigMap)
└── README.md              # This file
```

## Accessing the Gateway

```bash
# Port forward
kubectl port-forward svc/vllm-api-gateway 8000:8000

# Test
curl http://localhost:8000/health
curl http://localhost:8000/v1/models
```

See `../TEST_GATEWAY.md` for more examples.
