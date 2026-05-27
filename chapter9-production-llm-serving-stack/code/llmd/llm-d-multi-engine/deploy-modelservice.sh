#!/bin/bash
# Deploy using ModelService Helm charts (same approach as llm-d-multi-model)
# This uses helmfile from llm-d guides/inference-scheduling

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLUSTER_NAME="llmd-multiengine"
NAMESPACE="${NAMESPACE:-llm-d-multi-engine}"

# Check LLMD_HOME
if [ -z "$LLMD_HOME" ]; then
    echo "❌ Error: LLMD_HOME environment variable not set."
    echo "   Please set: export LLMD_HOME=/path/to/llm-d"
    exit 1
fi

echo "=========================================="
echo "  LLM-d Multi-Engine Deployment (ModelService)"
echo "=========================================="
echo ""
echo "LLMD_HOME: $LLMD_HOME"
echo "NAMESPACE: $NAMESPACE"
echo ""

# Switch to cluster context
kubectl config use-context "k3d-$CLUSTER_NAME" || {
    echo "❌ Error: Cluster context not found. Please run deploy.sh first."
    exit 1
}

# Create namespace
kubectl create namespace "$NAMESPACE" 2>/dev/null || echo "✅ Namespace already exists"

# Install Gateway Provider (if needed)
if ! kubectl get crd gateways.gateway.networking.k8s.io &>/dev/null; then
    echo "📦 Installing Gateway Provider dependencies..."
    cd "${LLMD_HOME}/guides/prereq/gateway-provider"
    bash install-gateway-provider-dependencies.sh
    if command -v helmfile &>/dev/null; then
        helmfile apply -f istio.helmfile.yaml || echo "⚠️  Istio installation may have issues"
    fi
    cd - > /dev/null
fi

# Create HF token secret
if [ -n "$HF_TOKEN" ]; then
    kubectl create secret generic llm-d-hf-token \
      --from-literal="HF_TOKEN=${HF_TOKEN}" \
      --namespace "${NAMESPACE}" \
      --dry-run=client -o yaml | kubectl apply -f -
fi

# Deploy vLLM
echo "=========================================="
echo "Deploying vLLM ModelService"
echo "=========================================="
cd "${LLMD_HOME}/guides/inference-scheduling"
mkdir -p ms-inference-scheduling
cp "${SCRIPT_DIR}/qwen2.5-0.5b-vllm-values.yaml" ms-inference-scheduling/values.yaml

# Update for vLLM (GPU 0)
yq eval '.decode.extraConfig.nodeSelector."kubernetes.io/hostname" = "k3d-llmd-multiengine-agent-0"' -i ms-inference-scheduling/values.yaml
yq eval '.decode.containers[0].env = [{"name": "NVIDIA_VISIBLE_DEVICES", "value": "0"}, {"name": "CUDA_VISIBLE_DEVICES", "value": "0"}, {"name": "CUDA_DEVICE_ORDER", "value": "PCI_BUS_ID"}]' -i ms-inference-scheduling/values.yaml

RELEASE_NAME_POSTFIX=vllm-qwen2-5-0-5b helmfile apply -n "${NAMESPACE}"
cd - > /dev/null

# Deploy SGLang (create separate values)
echo "=========================================="
echo "Deploying SGLang ModelService"
echo "=========================================="
SGLANG_VALUES="${SCRIPT_DIR}/qwen2.5-0.5b-sglang-values.yaml"
# Note: SGLANG_VALUES already exists, no need to copy from vllm values

# Update for SGLang
yq eval '.decode.containers[0].name = "sglang"' -i "$SGLANG_VALUES"
yq eval '.decode.containers[0].modelCommand = "sglangServe"' -i "$SGLANG_VALUES"
yq eval '.decode.containers[0].args = ["--mem-fraction-static", "0.1"]' -i "$SGLANG_VALUES"
yq eval '.decode.extraConfig.nodeSelector."kubernetes.io/hostname" = "k3d-llmd-multiengine-agent-0"' -i "$SGLANG_VALUES"
yq eval '.decode.containers[0].env = [{"name": "NVIDIA_VISIBLE_DEVICES", "value": "1"}, {"name": "CUDA_VISIBLE_DEVICES", "value": "0"}, {"name": "CUDA_DEVICE_ORDER", "value": "PCI_BUS_ID"}]' -i "$SGLANG_VALUES"

cd "${LLMD_HOME}/guides/inference-scheduling"
mkdir -p ms-inference-scheduling
cp "$SGLANG_VALUES" ms-inference-scheduling/values.yaml

RELEASE_NAME_POSTFIX=sglang-qwen2-5-0-5b helmfile apply -n "${NAMESPACE}"
cd - > /dev/null

echo "✅ Deployment complete!"
kubectl get pods -n "${NAMESPACE}" -o wide
