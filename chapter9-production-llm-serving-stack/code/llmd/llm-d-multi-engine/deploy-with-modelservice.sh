#!/bin/bash
# Deploy using ModelService Helm charts (same as llm-d-multi-model)
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

if [ ! -d "$LLMD_HOME" ]; then
    echo "❌ Error: LLMD_HOME directory does not exist: $LLMD_HOME"
    exit 1
fi

echo "=========================================="
echo "  LLM-d Multi-Engine Deployment (ModelService)"
echo "=========================================="
echo ""
echo "LLMD_HOME: $LLMD_HOME"
echo "NAMESPACE: $NAMESPACE"
echo ""

# Step 1: Ensure cluster is ready
echo "=========================================="
echo "Step 1: Verifying cluster"
echo "=========================================="
kubectl config use-context "k3d-$CLUSTER_NAME" || {
    echo "❌ Error: Cluster context not found. Please run deploy.sh first."
    exit 1
}
kubectl get nodes
echo ""

# Step 2: Create namespace
echo "=========================================="
echo "Step 2: Creating namespace"
echo "=========================================="
kubectl create namespace "$NAMESPACE" 2>/dev/null || echo "✅ Namespace already exists"
echo ""

# Step 3: Install Gateway Provider (if not already installed)
echo "=========================================="
echo "Step 3: Installing Gateway Provider dependencies"
echo "=========================================="
if kubectl get crd gateways.gateway.networking.k8s.io &>/dev/null; then
    echo "✅ Gateway API CRDs already installed"
else
    echo "📦 Installing Gateway Provider dependencies..."
    cd "${LLMD_HOME}/guides/prereq/gateway-provider"
    bash install-gateway-provider-dependencies.sh
    cd - > /dev/null
fi

if kubectl get crd telemetries.telemetry.istio.io &>/dev/null; then
    echo "✅ Istio CRDs already installed"
else
    echo "📦 Installing Istio Gateway Provider..."
    cd "${LLMD_HOME}/guides/prereq/gateway-provider"
    if command -v helmfile &>/dev/null; then
        helmfile apply -f istio.helmfile.yaml || echo "⚠️  Istio installation may have issues"
    else
        echo "⚠️  helmfile not found. Skipping Istio installation."
    fi
    cd - > /dev/null
fi
echo ""

# Step 4: Create HF token secret
echo "=========================================="
echo "Step 4: Creating HuggingFace token secret"
echo "=========================================="
if [ -n "$HF_TOKEN" ]; then
    kubectl create secret generic llm-d-hf-token \
      --from-literal="HF_TOKEN=${HF_TOKEN}" \
      --namespace "${NAMESPACE}" \
      --dry-run=client -o yaml | kubectl apply -f -
    echo "✅ HF token secret created"
else
    echo "ℹ️  HF_TOKEN not set (optional for Qwen2.5-0.5B-Instruct)"
fi
echo ""

# Step 5: Deploy vLLM using ModelService
echo "=========================================="
echo "Step 5: Deploying vLLM using ModelService"
echo "=========================================="
cd "${LLMD_HOME}/guides/inference-scheduling"

# Prepare values file for vLLM
mkdir -p ms-inference-scheduling-vllm
cp "${SCRIPT_DIR}/qwen2.5-0.5b-vllm-values.yaml" ms-inference-scheduling-vllm/values.yaml

# Update nodeSelector for vLLM (GPU 0)
yq eval '.decode.extraConfig.nodeSelector."kubernetes.io/hostname" = "k3d-llmd-multiengine-agent-0"' -i ms-inference-scheduling-vllm/values.yaml
yq eval '.decode.containers[0].env[] | select(.name == "NVIDIA_VISIBLE_DEVICES").value = "0"' -i ms-inference-scheduling-vllm/values.yaml || \
yq eval '.decode.containers[0].env += [{"name": "NVIDIA_VISIBLE_DEVICES", "value": "0"}]' -i ms-inference-scheduling-vllm/values.yaml

# Deploy vLLM
echo "📦 Deploying vLLM ModelService..."
RELEASE_NAME_POSTFIX=vllm-qwen2-5-0-5b \
helmfile apply -n "${NAMESPACE}" \
  --set-file ms-inference-scheduling.values[0]="${SCRIPT_DIR}/qwen2.5-0.5b-vllm-values.yaml" || {
    echo "⚠️  Helmfile deployment failed. Trying alternative approach..."
    # Alternative: copy values to expected location
    mkdir -p ms-inference-scheduling
    cp "${SCRIPT_DIR}/qwen2.5-0.5b-vllm-values.yaml" ms-inference-scheduling/values.yaml
    RELEASE_NAME_POSTFIX=vllm-qwen2-5-0-5b helmfile apply -n "${NAMESPACE}"
}

echo "⏳ Waiting for vLLM pods..."
kubectl wait --for=condition=ready pod \
  -l llm-d.ai/role=decode \
  -n "${NAMESPACE}" \
  --timeout=600s || echo "⚠️  Pods may still be starting"

cd - > /dev/null
echo ""

# Step 6: Deploy SGLang using ModelService (need to create SGLang values)
echo "=========================================="
echo "Step 6: Preparing SGLang values file"
echo "=========================================="
# Create SGLang values file from vLLM template
SGLANG_VALUES="${SCRIPT_DIR}/qwen2.5-0.5b-sglang-values.yaml"
cp "${SCRIPT_DIR}/qwen2.5-0.5b-vllm-values.yaml" "$SGLANG_VALUES"

# Update for SGLang: change container to SGLang
yq eval '.decode.containers[0].name = "sglang"' -i "$SGLANG_VALUES"
yq eval '.decode.containers[0].image = "ghcr.io/llm-d/llm-d-cuda:v0.4.0"' -i "$SGLANG_VALUES"
yq eval '.decode.containers[0].modelCommand = "sglangServe"' -i "$SGLANG_VALUES"
yq eval '.decode.containers[0].args = ["--mem-fraction-static", "0.1"]' -i "$SGLANG_VALUES"
yq eval '.decode.extraConfig.nodeSelector."kubernetes.io/hostname" = "k3d-llmd-multiengine-agent-0"' -i "$SGLANG_VALUES"

# Update GPU assignment for SGLang (GPU 1)
yq eval 'del(.decode.containers[0].env[] | select(.name == "NVIDIA_VISIBLE_DEVICES"))' -i "$SGLANG_VALUES" || true
yq eval '.decode.containers[0].env += [{"name": "NVIDIA_VISIBLE_DEVICES", "value": "1"}]' -i "$SGLANG_VALUES"
yq eval '.decode.containers[0].env += [{"name": "CUDA_VISIBLE_DEVICES", "value": "0"}]' -i "$SGLANG_VALUES"
yq eval '.decode.containers[0].env += [{"name": "CUDA_DEVICE_ORDER", "value": "PCI_BUS_ID"}]' -i "$SGLANG_VALUES"

echo "✅ SGLang values file created: $SGLANG_VALUES"
echo ""

# Step 7: Deploy SGLang
echo "=========================================="
echo "Step 7: Deploying SGLang using ModelService"
echo "=========================================="
cd "${LLMD_HOME}/guides/inference-scheduling"

# Prepare values file for SGLang
mkdir -p ms-inference-scheduling-sglang
cp "$SGLANG_VALUES" ms-inference-scheduling-sglang/values.yaml

# Deploy SGLang
echo "📦 Deploying SGLang ModelService..."
RELEASE_NAME_POSTFIX=sglang-qwen2-5-0-5b \
helmfile apply -n "${NAMESPACE}" \
  --set-file ms-inference-scheduling.values[0]="$SGLANG_VALUES" || {
    # Alternative approach
    mkdir -p ms-inference-scheduling
    cp "$SGLANG_VALUES" ms-inference-scheduling/values.yaml
    RELEASE_NAME_POSTFIX=sglang-qwen2-5-0-5b helmfile apply -n "${NAMESPACE}"
}

echo "⏳ Waiting for SGLang pods..."
kubectl wait --for=condition=ready pod \
  -l llm-d.ai/role=decode \
  -n "${NAMESPACE}" \
  --timeout=600s || echo "⚠️  Pods may still be starting"

cd - > /dev/null
echo ""

# Final status
echo "=========================================="
echo "  Deployment Summary"
echo "=========================================="
echo ""
kubectl get pods -n "${NAMESPACE}" -o wide
echo ""
kubectl get svc -n "${NAMESPACE}"
echo ""

echo "✅ Deployment complete!"
echo ""
echo "💡 Next steps:"
echo "   1. Deploy API Gateway: ./deploy-gateway.sh"
echo "   2. Test unified interface: ./test-unified-interface.sh"
echo ""
