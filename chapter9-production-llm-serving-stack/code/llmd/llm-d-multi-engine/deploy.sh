#!/bin/bash
# Automated deployment script for llm-d engine cluster
# This script sets up the entire llm-d cluster with vLLM and SGLang servers for Qwen2.5-0.5B-Instruct

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLUSTER_NAME="llmd-multiengine"

echo "=========================================="
echo "  LLM-d Engine Cluster Deployment Script"
echo "=========================================="
echo ""

# Check prerequisites
echo "📋 Checking prerequisites..."

# Check k3d
if ! command -v k3d &> /dev/null; then
    echo "❌ Error: k3d not found. Please install k3d first."
    exit 1
fi
echo "✅ k3d found: $(k3d --version | head -n1)"

# Check Helm
if ! command -v helm &> /dev/null; then
    echo "❌ Error: Helm not found. Please install Helm 3.x first."
    exit 1
fi
echo "✅ Helm found: $(helm version --short)"

# Check kubectl
if ! command -v kubectl &> /dev/null; then
    echo "❌ Error: kubectl not found. Please install kubectl first."
    exit 1
fi
echo "✅ kubectl found: $(kubectl version --client 2>/dev/null | head -n1 || kubectl version 2>/dev/null | head -n1 || echo 'installed')"

# Note: HF_TOKEN is optional for Qwen2.5-0.5B-Instruct (not gated)
# But we'll check if it's set for other potential gated models
if [ -z "$HF_TOKEN" ]; then
    echo "ℹ️  HF_TOKEN not set (optional for Qwen2.5-0.5B-Instruct as it's not gated)"
    echo "   If you need to access other gated models, set: export HF_TOKEN='your_token_here'"
else
    echo "✅ HF_TOKEN environment variable detected"
fi
echo ""

# Step 1: Create k3d cluster
echo "=========================================="
echo "Step 1: Creating k3d cluster"
echo "=========================================="
echo ""

if k3d cluster list | grep -q "$CLUSTER_NAME"; then
    echo "⚠️  Cluster $CLUSTER_NAME already exists"
    if [ "${RECREATE_CLUSTER:-false}" = "true" ]; then
        echo "🗑️  Deleting existing cluster (RECREATE_CLUSTER=true)..."
        k3d cluster delete "$CLUSTER_NAME"
    else
        echo "ℹ️  Using existing cluster (set RECREATE_CLUSTER=true to recreate)"
    fi
fi

if ! k3d cluster list | grep -q "$CLUSTER_NAME"; then
    echo "🚀 Creating k3d cluster: $CLUSTER_NAME"
    k3d cluster create "$CLUSTER_NAME" \
      --image k3s-cuda:v1.33.6-cuda-12.2.0-working \
      --gpus=all \
      --servers 1 \
      --agents 1 \
      --volume /raid/models:/models \
      --k3s-arg '--disable=traefik@server:0' \
      --port "9000:80@loadbalancer" \
      --port "9443:443@loadbalancer"
    
    echo "✅ Cluster created"
else
    echo "✅ Cluster already exists"
fi

# Merge kubeconfig
echo ""
echo "🔗 Merging kubeconfig..."
k3d kubeconfig merge "$CLUSTER_NAME" --kubeconfig-merge-default

# Switch context
echo "🔄 Switching to $CLUSTER_NAME context..."
kubectl config use-context "k3d-$CLUSTER_NAME"

# Wait for cluster to be ready
echo "⏳ Waiting for cluster to be ready..."
sleep 5
kubectl wait --for=condition=Ready nodes --all --timeout=60s || true

echo ""
echo "✅ Cluster is ready"
kubectl get nodes
echo ""

# Step 2: Install NVIDIA device plugin
echo "=========================================="
echo "Step 2: Installing NVIDIA device plugin"
echo "=========================================="
echo ""

if kubectl get daemonset nvidia-device-plugin-daemonset -n kube-system &>/dev/null; then
    echo "✅ NVIDIA device plugin already installed"
else
    echo "📦 Installing NVIDIA device plugin..."
    kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.1/nvidia-device-plugin.yml
    
    echo "⏳ Waiting for device plugin to be ready..."
    kubectl wait --for=condition=ready pod \
      -l name=nvidia-device-plugin-ds \
      -n kube-system \
      --timeout=60s || echo "⚠️  Device plugin may need more time"
fi

echo "⏳ Waiting for GPU detection (this may take 20-30 seconds)..."
for i in {1..30}; do
    if kubectl get nodes -o json | python3 -c "import sys, json; d=json.load(sys.stdin); nodes=[n for n in d['items'] if n['status'].get('allocatable', {}).get('nvidia.com/gpu')]; exit(0 if nodes else 1)" 2>/dev/null; then
        echo "✅ GPUs detected on nodes"
        break
    fi
    if [ $i -eq 30 ]; then
        echo "⚠️  GPUs not yet detected (device plugin may need more time)"
    else
        echo -n "."
        sleep 1
    fi
done
echo ""

# Step 3: Install llm-d framework
echo "=========================================="
echo "Step 3: Installing llm-d framework"
echo "=========================================="
echo ""

# Add Helm repos if not already added
if ! helm repo list | grep -q llm-d-infra; then
    echo "📦 Adding llm-d-infra Helm repository..."
    helm repo add llm-d-infra https://llm-d-incubation.github.io/llm-d-infra/ || {
        echo "⚠️  Warning: Failed to add llm-d-infra Helm repo."
        USE_LLMD_CRD=false
    }
fi

if ! helm repo list | grep -q llm-d-modelservice; then
    echo "📦 Adding llm-d-modelservice Helm repository..."
    helm repo add llm-d-modelservice https://llm-d-incubation.github.io/llm-d-modelservice/ || {
        echo "⚠️  Warning: Failed to add llm-d-modelservice Helm repo."
    }
fi

if [ "${USE_LLMD_CRD:-true}" = "true" ]; then
    helm repo update
    echo "✅ Helm repos updated"
fi

# Install llm-d-infra (this installs CRDs and controller for LLMInferenceService)
if [ "${USE_LLMD_CRD:-true}" = "true" ]; then
    if helm list -n llm-d 2>/dev/null | grep -q llm-d-infra; then
        echo "✅ llm-d-infra already installed"
    elif kubectl get crd llminferenceservices.llm-d.ai &>/dev/null; then
        echo "✅ LLMInferenceService CRD already exists (may have been installed manually)"
    else
        echo "📦 Installing llm-d-infra (installs LLMInferenceService CRD and controller)..."
        if helm install llm-d-infra llm-d-infra/llm-d-infra \
          --namespace llm-d \
          --create-namespace \
          --wait \
          --timeout 10m; then
            echo "✅ llm-d-infra installed successfully"
        else
            echo "⚠️  Warning: llm-d-infra Helm installation failed or timed out."
            echo "   This might be expected if the Helm chart is not available."
            echo "   Falling back to direct Pod deployment..."
            USE_LLMD_CRD=false
        fi
    fi
    
    if [ "${USE_LLMD_CRD:-true}" = "true" ]; then
        echo ""
        echo "🔍 Verifying llm-d installation..."
        if kubectl get crd llminferenceservices.llm-d.ai &>/dev/null; then
            echo "✅ LLMInferenceService CRD found"
            kubectl get crd | grep llm-d
        else
            echo "⚠️  LLMInferenceService CRD not found. CRDs may need more time to be installed."
            echo "   You can check with: kubectl get crd | grep llm-d"
        fi
        
        echo ""
        echo "📦 llm-d pods:"
        kubectl get pods -n llm-d 2>/dev/null || echo "⚠️  No pods found in llm-d namespace (this may be normal if only CRDs are installed)"
    fi
fi
echo ""

# Step 4: Create HF token secret (optional for Qwen2.5-0.5B-Instruct)
echo "=========================================="
echo "Step 4: Creating HuggingFace token secret (optional)"
echo "=========================================="
echo ""

if [ -n "$HF_TOKEN" ]; then
    if kubectl get secret hf-token-secret &>/dev/null; then
        echo "🔄 Updating existing secret..."
        kubectl delete secret hf-token-secret 2>/dev/null || true
    fi

    echo "📝 Creating HF token secret..."
    kubectl create secret generic hf-token-secret \
      --from-literal=token="$HF_TOKEN"

    echo "✅ Secret created"
else
    echo "ℹ️  Skipping HF token secret (Qwen2.5-0.5B-Instruct is not gated)"
    echo "   If you need to access other gated models, set HF_TOKEN and rerun this step"
fi
echo ""

# Step 5: Deploy vLLM server
echo "=========================================="
echo "Step 5: Deploying vLLM server"
echo "=========================================="
echo ""

if [ "${USE_LLMD_CRD:-false}" = "true" ] && kubectl get crd llminferenceservices.llm-d.ai &>/dev/null; then
    echo "📦 Deploying vLLM using llm-d LLMInferenceService..."
    echo "⚠️  NOTE: vllm-llminference.yaml has been removed. Use deploy-modelservice.sh instead."
    # kubectl apply -f "$SCRIPT_DIR/vllm-llminference.yaml"  # File removed - use deploy-modelservice.sh
    echo "   Please use: ./deploy-modelservice.sh"
    exit 1
else
    echo "📦 Deploying vLLM using direct Pod deployment..."
    echo "⚠️  NOTE: vllm-pod.yaml has been removed. Use deploy-modelservice.sh instead."
    # kubectl apply -f "$SCRIPT_DIR/vllm-pod.yaml"  # File removed - use deploy-modelservice.sh
    echo "   Please use: ./deploy-modelservice.sh"
    exit 1
    
    echo "⏳ Waiting for vLLM pod to be ready..."
    kubectl wait --for=condition=ready pod/vllm-qwen2-5-0-5b --timeout=600s || echo "⚠️  Pod may still be starting (model loading can take time)"
fi

echo "✅ vLLM deployment complete"
kubectl get pod,svc -l app=vllm,model=qwen2-5-0-5b
echo ""

# Step 6: Deploy SGLang server
echo "=========================================="
echo "Step 6: Deploying SGLang server"
echo "=========================================="
echo ""

if [ "${USE_LLMD_CRD:-false}" = "true" ] && kubectl get crd llminferenceservices.llm-d.ai &>/dev/null; then
    echo "📦 Deploying SGLang using llm-d LLMInferenceService..."
    kubectl apply -f "$SCRIPT_DIR/sglang-llminference.yaml"
    
    echo "⏳ Waiting for SGLang service to be ready..."
    kubectl wait --for=condition=ready pod \
      -l llm-d.ai/inference-service=sglang-qwen2-5-0-5b \
      --timeout=600s || echo "⚠️  Pod may still be starting (model loading can take time)"
else
    echo "📦 Deploying SGLang using direct Pod deployment..."
    kubectl apply -f "$SCRIPT_DIR/sglang-pod.yaml"
    
    echo "⏳ Waiting for SGLang pod to be ready..."
    kubectl wait --for=condition=ready pod/sglang-qwen2-5-0-5b --timeout=600s || echo "⚠️  Pod may still be starting (model loading can take time)"
fi

echo "✅ SGLang deployment complete"
kubectl get pod,svc -l app=sglang,model=qwen2-5-0-5b
echo ""

# Final status
echo "=========================================="
echo "  Deployment Summary"
echo "=========================================="
echo ""
echo "📊 Cluster status:"
kubectl get nodes
echo ""
echo "📦 All pods:"
kubectl get pods -o wide
echo ""
echo "🔗 All services:"
kubectl get svc
echo ""

if [ "${USE_LLMD_CRD:-false}" = "true" ] && kubectl get crd llminferenceservices.llm-d.ai &>/dev/null; then
    echo "🎯 LLMInferenceServices:"
    kubectl get llminferenceservice
    echo ""
fi

echo "✅ Deployment complete!"
echo ""
echo "💡 Next steps:"
echo "   1. Check pod logs: kubectl logs -f vllm-qwen2-5-0-5b"
echo "   2. Check pod logs: kubectl logs -f sglang-qwen2-5-0-5b"
echo "   3. Test servers: ./test.sh"
echo "   4. Port forward: kubectl port-forward svc/vllm-qwen2-5-0-5b 8001:8000"
echo "   5. Port forward: kubectl port-forward svc/sglang-qwen2-5-0-5b 8002:8000"
echo ""
