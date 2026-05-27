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
echo "✅ LLMD_HOME: $LLMD_HOME"

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

# Step 3: Install Gateway Provider dependencies (required for llm-d-infra)
echo "=========================================="
echo "Step 3: Installing Gateway Provider dependencies"
echo "=========================================="
echo ""

# Check if Gateway API CRDs are installed
if kubectl get crd gateways.gateway.networking.k8s.io &>/dev/null; then
    echo "✅ Gateway API CRDs already installed"
else
    echo "📦 Installing Gateway Provider dependencies..."
    if [ -f "${LLMD_HOME}/guides/prereq/gateway-provider/install-gateway-provider-dependencies.sh" ]; then
        cd "${LLMD_HOME}/guides/prereq/gateway-provider"
        bash install-gateway-provider-dependencies.sh
        cd - > /dev/null
        echo "✅ Gateway Provider dependencies installed"
    else
        echo "⚠️  Warning: Gateway Provider installation script not found at ${LLMD_HOME}/guides/prereq/gateway-provider/install-gateway-provider-dependencies.sh"
        echo "   Skipping Gateway Provider installation. llm-d-infra may fail if CRDs are missing."
    fi
fi

# Install Istio Gateway Provider (required for llm-d-infra)
if kubectl get crd telemetries.telemetry.istio.io &>/dev/null; then
    echo "✅ Istio CRDs already installed"
else
    echo "📦 Installing Istio Gateway Provider..."
    if [ -f "${LLMD_HOME}/guides/prereq/gateway-provider/istio.helmfile.yaml" ]; then
        cd "${LLMD_HOME}/guides/prereq/gateway-provider"
        if command -v helmfile &>/dev/null; then
            helmfile apply -f istio.helmfile.yaml || {
                echo "⚠️  Warning: Istio Gateway Provider installation failed or timed out."
                echo "   This may be expected. Continuing..."
            }
        else
            echo "⚠️  Warning: helmfile not found. Skipping Istio Gateway Provider installation."
            echo "   Install helmfile or install Istio manually if needed."
        fi
        cd - > /dev/null
    else
        echo "⚠️  Warning: Istio helmfile not found. Skipping Istio installation."
    fi
fi
echo ""

# Step 4: Install llm-d framework
echo "=========================================="
echo "Step 4: Installing llm-d framework"
echo "=========================================="
echo ""

# Check if llm-d is already installed
if kubectl get crd llminferenceservices.llm-d.ai &>/dev/null; then
    echo "✅ llm-d framework already installed (CRD found)"
    kubectl get pods -n llm-d 2>/dev/null || echo "⚠️  Note: llm-d namespace may not exist yet"
else
    echo "📦 Installing llm-d framework..."
    
    # Add Helm repos if not already added
    if ! helm repo list | grep -q llm-d-infra; then
        echo "📦 Adding llm-d-infra Helm repository..."
        helm repo add llm-d-infra https://llm-d-incubation.github.io/llm-d-infra/ || {
            echo "⚠️  Warning: Failed to add llm-d-infra Helm repo."
            echo "   Trying alternative installation methods..."
        }
    else
        echo "✅ llm-d-infra Helm repo already added"
    fi
    
    if ! helm repo list | grep -q llm-d-modelservice; then
        echo "📦 Adding llm-d-modelservice Helm repository..."
        helm repo add llm-d-modelservice https://llm-d-incubation.github.io/llm-d-modelservice/ || {
            echo "⚠️  Warning: Failed to add llm-d-modelservice Helm repo."
        }
    else
        echo "✅ llm-d-modelservice Helm repo already added"
    fi
    
    helm repo update
    echo "✅ Helm repos updated"
    
    # Try to install via Helm
    if helm repo list | grep -q llm-d-infra; then
        if helm list -n llm-d 2>/dev/null | grep -q llm-d-infra; then
            echo "✅ llm-d-infra already installed via Helm"
        else
            echo "📦 Installing llm-d-infra (installs LLMInferenceService CRD and controller)..."
            if helm install llm-d-infra llm-d-infra/llm-d-infra \
              --namespace llm-d \
              --create-namespace \
              --wait \
              --timeout 10m; then
                echo "✅ llm-d-infra installed successfully"
            else
                echo "❌ Failed to install llm-d-infra via Helm"
                echo ""
                echo "This may be due to missing Gateway API or Istio CRDs."
                echo "Please ensure Gateway Provider dependencies are installed:"
                echo "  1. Run: cd ${LLMD_HOME}/guides/prereq/gateway-provider && ./install-gateway-provider-dependencies.sh"
                echo "  2. Run: helmfile apply -f istio.helmfile.yaml"
                echo "  3. Then retry this script"
                exit 1
            fi
        fi
    else
        echo "❌ Error: Cannot install llm-d framework."
        echo "   The llm-d-infra Helm repository is not available."
        echo ""
        echo "Please install llm-d framework manually before running this script."
        echo "You may need to:"
        echo "  1. Add the repo manually: helm repo add llm-d-infra https://llm-d-incubation.github.io/llm-d-infra/"
        echo "  2. Build and install llm-d from source"
        echo "  3. Install CRDs manually from the llm-d source code"
        exit 1
    fi
fi

# Verify installation
echo ""
echo "🔍 Verifying llm-d installation..."
if kubectl get crd llminferenceservices.llm-d.ai &>/dev/null; then
    echo "✅ LLMInferenceService CRD found"
    kubectl get crd | grep llm-d
else
    echo "⚠️  Warning: LLMInferenceService CRD not found after installation"
    echo "   CRDs may need more time to be installed. Checking again..."
    sleep 5
    if kubectl get crd llminferenceservices.llm-d.ai &>/dev/null; then
        echo "✅ LLMInferenceService CRD found (after retry)"
        kubectl get crd | grep llm-d
    else
        echo "❌ Error: LLMInferenceService CRD still not found"
        echo "   Please check: kubectl get crd | grep llm-d"
        exit 1
    fi
fi

echo ""
echo "📦 llm-d pods:"
kubectl get pods -n llm-d 2>/dev/null || echo "⚠️  No pods found in llm-d namespace (this may be normal if only CRDs are installed)"
echo ""

# Step 5: Create HF token secret (optional for Qwen2.5-0.5B-Instruct)
echo "=========================================="
echo "Step 5: Creating HuggingFace token secret (optional)"
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

# Step 6: Deploy vLLM server
echo "=========================================="
echo "Step 6: Deploying vLLM server"
echo "=========================================="
echo ""

# Verify llm-d is installed
if ! kubectl get crd llminferenceservices.llm-d.ai &>/dev/null; then
    echo "❌ Error: llm-d framework is not installed (LLMInferenceService CRD not found)"
    echo "   Please install llm-d framework first (Step 3 should have done this)"
    exit 1
fi

echo "📦 Deploying vLLM using llm-d LLMInferenceService..."
kubectl apply -f "$SCRIPT_DIR/vllm-llminference.yaml"

echo "⏳ Waiting for vLLM service to be ready..."
kubectl wait --for=condition=ready pod \
  -l llm-d.ai/inference-service=vllm-qwen2-5-0-5b \
  --timeout=600s || echo "⚠️  Pod may still be starting (model loading can take time)"

echo "✅ vLLM deployment complete"
kubectl get llminferenceservice vllm-qwen2-5-0-5b
kubectl get pod,svc -l llm-d.ai/inference-service=vllm-qwen2-5-0-5b
echo ""

# Step 7: Deploy SGLang server
echo "=========================================="
echo "Step 7: Deploying SGLang server"
echo "=========================================="
echo ""

# Verify llm-d is installed
if ! kubectl get crd llminferenceservices.llm-d.ai &>/dev/null; then
    echo "❌ Error: llm-d framework is not installed (LLMInferenceService CRD not found)"
    echo "   Please install llm-d framework first (Step 3 should have done this)"
    exit 1
fi

echo "📦 Deploying SGLang using llm-d LLMInferenceService..."
kubectl apply -f "$SCRIPT_DIR/sglang-llminference.yaml"

echo "⏳ Waiting for SGLang service to be ready..."
kubectl wait --for=condition=ready pod \
  -l llm-d.ai/inference-service=sglang-qwen2-5-0-5b \
  --timeout=600s || echo "⚠️  Pod may still be starting (model loading can take time)"

echo "✅ SGLang deployment complete"
kubectl get llminferenceservice sglang-qwen2-5-0-5b
kubectl get pod,svc -l llm-d.ai/inference-service=sglang-qwen2-5-0-5b
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

echo "🎯 LLMInferenceServices:"
kubectl get llminferenceservice
echo ""

echo "✅ Deployment complete!"
echo ""
echo "💡 Next steps:"
echo "   1. Check LLMInferenceService status: kubectl get llminferenceservice"
echo "   2. Check pod logs: kubectl logs -f <vllm-pod-name>"
echo "   3. Check pod logs: kubectl logs -f <sglang-pod-name>"
echo "   4. Deploy API Gateway for unified interface: ./deploy-gateway.sh"
echo "   5. Test unified interface: ./test-unified-interface.sh"
echo ""
echo "📝 Note: Pod names are managed by llm-d. Use:"
echo "   kubectl get pods -l llm-d.ai/inference-service=vllm-qwen2-5-0-5b"
echo "   kubectl get pods -l llm-d.ai/inference-service=sglang-qwen2-5-0-5b"
echo ""
