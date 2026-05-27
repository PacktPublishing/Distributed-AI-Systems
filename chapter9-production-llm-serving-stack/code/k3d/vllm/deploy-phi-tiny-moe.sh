#!/bin/bash
# Deploy vLLM serving Phi-tiny-MoE-instruct model in k3d cluster

set -e

echo "=== Deploying vLLM for Phi-tiny-MoE-instruct ==="

# Check if cluster exists
if ! kubectl cluster-info &>/dev/null; then
    echo "Error: No Kubernetes cluster found. Please create a k3d cluster first."
    echo "Example:"
    echo "  k3d cluster create mycluster-gpu \\"
    echo "    --image k3s-cuda:v1.33.6-cuda-12.4.1 \\"
    echo "    --gpus=all \\"
    echo "    --servers 1 \\"
    echo "    --agents 1 \\"
    echo "    --volume /raid/models:/models"
    exit 1
fi

# Check if model directory exists
if [ ! -d "/raid/models/Phi-tiny-MoE-instruct" ]; then
    echo "Warning: Model directory /raid/models/Phi-tiny-MoE-instruct not found"
    echo "Please ensure the model is available at this path"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Apply deployment
echo "Applying vLLM deployment..."
kubectl apply -f "$(dirname "$0")/phi-tiny-moe.yaml"

# Wait for deployment
echo "Waiting for pod to be ready (this may take several minutes for model loading)..."
kubectl wait --for=condition=Ready pod -l app=vllm,model=phi-tiny-moe --timeout=600s || {
    echo "Pod did not become ready in time. Checking status..."
    kubectl get pods -l app=vllm,model=phi-tiny-moe
    kubectl logs -l app=vllm,model=phi-tiny-moe --tail=50
    exit 1
}

echo ""
echo "✅ vLLM deployment successful!"
echo ""
echo "Pod status:"
kubectl get pods -l app=vllm,model=phi-tiny-moe

echo ""
echo "To access the service:"
echo ""
echo "1. Port-forward:"
echo "   kubectl port-forward svc/vllm-phi-tiny-moe-service 8000:8000 &"
echo ""
echo "2. Test health:"
echo "   curl http://localhost:8000/health"
echo ""
echo "3. Test completion:"
echo "   curl http://localhost:8000/v1/completions \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"model\": \"/models/Phi-tiny-MoE-instruct\", \"prompt\": \"What is AI?\", \"max_tokens\": 50}'"
echo ""
echo "4. View logs:"
echo "   kubectl logs -f -l app=vllm,model=phi-tiny-moe"
echo ""
