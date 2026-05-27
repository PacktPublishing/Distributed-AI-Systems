#!/bin/bash
# Deploy TensorRT-LLM Llama-3.2-1B-Instruct
# Create Secret from environment variable $HF_TOKEN
#
# Note: TensorRT-LLM requires a pre-compiled model repository
# Make sure you have the TensorRT-LLM model ready at /models/tensorrt-llm/
# before deploying this service

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
YAML_FILE="$SCRIPT_DIR/llama-3.2-1b.yaml"
NAMESPACE="${NAMESPACE:-multi-engines}"

echo "=== Deploy TensorRT-LLM Llama-3.2-1B-Instruct ==="
echo ""

# Check HF_TOKEN environment variable
if [ -z "$HF_TOKEN" ]; then
    echo "❌ Error: HF_TOKEN environment variable not set"
    echo ""
    echo "Please set the environment variable first:"
    echo "  export HF_TOKEN='your_token_here'"
    echo ""
    echo "Or:"
    echo "  HF_TOKEN='your_token_here' $0"
    exit 1
fi

echo "✅ HF_TOKEN environment variable detected"
echo ""

# Create namespace if it doesn't exist
echo "📦 Creating namespace: $NAMESPACE"
kubectl create namespace "$NAMESPACE" 2>/dev/null || echo "  ✅ Namespace already exists"
echo ""

# Create or update Secret (reuse existing secret if it exists)
echo "📝 Creating/updating Secret: hf-token-secret in namespace $NAMESPACE"
kubectl delete secret hf-token-secret -n "$NAMESPACE" 2>/dev/null || true
kubectl create secret generic hf-token-secret \
  --from-literal=token="$HF_TOKEN" \
  -n "$NAMESPACE"

if [ $? -eq 0 ]; then
    echo "✅ Secret created successfully"
else
    echo "❌ Secret creation failed"
    exit 1
fi

echo ""
echo "📝 Deploying Deployment and Service..."
kubectl apply -f "$YAML_FILE" -n "$NAMESPACE"

echo ""
echo "✅ Deployment complete!"
echo ""
echo "📊 Check Pod status:"
echo "   kubectl get pod -n $NAMESPACE -l app=tensorrt,model=llama-32-1b -w"
echo ""
echo "📝 View logs:"
echo "   kubectl logs -f -n $NAMESPACE -l app=tensorrt,model=llama-32-1b"
echo ""
echo "🔗 Access service:"
echo "   kubectl port-forward svc/tensorrt-llama-32-1b-service -n $NAMESPACE 8000:8000"
echo "   curl http://localhost:8000/v2/health/ready"
echo ""
echo "⚠️  Note: TensorRT-LLM requires a pre-compiled model repository"
echo "   Make sure /models/tensorrt-llm/ contains the compiled TensorRT-LLM model"
