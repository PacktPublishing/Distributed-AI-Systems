#!/bin/bash
# Deploy vLLM API Gateway

set -e

echo "🚀 Deploying vLLM API Gateway..."
echo ""

# Check if Gateway code file exists
if [ ! -f "api-gateway.py" ]; then
    echo "❌ Error: api-gateway.py not found in current directory"
    echo "   Please run this script from the gateway/ directory"
    exit 1
fi

# Generate ConfigMap from Python file and routing config (single source of truth)
echo "📝 Generating ConfigMap from api-gateway.py and routing-config.yaml..."
if [ -f "routing-config.yaml" ]; then
    kubectl create configmap vllm-api-gateway-code \
      --from-file=api-gateway.py=api-gateway.py \
      --from-file=routing-config.yaml=routing-config.yaml \
      --dry-run=client -o yaml | kubectl apply -f -
    echo "✅ ConfigMap includes both api-gateway.py and routing-config.yaml"
else
    kubectl create configmap vllm-api-gateway-code \
      --from-file=api-gateway.py=api-gateway.py \
      --dry-run=client -o yaml | kubectl apply -f -
    echo "⚠️  routing-config.yaml not found, using default hardcoded routing config"
fi

# Deploy Gateway (Pod and Service)
echo "📦 Applying Pod and Service..."
kubectl apply -f api-gateway.yaml

echo ""
echo "⏳ Waiting for Gateway Pod to be ready..."
kubectl wait --for=condition=ready pod/vllm-api-gateway --timeout=60s || {
    echo "⚠️  Pod not ready yet, checking status..."
    kubectl get pod vllm-api-gateway
    kubectl logs vllm-api-gateway || true
}

echo ""
echo "✅ Gateway deployed!"
echo ""
echo "📊 Status:"
kubectl get pod,svc -l app=vllm-gateway

echo ""
echo "🔍 Gateway logs:"
kubectl logs vllm-api-gateway --tail=20 || true

echo ""
echo "💡 Usage:"
echo "   # Port forward to access Gateway"
echo "   kubectl port-forward svc/vllm-api-gateway 8000:8000"
echo ""
echo "   # Test with curl"
echo "   curl http://localhost:8000/v1/chat/completions \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"model\": \"meta-llama/Llama-3.2-1B-Instruct\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}]}'"
echo ""
echo "   # Gateway will automatically route based on 'model' field in request body"
