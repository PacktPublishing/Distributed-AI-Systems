#!/bin/bash
# Deploy API Gateway for engine comparison
# This script deploys the API Gateway that routes requests based on 'owned_by' field

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAMESPACE="${NAMESPACE:-llm-d-multi-engine}"

echo "=========================================="
echo "  Deploying Engine Comparison API Gateway"
echo "=========================================="
echo ""
echo "Namespace: ${NAMESPACE}"
echo ""

# Update ConfigMap from api-gateway.py
echo "📝 Updating ConfigMap from api-gateway.py..."
"${SCRIPT_DIR}/update-configmap.sh"

# Deploy Gateway
echo ""
echo "📦 Deploying API Gateway..."
kubectl apply -f "${SCRIPT_DIR}/api-gateway.yaml"

# Wait for Gateway to be ready
echo ""
echo "⏳ Waiting for Gateway to be ready..."
kubectl wait --for=condition=ready pod \
  -l app=engine-comparison-gateway \
  -n "${NAMESPACE}" \
  --timeout=60s || echo "⚠️  Gateway may still be starting"

echo ""
echo "✅ API Gateway deployed successfully!"
echo ""
echo "💡 Next steps:"
echo "   1. Port-forward to Gateway:"
echo "      kubectl port-forward -n ${NAMESPACE} svc/engine-comparison-gateway 8000:8000"
echo ""
echo "   2. Trigger service discovery:"
echo "      curl -X POST http://localhost:8000/admin/discover"
echo ""
echo "   3. Test routing:"
echo "      # Test vLLM"
echo "      curl -X POST http://localhost:8000/v1/chat/completions \\"
echo "        -H 'Content-Type: application/json' \\"
echo "        -d '{\"model\": \"Qwen/Qwen2.5-0.5B-Instruct\", \"owned_by\": \"vllm\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}]}'"
echo ""
echo "      # Test SGLang"
echo "      curl -X POST http://localhost:8000/v1/chat/completions \\"
echo "        -H 'Content-Type: application/json' \\"
echo "        -d '{\"model\": \"Qwen/Qwen2.5-0.5B-Instruct\", \"owned_by\": \"sglang\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}]}'"
echo ""
