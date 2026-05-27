#!/bin/bash
# Script to update the ConfigMap from api-gateway.py
# This keeps the YAML file clean by not embedding Python code
# Usage: ./update-configmap.sh [--restart]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAMESPACE="${NAMESPACE:-llm-d-multi-engine}"
API_GATEWAY_FILE="${SCRIPT_DIR}/api-gateway.py"

if [ ! -f "${API_GATEWAY_FILE}" ]; then
    echo "❌ Error: api-gateway.py not found at ${API_GATEWAY_FILE}"
    exit 1
fi

echo "🔄 Updating ConfigMap from api-gateway.py..."
echo "   File: ${API_GATEWAY_FILE}"
echo "   Namespace: ${NAMESPACE}"

kubectl create configmap engine-comparison-gateway-code \
  --from-file=api-gateway.py="${API_GATEWAY_FILE}" \
  --namespace="${NAMESPACE}" \
  --dry-run=client -o yaml | kubectl apply -f -

echo "✅ ConfigMap updated successfully!"

if [[ "$1" == "--restart" ]]; then
    echo "⏳ Restarting Gateway deployment..."
    kubectl rollout restart deployment/engine-comparison-gateway -n "${NAMESPACE}"
    echo "✅ Gateway restart initiated"
    echo "   Check status: kubectl get pods -l app=engine-comparison-gateway -n ${NAMESPACE}"
else
    echo ""
    echo "💡 To restart the Gateway:"
    echo "   kubectl rollout restart deployment/engine-comparison-gateway -n ${NAMESPACE}"
    echo ""
    echo "   Or run: ./update-configmap.sh --restart"
fi
