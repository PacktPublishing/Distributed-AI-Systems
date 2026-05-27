#!/bin/bash
# Script to access vLLM API Gateway via Ingress using port-forward
# This is needed because k3d NodePort services are not automatically exposed to the host

set -e

PORT=${1:-8443}
NAMESPACE="ingress-nginx"
SERVICE="ingress-nginx-controller"

echo "🔗 Setting up port-forward to access vLLM API Gateway via Ingress"
echo ""
echo "This will forward Ingress Controller Service (port 443) to localhost:$PORT"
echo ""
echo "Press Ctrl+C to stop the port-forward"
echo ""

# Check if port-forward is already running
if pgrep -f "port-forward.*$SERVICE.*$PORT" > /dev/null; then
    echo "⚠️  Port-forward is already running on port $PORT"
    echo "   Killing existing port-forward..."
    pkill -f "port-forward.*$SERVICE.*$PORT" || true
    sleep 2
fi

# Start port-forward
echo "🚀 Starting port-forward..."
kubectl port-forward -n $NAMESPACE svc/$SERVICE $PORT:443

