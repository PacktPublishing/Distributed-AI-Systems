#!/bin/bash
# View logs for vLLM or SGLang pods, filtering out /health endpoint logs
# 
# Usage:
#   ./view-logs.sh [vllm|sglang] [namespace]
# 
# Examples:
#   ./view-logs.sh vllm multi-engines
#   ./view-logs.sh sglang multi-engines
#   ./view-logs.sh vllm  # Uses default namespace 'multi-engines'

set -e

SERVICE="${1:-vllm}"
NAMESPACE="${2:-multi-engines}"

if [ "$SERVICE" = "vllm" ]; then
    POD_NAME=$(kubectl get pods -n "$NAMESPACE" -l app=vllm,model=llama-32-1b -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
    if [ -z "$POD_NAME" ]; then
        echo "❌ vLLM pod not found in namespace $NAMESPACE"
        exit 1
    fi
elif [ "$SERVICE" = "sglang" ]; then
    POD_NAME=$(kubectl get pods -n "$NAMESPACE" -l app=sglang,model=llama-32-1b -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
    if [ -z "$POD_NAME" ]; then
        echo "❌ SGLang pod not found in namespace $NAMESPACE"
        exit 1
    fi
else
    echo "❌ Invalid service: $SERVICE (must be 'vllm' or 'sglang')"
    exit 1
fi

echo "📋 Viewing logs for $SERVICE pod: $POD_NAME"
echo "   (Filtering out /health endpoint logs)"
echo "   Press Ctrl+C to stop"
echo ""

# View logs and filter out /health requests
kubectl logs -f -n "$NAMESPACE" "$POD_NAME" 2>&1 | grep -v --line-buffered '/health' || true
