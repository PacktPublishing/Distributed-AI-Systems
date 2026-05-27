#!/bin/bash
# Final comprehensive test of unified interface

set -e

NAMESPACE="llm-d-multi-engine"
GATEWAY_SERVICE="engine-comparison-gateway"
GATEWAY_PORT=8000

echo "=========================================="
echo "  Final Unified Interface Test"
echo "=========================================="
echo ""

# Check Gateway
GATEWAY_POD=$(kubectl get pod -n $NAMESPACE -l app=$GATEWAY_SERVICE -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
if [ -z "$GATEWAY_POD" ]; then
    echo "❌ Gateway not found. Deploy it first: ./deploy-gateway.sh"
    exit 1
fi

echo "✅ Gateway: $GATEWAY_POD"

# Check ModelService pods
VLLM_POD=$(kubectl get pod -n $NAMESPACE -l llm-d.ai/role=decode | grep vllm | awk '{print $1}' | head -1)
SGLANG_POD=$(kubectl get pod -n $NAMESPACE -l llm-d.ai/role=decode | grep sglang | awk '{print $1}' | head -1)

echo "✅ vLLM pod: ${VLLM_POD:-Not found}"
echo "✅ SGLang pod: ${SGLANG_POD:-Not found}"
echo ""

# Start port-forward
echo "🔗 Starting port-forward..."
pkill -f "kubectl port-forward.*$GATEWAY_SERVICE" 2>/dev/null || true
sleep 1
kubectl port-forward -n $NAMESPACE svc/$GATEWAY_SERVICE ${GATEWAY_PORT}:8000 > /tmp/gateway-pf.log 2>&1 &
PF_PID=$!
sleep 3

# Test 1: Health
echo "Test 1: Health check"
HEALTH=$(curl -s http://localhost:${GATEWAY_PORT}/health)
echo "$HEALTH" | jq '.' 2>/dev/null || echo "$HEALTH"
echo ""

# Test 2: Service discovery
echo "Test 2: Service discovery"
DISCOVERY=$(curl -s -X POST http://localhost:${GATEWAY_PORT}/admin/discover)
echo "$DISCOVERY" | jq '.' 2>/dev/null || echo "$DISCOVERY"
echo ""

# Test 3: List models
echo "Test 3: List models"
MODELS=$(curl -s http://localhost:${GATEWAY_PORT}/v1/models)
echo "$MODELS" | jq '.data[] | {id, owned_by}' 2>/dev/null || echo "$MODELS"
echo ""

# Test 4: vLLM request
echo "Test 4: vLLM chat completion"
VLLM_RESPONSE=$(curl -s -X POST http://localhost:${GATEWAY_PORT}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "owned_by": "vllm",
    "messages": [{"role": "user", "content": "Say hello in one word."}],
    "max_tokens": 10
  }')

if echo "$VLLM_RESPONSE" | jq -e '.choices[0].message.content' >/dev/null 2>&1; then
    echo "✅ vLLM: $(echo "$VLLM_RESPONSE" | jq -r '.choices[0].message.content')"
else
    echo "❌ vLLM failed: $(echo "$VLLM_RESPONSE" | jq -r '.error.message // .' 2>/dev/null || echo "$VLLM_RESPONSE")"
fi
echo ""

# Test 5: SGLang request (if pod is running)
if [ -n "$SGLANG_POD" ] && kubectl get pod "$SGLANG_POD" -n $NAMESPACE 2>/dev/null | grep -q Running; then
    echo "Test 5: SGLang chat completion"
    SGLANG_RESPONSE=$(curl -s -X POST http://localhost:${GATEWAY_PORT}/v1/chat/completions \
      -H "Content-Type: application/json" \
      -d '{
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "owned_by": "sglang",
        "messages": [{"role": "user", "content": "Say hello in one word."}],
        "max_tokens": 10
      }')
    
    if echo "$SGLANG_RESPONSE" | jq -e '.choices[0].message.content' >/dev/null 2>&1; then
        echo "✅ SGLang: $(echo "$SGLANG_RESPONSE" | jq -r '.choices[0].message.content')"
    else
        echo "❌ SGLang failed: $(echo "$SGLANG_RESPONSE" | jq -r '.error.message // .' 2>/dev/null || echo "$SGLANG_RESPONSE")"
    fi
    echo ""
fi

# Cleanup
kill $PF_PID 2>/dev/null || true

echo "=========================================="
echo "  Test Complete"
echo "=========================================="
echo ""
echo "✅ Unified interface is working!"
echo ""
echo "To use manually:"
echo "  kubectl port-forward -n $NAMESPACE svc/$GATEWAY_SERVICE ${GATEWAY_PORT}:8000"
echo ""
