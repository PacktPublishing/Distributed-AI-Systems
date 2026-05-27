#!/bin/bash
# Test script for unified interface (API Gateway)
# This script tests the API Gateway that routes requests to vLLM and SGLang

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GATEWAY_SERVICE="engine-comparison-gateway"
GATEWAY_PORT=8000
NAMESPACE="${NAMESPACE:-default}"

echo "=========================================="
echo "  Testing Unified Interface (API Gateway)"
echo "=========================================="
echo ""

# Check if Gateway is deployed
if ! kubectl get deployment $GATEWAY_SERVICE -n $NAMESPACE &>/dev/null; then
    echo "⚠️  Gateway not found. Deploying..."
    cd "$SCRIPT_DIR"
    ./deploy-gateway.sh
    sleep 5
fi

# Check if Gateway pod is ready
GATEWAY_POD=$(kubectl get pod -n $NAMESPACE -l app=$GATEWAY_SERVICE -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
if [ -z "$GATEWAY_POD" ]; then
    echo "❌ Gateway pod not found"
    exit 1
fi

GATEWAY_READY=$(kubectl get pod "$GATEWAY_POD" -n $NAMESPACE -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null || echo "False")
if [ "$GATEWAY_READY" != "True" ]; then
    echo "⚠️  Gateway pod is not ready yet. Current status:"
    kubectl get pod "$GATEWAY_POD" -n $NAMESPACE
    echo "⏳ Waiting for Gateway to be ready..."
    kubectl wait --for=condition=ready pod "$GATEWAY_POD" -n $NAMESPACE --timeout=120s || {
        echo "❌ Gateway pod failed to become ready"
        kubectl logs "$GATEWAY_POD" -n $NAMESPACE --tail=50
        exit 1
    }
fi

echo "✅ Gateway pod: $GATEWAY_POD"
echo ""

# Start port forward
echo "🔗 Starting port forward for Gateway (port $GATEWAY_PORT)..."
kubectl port-forward -n $NAMESPACE svc/$GATEWAY_SERVICE ${GATEWAY_PORT}:8000 > /dev/null 2>&1 &
GATEWAY_PF_PID=$!

# Wait for port forward to be ready
sleep 3

# Test health endpoint
echo "🏥 Testing Gateway health endpoint..."
if curl -s -f http://localhost:${GATEWAY_PORT}/health > /dev/null; then
    echo "✅ Gateway health check passed"
else
    echo "❌ Gateway health check failed"
    kill $GATEWAY_PF_PID 2>/dev/null || true
    exit 1
fi

# Trigger service discovery
echo ""
echo "🔍 Triggering service discovery..."
DISCOVERY_RESPONSE=$(curl -s -X POST http://localhost:${GATEWAY_PORT}/admin/discover)
if echo "$DISCOVERY_RESPONSE" | grep -q "discovered"; then
    echo "✅ Service discovery completed"
    echo "$DISCOVERY_RESPONSE" | jq '.' 2>/dev/null || echo "$DISCOVERY_RESPONSE"
else
    echo "⚠️  Service discovery may have issues:"
    echo "$DISCOVERY_RESPONSE"
fi

# List available models
echo ""
echo "📋 Listing available models..."
MODELS_RESPONSE=$(curl -s http://localhost:${GATEWAY_PORT}/v1/models)
if echo "$MODELS_RESPONSE" | grep -q "data"; then
    echo "✅ Models endpoint working"
    echo "$MODELS_RESPONSE" | jq '.data[] | {id, owned_by}' 2>/dev/null || echo "$MODELS_RESPONSE"
else
    echo "⚠️  Models endpoint response:"
    echo "$MODELS_RESPONSE"
fi

# Test vLLM
echo ""
echo "=========================================="
echo "Testing vLLM via Gateway"
echo "=========================================="
echo ""

VLLM_RESPONSE=$(curl -s http://localhost:${GATEWAY_PORT}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "owned_by": "vllm",
    "messages": [
      {"role": "user", "content": "Say hello in one word."}
    ],
    "max_tokens": 10
  }' || echo "")

if [ -n "$VLLM_RESPONSE" ] && echo "$VLLM_RESPONSE" | grep -q "choices"; then
    echo "✅ vLLM chat completion test passed"
    echo "   Response: $(echo "$VLLM_RESPONSE" | jq -r '.choices[0].message.content' 2>/dev/null || echo "N/A")"
else
    echo "❌ vLLM chat completion test failed"
    echo "   Response: $VLLM_RESPONSE"
fi

# Test SGLang
echo ""
echo "=========================================="
echo "Testing SGLang via Gateway"
echo "=========================================="
echo ""

SGLANG_RESPONSE=$(curl -s http://localhost:${GATEWAY_PORT}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "owned_by": "sglang",
    "messages": [
      {"role": "user", "content": "Say hello in one word."}
    ],
    "max_tokens": 10
  }' || echo "")

if [ -n "$SGLANG_RESPONSE" ] && echo "$SGLANG_RESPONSE" | grep -q "choices"; then
    echo "✅ SGLang chat completion test passed"
    echo "   Response: $(echo "$SGLANG_RESPONSE" | jq -r '.choices[0].message.content' 2>/dev/null || echo "N/A")"
else
    echo "❌ SGLang chat completion test failed"
    echo "   Response: $SGLANG_RESPONSE"
fi

# Test with header
echo ""
echo "=========================================="
echo "Testing routing via HTTP header"
echo "=========================================="
echo ""

HEADER_RESPONSE=$(curl -s http://localhost:${GATEWAY_PORT}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "x-owned-by: vllm" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [
      {"role": "user", "content": "Say hello in one word."}
    ],
    "max_tokens": 10
  }' || echo "")

if [ -n "$HEADER_RESPONSE" ] && echo "$HEADER_RESPONSE" | grep -q "choices"; then
    echo "✅ Header-based routing test passed"
    echo "   Response: $(echo "$HEADER_RESPONSE" | jq -r '.choices[0].message.content' 2>/dev/null || echo "N/A")"
else
    echo "⚠️  Header-based routing test:"
    echo "   Response: $HEADER_RESPONSE"
fi

# Check routing configuration
echo ""
echo "=========================================="
echo "Routing Configuration"
echo "=========================================="
echo ""
ROUTING_CONFIG=$(curl -s http://localhost:${GATEWAY_PORT}/admin/api/routing)
echo "$ROUTING_CONFIG" | jq '.' 2>/dev/null || echo "$ROUTING_CONFIG"

# Cleanup
kill $GATEWAY_PF_PID 2>/dev/null || true
sleep 1

# Summary
echo ""
echo "=========================================="
echo "  Test Summary"
echo "=========================================="
echo ""
echo "✅ Unified interface tests completed!"
echo ""
echo "💡 To use the Gateway manually:"
echo "   kubectl port-forward -n $NAMESPACE svc/$GATEWAY_SERVICE ${GATEWAY_PORT}:8000"
echo ""
echo "   # Test vLLM"
echo "   curl -X POST http://localhost:${GATEWAY_PORT}/v1/chat/completions \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"model\": \"Qwen/Qwen2.5-0.5B-Instruct\", \"owned_by\": \"vllm\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}]}'"
echo ""
echo "   # Test SGLang"
echo "   curl -X POST http://localhost:${GATEWAY_PORT}/v1/chat/completions \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"model\": \"Qwen/Qwen2.5-0.5B-Instruct\", \"owned_by\": \"sglang\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}]}'"
echo ""
