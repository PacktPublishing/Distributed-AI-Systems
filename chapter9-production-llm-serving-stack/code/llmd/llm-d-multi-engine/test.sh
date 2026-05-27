#!/bin/bash
# Test script for llm-d engine cluster deployments
# Tests both vLLM and SGLang servers for Qwen2.5-0.5B-Instruct

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLUSTER_NAME="llmd-multiengine"

echo "=========================================="
echo "  LLM-d Engine Cluster Test Script"
echo "=========================================="
echo ""

# Check if we're in the right context
CURRENT_CONTEXT=$(kubectl config current-context)
if [[ "$CURRENT_CONTEXT" != "k3d-$CLUSTER_NAME" ]]; then
    echo "⚠️  Warning: Current context is $CURRENT_CONTEXT"
    echo "   Expected: k3d-$CLUSTER_NAME"
    read -p "Do you want to switch to k3d-$CLUSTER_NAME? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        kubectl config use-context "k3d-$CLUSTER_NAME"
    else
        echo "❌ Please switch to the correct context first"
        exit 1
    fi
fi

echo "✅ Using context: $(kubectl config current-context)"
echo ""

# Check if pods exist
echo "📊 Checking pod status..."
VLLM_POD=$(kubectl get pod -l app=vllm,model=qwen2-5-0-5b -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")
SGLANG_POD=$(kubectl get pod -l app=sglang,model=qwen2-5-0-5b -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")

if [ -z "$VLLM_POD" ]; then
    echo "❌ vLLM pod not found"
    exit 1
fi

if [ -z "$SGLANG_POD" ]; then
    echo "❌ SGLang pod not found"
    exit 1
fi

echo "✅ vLLM pod: $VLLM_POD"
echo "✅ SGLang pod: $SGLANG_POD"
echo ""

# Check pod readiness
echo "⏳ Checking pod readiness..."
VLLM_READY=$(kubectl get pod "$VLLM_POD" -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null || echo "False")
SGLANG_READY=$(kubectl get pod "$SGLANG_POD" -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null || echo "False")

if [ "$VLLM_READY" != "True" ]; then
    echo "⚠️  vLLM pod is not ready yet. Current status:"
    kubectl get pod "$VLLM_POD"
    echo ""
    echo "View logs: kubectl logs -f $VLLM_POD"
    echo ""
fi

if [ "$SGLANG_READY" != "True" ]; then
    echo "⚠️  SGLang pod is not ready yet. Current status:"
    kubectl get pod "$SGLANG_POD"
    echo ""
    echo "View logs: kubectl logs -f $SGLANG_POD"
    echo ""
fi

# Test vLLM
echo "=========================================="
echo "Testing vLLM Server"
echo "=========================================="
echo ""

# Start port forward in background
echo "🔗 Starting port forward for vLLM (port 8001)..."
kubectl port-forward svc/vllm-qwen2-5-0-5b 8001:8000 > /dev/null 2>&1 &
VLLM_PF_PID=$!

# Wait for port forward to be ready
sleep 2

# Test health endpoint
echo "🏥 Testing health endpoint..."
if curl -s -f http://localhost:8001/health > /dev/null; then
    echo "✅ vLLM health check passed"
else
    echo "❌ vLLM health check failed"
    kill $VLLM_PF_PID 2>/dev/null || true
    exit 1
fi

# Test chat completion
echo "💬 Testing chat completion..."
VLLM_RESPONSE=$(curl -s http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [
      {"role": "user", "content": "Say hello in one word."}
    ],
    "max_tokens": 10
  }' || echo "")

if [ -n "$VLLM_RESPONSE" ] && echo "$VLLM_RESPONSE" | grep -q "choices"; then
    echo "✅ vLLM chat completion test passed"
    echo "   Response preview: $(echo "$VLLM_RESPONSE" | jq -r '.choices[0].message.content' 2>/dev/null || echo "N/A")"
else
    echo "❌ vLLM chat completion test failed"
    echo "   Response: $VLLM_RESPONSE"
fi

# Stop port forward
kill $VLLM_PF_PID 2>/dev/null || true
sleep 1
echo ""

# Test SGLang
echo "=========================================="
echo "Testing SGLang Server"
echo "=========================================="
echo ""

# Start port forward in background
echo "🔗 Starting port forward for SGLang (port 8002)..."
kubectl port-forward svc/sglang-qwen2-5-0-5b 8002:8000 > /dev/null 2>&1 &
SGLANG_PF_PID=$!

# Wait for port forward to be ready
sleep 2

# Test health endpoint
echo "🏥 Testing health endpoint..."
if curl -s -f http://localhost:8002/health > /dev/null; then
    echo "✅ SGLang health check passed"
else
    echo "❌ SGLang health check failed"
    kill $SGLANG_PF_PID 2>/dev/null || true
    exit 1
fi

# Test chat completion
echo "💬 Testing chat completion..."
SGLANG_RESPONSE=$(curl -s http://localhost:8002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [
      {"role": "user", "content": "Say hello in one word."}
    ],
    "max_tokens": 10
  }' || echo "")

if [ -n "$SGLANG_RESPONSE" ] && echo "$SGLANG_RESPONSE" | grep -q "choices"; then
    echo "✅ SGLang chat completion test passed"
    echo "   Response preview: $(echo "$SGLANG_RESPONSE" | jq -r '.choices[0].message.content' 2>/dev/null || echo "N/A")"
else
    echo "❌ SGLang chat completion test failed"
    echo "   Response: $SGLANG_RESPONSE"
fi

# Stop port forward
kill $SGLANG_PF_PID 2>/dev/null || true
sleep 1
echo ""

# Summary
echo "=========================================="
echo "  Test Summary"
echo "=========================================="
echo ""
echo "✅ All tests completed!"
echo ""
echo "💡 To test manually:"
echo "   # vLLM"
echo "   kubectl port-forward svc/vllm-qwen2-5-0-5b 8001:8000"
echo "   curl http://localhost:8001/health"
echo ""
echo "   # SGLang"
echo "   kubectl port-forward svc/sglang-qwen2-5-0-5b 8002:8000"
echo "   curl http://localhost:8002/health"
echo ""
