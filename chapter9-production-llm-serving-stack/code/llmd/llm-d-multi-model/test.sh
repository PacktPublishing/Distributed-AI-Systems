#!/bin/bash
# Test script for llm-d multi-model deployment
# Tests Llama-3.2-1B and Qwen2.5-0.5B on vLLM

set -e

CLUSTER_NAME="llmd-cluster"

echo "=========================================="
echo "  llm-d Multi-Model Test"
echo "=========================================="
echo ""

# Check context
CURRENT_CONTEXT=$(kubectl config current-context 2>/dev/null || echo "")
if [[ "$CURRENT_CONTEXT" != "k3d-$CLUSTER_NAME" ]]; then
    echo "Switching to k3d-$CLUSTER_NAME context..."
    kubectl config use-context "k3d-$CLUSTER_NAME"
fi

echo "Using context: $(kubectl config current-context)"
echo ""

# Check pods
echo "Checking pod status..."
kubectl get pods
echo ""

# Test Llama model
echo "=========================================="
echo "Testing Llama-3.2-1B"
echo "=========================================="

kubectl port-forward svc/vllm-llama-32-1b 8001:8000 &
LLAMA_PF_PID=$!
sleep 3

echo "Health check..."
if curl -s -f http://localhost:8001/health > /dev/null; then
    echo "Health: OK"
else
    echo "Health: FAILED"
    kill $LLAMA_PF_PID 2>/dev/null || true
    exit 1
fi

echo "Chat completion test..."
RESPONSE=$(curl -s http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "messages": [{"role": "user", "content": "Say hello in one word."}],
    "max_tokens": 10
  }')

if echo "$RESPONSE" | grep -q "choices"; then
    echo "Response: $(echo "$RESPONSE" | jq -r '.choices[0].message.content' 2>/dev/null)"
    echo "Llama test: PASSED"
else
    echo "Llama test: FAILED"
    echo "$RESPONSE"
fi

kill $LLAMA_PF_PID 2>/dev/null || true
sleep 1
echo ""

# Test Qwen model
echo "=========================================="
echo "Testing Qwen2.5-0.5B"
echo "=========================================="

kubectl port-forward svc/vllm-qwen2-5-0-5b 8002:8000 &
QWEN_PF_PID=$!
sleep 3

echo "Health check..."
if curl -s -f http://localhost:8002/health > /dev/null; then
    echo "Health: OK"
else
    echo "Health: FAILED"
    kill $QWEN_PF_PID 2>/dev/null || true
    exit 1
fi

echo "Chat completion test..."
RESPONSE=$(curl -s http://localhost:8002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [{"role": "user", "content": "Say hello in one word."}],
    "max_tokens": 10
  }')

if echo "$RESPONSE" | grep -q "choices"; then
    echo "Response: $(echo "$RESPONSE" | jq -r '.choices[0].message.content' 2>/dev/null)"
    echo "Qwen test: PASSED"
else
    echo "Qwen test: FAILED"
    echo "$RESPONSE"
fi

kill $QWEN_PF_PID 2>/dev/null || true
echo ""

echo "=========================================="
echo "All tests completed!"
echo "=========================================="
