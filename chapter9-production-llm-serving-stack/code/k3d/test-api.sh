#!/bin/bash
# Test script for vLLM OpenAI-compatible API
# 
# Usage:
#   ./test-api.sh [API_URL]
# 
# Examples:
#   # Using API Gateway (automatic routing)
#   ./test-api.sh http://localhost:8000
#   kubectl port-forward svc/vllm-api-gateway 8000:8000
#   
#   # Direct access to specific model
#   ./test-api.sh http://localhost:8001
#   kubectl port-forward svc/vllm-llama-32-1b 8001:8000

set -e

API_URL="${1:-http://localhost:8000}"
echo "Testing vLLM API at: $API_URL"
echo "=================================="
echo ""
echo "💡 Tip: If using API Gateway, requests are automatically routed based on 'model' field in request body"
echo ""

# Test health
echo "1. Testing health endpoint..."
echo "   GET $API_URL/health"
HEALTH_RESPONSE=$(curl --max-time 5 -s -w "\nHTTP_CODE:%{http_code}" "$API_URL/health" || echo "HTTP_CODE:000")
HTTP_CODE=$(echo "$HEALTH_RESPONSE" | grep "HTTP_CODE" | cut -d: -f2)
BODY=$(echo "$HEALTH_RESPONSE" | grep -v "HTTP_CODE")
if [ "$HTTP_CODE" = "200" ]; then
    echo "   ✅ Health check passed"
    echo "$BODY" | python3 -m json.tool 2>/dev/null || echo "$BODY"
else
    echo "   ❌ Health check failed (HTTP $HTTP_CODE)"
    echo "$BODY"
fi
echo ""

# List models
echo "2. Listing available models..."
echo "   GET $API_URL/v1/models"
MODELS_RESPONSE=$(curl --max-time 5 -s -w "\nHTTP_CODE:%{http_code}" "$API_URL/v1/models" || echo "HTTP_CODE:000")
HTTP_CODE=$(echo "$MODELS_RESPONSE" | grep "HTTP_CODE" | cut -d: -f2)
BODY=$(echo "$MODELS_RESPONSE" | grep -v "HTTP_CODE")
if [ "$HTTP_CODE" = "200" ]; then
    echo "   ✅ Models listed successfully"
    echo "$BODY" | python3 -m json.tool 2>/dev/null || echo "$BODY"
else
    echo "   ❌ List models failed (HTTP $HTTP_CODE)"
    echo "$BODY"
fi
echo ""

# Test completion
echo "3. Testing completion endpoint..."
echo "   POST $API_URL/v1/completions"
COMPLETION_RESPONSE=$(curl --max-time 30 -s -w "\nHTTP_CODE:%{http_code}" "$API_URL/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "prompt": "Which is more beautiful—the Gaussian integral or Euler formula, and why, in one sentence?",
    "max_tokens": 50,
    "temperature": 0.7
  }' || echo "HTTP_CODE:000")
HTTP_CODE=$(echo "$COMPLETION_RESPONSE" | grep "HTTP_CODE" | cut -d: -f2)
BODY=$(echo "$COMPLETION_RESPONSE" | grep -v "HTTP_CODE")
if [ "$HTTP_CODE" = "200" ]; then
    echo "   ✅ Completion successful"
    echo "$BODY" | python3 -m json.tool 2>/dev/null || echo "$BODY"
else
    echo "   ❌ Completion failed (HTTP $HTTP_CODE)"
    echo "$BODY"
fi
echo ""

# Test chat (if supported)
echo "4. Testing chat completion endpoint..."
echo "   POST $API_URL/v1/chat/completions"
CHAT_RESPONSE=$(curl --max-time 30 -s -w "\nHTTP_CODE:%{http_code}" "$API_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "messages": [
      {"role": "user", "content": "Which is more beautiful—the Gaussian integral or Euler formula, and why, in one sentence?"}
    ],
    "max_tokens": 50
  }' || echo "HTTP_CODE:000")
HTTP_CODE=$(echo "$CHAT_RESPONSE" | grep "HTTP_CODE" | cut -d: -f2)
BODY=$(echo "$CHAT_RESPONSE" | grep -v "HTTP_CODE")
if [ "$HTTP_CODE" = "200" ]; then
    echo "   ✅ Chat completion successful"
    echo "$BODY" | python3 -m json.tool 2>/dev/null || echo "$BODY"
else
    echo "   ⚠️  Chat completion failed (HTTP $HTTP_CODE) - may not be supported by this model"
    echo "$BODY"
fi
echo ""

echo ""
echo "5. Testing different model (Phi-tiny-MoE) via Gateway..."
echo "   POST $API_URL/v1/chat/completions"
echo "   (This will be automatically routed to phi-tiny-moe service based on model field)"
PHI_RESPONSE=$(curl --max-time 30 -s -w "\nHTTP_CODE:%{http_code}" "$API_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/models/Phi-tiny-MoE-instruct",
    "messages": [
      {"role": "user", "content": "Which is more beautiful—the Gaussian integral or Euler formula, and why, in one sentence?"}
    ],
    "max_tokens": 50
  }' || echo "HTTP_CODE:000")
HTTP_CODE=$(echo "$PHI_RESPONSE" | grep "HTTP_CODE" | cut -d: -f2)
BODY=$(echo "$PHI_RESPONSE" | grep -v "HTTP_CODE")
if [ "$HTTP_CODE" = "200" ]; then
    echo "   ✅ Phi-tiny-MoE chat completion successful (routed automatically)"
    echo "$BODY" | python3 -m json.tool 2>/dev/null || echo "$BODY"
else
    echo "   ⚠️  Phi-tiny-MoE chat completion failed (HTTP $HTTP_CODE)"
    echo "$BODY"
fi
echo ""

echo "=================================="
echo "Testing complete!"
echo ""
echo "📝 Note: API Gateway automatically routes requests based on 'model' and 'owned_by' fields:"
echo "   - 'meta-llama/Llama-3.2-1B-Instruct' with 'owned_by: vllm' → vllm-llama-32-1b"
echo "   - 'meta-llama/Llama-3.2-1B-Instruct' with 'owned_by: sglang' → sglang-llama-32-1b"
echo "   - '/models/Phi-tiny-MoE-instruct' → vllm-phi-tiny-moe-service"
