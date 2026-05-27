#!/bin/bash
# Benchmark comparison script for vLLM vs SGLang
# This script sends requests to both engines and compares performance

set -e

GATEWAY_SERVICE="engine-comparison-gateway"
GATEWAY_PORT=8000
MODEL="Qwen/Qwen2.5-0.5B-Instruct"
NUM_REQUESTS=${NUM_REQUESTS:-50}
CONCURRENT=${CONCURRENT:-5}

echo "=========================================="
echo "  Engine Performance Comparison"
echo "=========================================="
echo ""
echo "Model: $MODEL"
echo "Requests per engine: $NUM_REQUESTS"
echo "Concurrent requests: $CONCURRENT"
echo ""

# Check if gateway is available
if ! kubectl get svc $GATEWAY_SERVICE &>/dev/null; then
    echo "⚠️  Gateway service not found. Deploying..."
    kubectl apply -f api-gateway.yaml
    echo "⏳ Waiting for gateway to be ready..."
    kubectl wait --for=condition=ready pod -l app=$GATEWAY_SERVICE --timeout=60s
    sleep 5
fi

# Start port forward
echo "🔗 Starting port forward for gateway..."
kubectl port-forward svc/$GATEWAY_SERVICE ${GATEWAY_PORT}:8000 > /dev/null 2>&1 &
GATEWAY_PF=$!
sleep 3

# Test function
test_engine() {
    local engine=$1
    local output_file="/tmp/benchmark_${engine}.json"
    
    echo "Testing $engine..."
    
    local start_time=$(date +%s.%N)
    local success=0
    local failed=0
    
    for i in $(seq 1 $NUM_REQUESTS); do
        if curl -s -f http://localhost:${GATEWAY_PORT}/v1/chat/completions \
          -H "Content-Type: application/json" \
          -d "{
            \"model\": \"$MODEL\",
            \"inference_server\": \"$engine\",
            \"messages\": [
              {\"role\": \"user\", \"content\": \"Say hello in one word. Request $i\"}
            ],
            \"max_tokens\": 10
          }" > /dev/null; then
            ((success++))
        else
            ((failed++))
        fi
        
        # Show progress
        if [ $((i % 10)) -eq 0 ]; then
            echo -n "."
        fi
    done
    
    local end_time=$(date +%s.%N)
    local duration=$(echo "$end_time - $start_time" | bc)
    local rps=$(echo "scale=2; $NUM_REQUESTS / $duration" | bc)
    
    echo ""
    echo "✅ $engine Results:"
    echo "   Success: $success/$NUM_REQUESTS"
    echo "   Failed: $failed/$NUM_REQUESTS"
    echo "   Duration: ${duration}s"
    echo "   Requests/sec: $rps"
    echo ""
    
    echo "$engine,$success,$failed,$duration,$rps" >> /tmp/benchmark_results.csv
}

# Create results file
echo "Engine,Success,Failed,Duration(s),RPS" > /tmp/benchmark_results.csv

# Test vLLM
echo "=========================================="
echo "Testing vLLM"
echo "=========================================="
test_engine "vllm"

# Wait a bit between tests
sleep 2

# Test SGLang
echo "=========================================="
echo "Testing SGLang"
echo "=========================================="
test_engine "sglang"

# Cleanup
kill $GATEWAY_PF 2>/dev/null || true

# Summary
echo "=========================================="
echo "  Comparison Summary"
echo "=========================================="
echo ""
cat /tmp/benchmark_results.csv | column -t -s,
echo ""
echo "💡 To test manually:"
echo "   kubectl port-forward svc/$GATEWAY_SERVICE ${GATEWAY_PORT}:8000"
echo ""
echo "   # Test vLLM"
echo "   curl http://localhost:${GATEWAY_PORT}/v1/chat/completions \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"model\": \"$MODEL\", \"inference_server\": \"vllm\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}]}'"
echo ""
echo "   # Test SGLang"
echo "   curl http://localhost:${GATEWAY_PORT}/v1/chat/completions \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"model\": \"$MODEL\", \"inference_server\": \"sglang\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello\"}]}'"
echo ""
