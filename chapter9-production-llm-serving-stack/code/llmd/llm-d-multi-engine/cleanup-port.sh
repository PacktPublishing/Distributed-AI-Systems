#!/bin/bash
# Clean up port 9000 for kubectl port-forward

echo "=== Cleaning up port 9000 ==="

# Method 1: Kill kubectl port-forward processes
echo "1. Killing kubectl port-forward processes..."
pkill -9 -f "kubectl.*port-forward.*9000" 2>/dev/null

# Method 2: Kill by port
echo "2. Killing processes on port 9000..."
lsof -ti:9000 | xargs kill -9 2>/dev/null

# Wait for port to be released
sleep 2

# Check if port is free
if lsof -i:9000 >/dev/null 2>&1; then
    echo "⚠️  Port 9000 is still in use:"
    lsof -i:9000
    echo ""
    echo "Try using a different port (e.g., 9001):"
    echo "  kubectl port-forward -n llm-d-multi-engine \${GATEWAY_POD} 9001:8000"
else
    echo "✅ Port 9000 is now free!"
fi
