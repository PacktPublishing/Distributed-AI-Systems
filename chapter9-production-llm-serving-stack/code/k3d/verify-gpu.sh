#!/bin/bash
set -e

CLUSTER_NAME="${CLUSTER_NAME:-mycluster-gpu}"

echo "Verifying GPU visibility in cluster: $CLUSTER_NAME"
echo ""

# Set KUBECONFIG
export KUBECONFIG=$HOME/.kube/config

# Wait for device plugin to be ready
echo "Waiting for NVIDIA device plugin to be ready..."
if kubectl wait --for=condition=ready pod -l name=nvidia-device-plugin-ds -n kube-system --timeout=120s 2>/dev/null; then
    echo "✓ Device plugin is ready"
else
    echo "WARNING: Device plugin may not be ready yet. Continuing..."
fi

# Verify device plugin pods are running
echo ""
echo "Checking device plugin pods:"
kubectl get pods -n kube-system | grep nvidia || echo "No NVIDIA pods found (this may be normal if plugin hasn't started)"

# Check GPU resources on nodes
echo ""
echo "Checking GPU resources on nodes:"
echo ""

SERVER_NODE="k3d-${CLUSTER_NAME}-server-0"
AGENT_NODE="k3d-${CLUSTER_NAME}-agent-0"

if kubectl get node "$SERVER_NODE" &> /dev/null; then
    echo "Server node ($SERVER_NODE):"
    kubectl describe node "$SERVER_NODE" | grep -A 2 "nvidia.com/gpu" || echo "  No GPU resources found"
    echo ""
fi

if kubectl get node "$AGENT_NODE" &> /dev/null; then
    echo "Agent node ($AGENT_NODE):"
    kubectl describe node "$AGENT_NODE" | grep -A 2 "nvidia.com/gpu" || echo "  No GPU resources found"
    echo ""
fi

# Test GPU access
echo "Testing GPU access with test pod..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
kubectl apply -f "$SCRIPT_DIR/gpu-test.yaml"

echo "Waiting for GPU test pod to complete..."
if kubectl wait --for=condition=Ready pod/gpu-test --timeout=60s 2>/dev/null; then
    echo ""
    echo "GPU test pod logs:"
    kubectl logs gpu-test
    echo ""
    echo "✓ GPU access verified!"
    
    # Cleanup
    kubectl delete pod gpu-test --ignore-not-found=true
else
    echo "WARNING: GPU test pod did not complete. Check logs:"
    echo "  kubectl logs gpu-test"
    echo "  kubectl describe pod gpu-test"
fi

echo ""
echo "GPU verification complete!"
