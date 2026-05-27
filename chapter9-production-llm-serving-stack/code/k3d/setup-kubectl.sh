#!/bin/bash
set -e

CLUSTER_NAME="${CLUSTER_NAME:-mycluster-gpu}"

echo "Configuring kubectl for cluster: $CLUSTER_NAME"
echo ""

# Merge k3d kubeconfig
echo "Merging k3d kubeconfig..."
k3d kubeconfig merge "$CLUSTER_NAME" --kubeconfig-merge-default

# Set KUBECONFIG
export KUBECONFIG=$HOME/.kube/config

# Verify access
echo "Verifying cluster access..."
if ! kubectl get nodes &> /dev/null; then
    echo "WARNING: Initial kubectl access failed. Attempting to fix kubeconfig..."
    
    # Check current server address
    KUBE_SERVER=$(kubectl config view -o jsonpath="{.clusters[?(@.name==\"k3d-$CLUSTER_NAME\")].cluster.server}" 2>/dev/null || echo "")
    
    if [ -n "$KUBE_SERVER" ]; then
        # If it shows 0.0.0.0, update to 127.0.0.1
        if echo "$KUBE_SERVER" | grep -q "0.0.0.0"; then
            echo "Fixing server address from 0.0.0.0 to 127.0.0.1..."
            NEW_SERVER=$(echo "$KUBE_SERVER" | sed 's/0.0.0.0/127.0.0.1/')
            kubectl config set-cluster "k3d-$CLUSTER_NAME" --server="$NEW_SERVER"
        fi
    fi
    
    # Try again
    if kubectl get nodes &> /dev/null; then
        echo "✓ kubectl access fixed"
    else
        echo "ERROR: Could not configure kubectl. Please check cluster status:"
        echo "  k3d cluster list"
        exit 1
    fi
else
    echo "✓ kubectl access verified"
fi

# Show nodes
echo ""
echo "Cluster nodes:"
kubectl get nodes

echo ""
echo "✓ kubectl configured successfully!"
echo ""
echo "Next step: Verify GPU with ./verify-gpu.sh"
