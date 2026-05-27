#!/bin/bash
# Script to add 443 port mapping to existing k3d cluster
# This uses k3d node edit command (available in k3d 5.0.0+)
# No need to delete and recreate the cluster!

set -e

CLUSTER_NAME="mycluster-gpu"

echo "=========================================="
echo "Add 443 Port Mapping to Existing k3d Cluster"
echo "=========================================="
echo ""
echo "✅ Good news: No need to delete the cluster!"
echo "   Using k3d node edit command to dynamically add port mapping"
echo ""

# Check k3d version
K3D_VERSION=$(k3d --version 2>&1 | grep -oP 'v\d+\.\d+' | head -1)
echo "k3d version: $K3D_VERSION"
echo ""

# Check if cluster exists
if ! k3d cluster list | grep -q "$CLUSTER_NAME"; then
    echo "❌ Error: Cluster $CLUSTER_NAME does not exist"
    exit 1
fi

# Find loadbalancer node
LB_NAME=$(k3d node list 2>/dev/null | grep loadbalancer | awk '{print $1}' | head -1)

if [ -z "$LB_NAME" ]; then
    echo "❌ Error: Loadbalancer node not found"
    echo "   Please check cluster status: k3d cluster list"
    exit 1
fi

echo "Found loadbalancer: $LB_NAME"
echo ""

# Check current port mappings
echo "Current port mappings:"
k3d node list | grep "$LB_NAME" || echo "  No port mappings"
echo ""

# Check if port 443 is already mapped
if docker port $(docker ps -q --filter "name=$LB_NAME") 2>/dev/null | grep -q "443"; then
    echo "⚠️  Warning: Port 443 may already be mapped"
    echo "   Continuing may fail or cause conflicts"
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "❌ Operation cancelled"
        exit 1
    fi
fi

# Add port mapping
echo ""
echo "🚀 Adding port mapping 443:443..."
echo "Command: k3d node edit $LB_NAME --port-add 443:443"
echo ""

if k3d node edit "$LB_NAME" --port-add 443:443; then
    echo ""
    echo "✅ Port mapping added successfully!"
    echo ""
    echo "Verifying port mapping:"
    sleep 2
    docker port $(docker ps -q --filter "name=$LB_NAME") 2>/dev/null | grep 443 || echo "  Checking..."
    echo ""
    echo "📝 Next steps:"
    echo "   1. Ensure Ingress Controller is installed"
    echo "   2. Ensure Ingress and Gateway are deployed"
    echo "   3. Test access: curl -k https://localhost/v1/models"
else
    echo ""
    echo "❌ Failed to add port mapping"
    echo ""
    echo "Possible reasons:"
    echo "  - k3d version too old (requires 5.0.0+)"
    echo "  - Port 443 is already in use"
    echo "  - Insufficient permissions"
    echo ""
    echo "If it fails, you can try recreating the cluster (see cluster creation steps in main README)"
    exit 1
fi

