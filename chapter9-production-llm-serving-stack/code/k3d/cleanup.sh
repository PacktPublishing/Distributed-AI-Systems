#!/bin/bash
set -e

CLUSTER_NAME="${CLUSTER_NAME:-mycluster-gpu}"
IMAGE_NAME="${IMAGE_NAME:-k3s-cuda:v1.33.6-cuda-12.2.0}"

echo "Cleaning up k3d cluster and resources..."
echo ""

# Delete cluster
if k3d cluster list | grep -q "$CLUSTER_NAME"; then
    echo "Deleting cluster: $CLUSTER_NAME"
    k3d cluster delete "$CLUSTER_NAME"
    echo "✓ Cluster deleted"
else
    echo "Cluster $CLUSTER_NAME not found"
fi

# Optionally remove the custom image
read -p "Do you want to remove the custom image $IMAGE_NAME? (y/N) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if docker images | grep -q "$IMAGE_NAME"; then
        echo "Removing image: $IMAGE_NAME"
        docker rmi "$IMAGE_NAME" || true
        echo "✓ Image removed"
    else
        echo "Image $IMAGE_NAME not found"
    fi
fi

echo ""
echo "Cleanup complete!"
