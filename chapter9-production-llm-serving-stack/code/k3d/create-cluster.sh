#!/bin/bash
set -e

# Check prerequisites
if ! command -v k3d &> /dev/null; then
    echo "ERROR: k3d not found. Run ./install-prerequisites.sh first."
    exit 1
fi

if ! docker info &> /dev/null; then
    echo "ERROR: Docker is not running. Please start Docker first."
    exit 1
fi

# Default values
CLUSTER_NAME="${CLUSTER_NAME:-mycluster-gpu}"
GPUS="${GPUS:-all}"
MODEL_PATH="${MODEL_PATH:-}"

# Auto-detect IMAGE_NAME from available k3s-cuda images if not specified
if [ -z "$IMAGE_NAME" ]; then
    IMAGE_NAME=$(docker images --format "{{.Repository}}:{{.Tag}}" | grep "^k3s-cuda:" | head -1)
    if [ -z "$IMAGE_NAME" ]; then
        echo "ERROR: No k3s-cuda image found!"
        echo "Please build it first with: ./build.sh"
        exit 1
    fi
    echo "Auto-detected image: $IMAGE_NAME"
fi

echo ""
echo "Creating k3d GPU cluster..."
echo "  Cluster name: $CLUSTER_NAME"
echo "  Image: $IMAGE_NAME"
echo "  GPUs: $GPUS"
if [ -n "$MODEL_PATH" ]; then
    echo "  Model path: $MODEL_PATH -> /models"
fi
echo ""

# Check if image exists
if ! docker images --format "{{.Repository}}:{{.Tag}}" | grep -q "^${IMAGE_NAME}$"; then
    echo "ERROR: Image $IMAGE_NAME not found!"
    echo "Please build it first with: ./build.sh"
    exit 1
fi

# Handle existing cluster
if k3d cluster list 2>/dev/null | grep -q "$CLUSTER_NAME"; then
    echo "Cluster $CLUSTER_NAME already exists."
    read -p "Delete and recreate? [y/N] " confirm
    if [[ "$confirm" =~ ^[Yy]$ ]]; then
        echo "Deleting existing cluster..."
        k3d cluster delete "$CLUSTER_NAME"
        sleep 2
    else
        echo "Cancelled."
        exit 0
    fi
fi

# Build cluster creation command
CMD="k3d cluster create $CLUSTER_NAME \
  --image $IMAGE_NAME \
  --gpus=$GPUS \
  --servers 1 \
  --agents 1"

# Add volume mount if MODEL_PATH is specified
if [ -n "$MODEL_PATH" ]; then
    CMD="$CMD --volume \"${MODEL_PATH}:/models\""
fi

echo "Creating cluster..."
eval "$CMD"

echo ""
echo "✓ Cluster created successfully!"
echo ""
echo "Verify the cluster (device plugin may take 30-60s to register GPUs):"
echo "  kubectl get nodes"
echo "  sleep 45 && kubectl describe node | grep nvidia"
echo "  # Or run: ./verify-gpu.sh"
echo ""
echo "To delete the cluster:"
echo "  k3d cluster delete $CLUSTER_NAME"
