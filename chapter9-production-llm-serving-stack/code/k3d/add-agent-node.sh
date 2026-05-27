#!/bin/bash
# Add a new agent node to k3d cluster with GPU support
# This script adds agent-2 node for TensorRT-LLM deployment

set -e

# Default values
CLUSTER_NAME="${CLUSTER_NAME:-mycluster-gpu}"
# Default to the working image tag (matches README.md and actual usage)
IMAGE_NAME="${IMAGE_NAME:-k3s-cuda:v1.33.6-cuda-12.2.0-working}"
AGENT_NAME="${AGENT_NAME:-agent-2}"
MODEL_PATH="${MODEL_PATH:-/raid/models}"

echo "=== Adding new Agent node to k3d cluster ==="
echo "Cluster: $CLUSTER_NAME"
echo "Agent name: $AGENT_NAME"
echo "Image: $IMAGE_NAME"
echo ""

# Check if cluster exists
if ! k3d cluster list | grep -q "$CLUSTER_NAME"; then
    echo "❌ Error: Cluster $CLUSTER_NAME not found"
    echo "Please create the cluster first with: ./create-cluster.sh"
    exit 1
fi

# Check if image exists (use docker images with format to check properly)
IMAGE_EXISTS=$(docker images "$IMAGE_NAME" --format "{{.Repository}}:{{.Tag}}" 2>/dev/null | head -1)
if [ -z "$IMAGE_EXISTS" ]; then
    echo "❌ Error: Image $IMAGE_NAME not found!"
    echo ""
    echo "Available k3s-cuda images:"
    docker images | grep -E "k3s-cuda|cuda" || echo "  (none found)"
    echo ""
    echo "Please either:"
    echo "  1. Build the image: ./build.sh"
    echo "  2. Set IMAGE_NAME environment variable: IMAGE_NAME=your-image-name ./add-agent-node.sh"
    echo "  3. Use an existing image from the list above (format: repository:tag)"
    exit 1
fi

# Check if agent node already exists
if docker ps -a --format "{{.Names}}" | grep -q "k3d-$AGENT_NAME-0"; then
    echo "⚠️  Agent node k3d-$AGENT_NAME-0 already exists"
    read -p "Do you want to remove and recreate it? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🗑️  Removing existing agent node..."
        docker stop "k3d-$AGENT_NAME-0" 2>/dev/null || true
        docker rm "k3d-$AGENT_NAME-0" 2>/dev/null || true
    else
        echo "❌ Aborted"
        exit 1
    fi
fi

# Step 1: Use k3d node create first (this sets up proper k3d configuration)
echo ""
echo "📦 Step 1: Creating node with k3d (without GPU support)..."
# Check if node already exists in k3d
if k3d node list 2>/dev/null | grep -q "$AGENT_NAME"; then
    echo "  ℹ️  Node $AGENT_NAME already exists in k3d, removing it first..."
    k3d node delete "$AGENT_NAME" 2>/dev/null || true
    sleep 2
fi

# Create node with k3d (correct syntax: k3d node create NAME --cluster CLUSTER)
echo "  Creating node: $AGENT_NAME"
k3d node create "$AGENT_NAME" \
  --cluster "$CLUSTER_NAME" \
  --role agent \
  --wait 10 2>&1 | grep -E "(INFO|WARN|ERROR|created|ready)" || true

K3D_CREATE_EXIT=$?
if [ $K3D_CREATE_EXIT -ne 0 ]; then
    echo "  ⚠️  k3d node create had issues (exit code: $K3D_CREATE_EXIT)"
    echo "  Continuing with manual setup..."
else
    echo "  ✅ Node created with k3d"
fi

# Step 2: Stop and remove the container created by k3d
echo ""
echo "🔄 Step 2: Reconfiguring container for GPU support..."
if docker ps -a --format "{{.Names}}" | grep -q "k3d-$AGENT_NAME-0"; then
    echo "  🛑 Stopping k3d-created container..."
    docker stop "k3d-$AGENT_NAME-0" 2>/dev/null || true
    echo "  🗑️  Removing k3d-created container..."
    docker rm "k3d-$AGENT_NAME-0" 2>/dev/null || true
    sleep 2
fi

# Get cluster network name
CLUSTER_NETWORK="k3d-$CLUSTER_NAME"

# Get K3S_TOKEN from existing agent node
echo "🔑 Getting K3S_TOKEN from cluster..."
# Extract just the token part (k3s token format: TOKEN::server:TOKEN, we need just TOKEN)
FULL_TOKEN=$(docker exec k3d-$CLUSTER_NAME-server-0 cat /var/lib/rancher/k3s/server/node-token 2>/dev/null || echo "")
if [ -n "$FULL_TOKEN" ]; then
    # Extract the last part after the last colon (the actual token)
    K3S_TOKEN=$(echo "$FULL_TOKEN" | awk -F'::' '{print $NF}' | awk -F':' '{print $NF}')
    # If extraction failed, use the full token
    if [ -z "$K3S_TOKEN" ] || [ "$K3S_TOKEN" = "$FULL_TOKEN" ]; then
        K3S_TOKEN="$FULL_TOKEN"
    fi
else
    echo "⚠️  Could not get K3S_TOKEN from server, using default"
    K3S_TOKEN="LmYHFPGciNataclGfjAI"
fi

# Get server URL
SERVER_NAME="k3d-$CLUSTER_NAME-server-0"
SERVER_URL="https://$SERVER_NAME:6443"

# Get image ID
echo "📦 Getting image ID..."
IMAGE_ID=$(docker images "$IMAGE_NAME" --format "{{.ID}}" | head -1)
if [ -z "$IMAGE_ID" ]; then
    echo "❌ Error: Could not find image $IMAGE_NAME"
    exit 1
fi
echo "   Image ID: $IMAGE_ID"

# Create agent node with GPU support
echo ""
echo "🚀 Creating agent node: k3d-$AGENT_NAME-0"
echo "   Network: $CLUSTER_NETWORK"
echo "   Server URL: $SERVER_URL"
echo "   Model path: $MODEL_PATH -> /models"
echo ""

docker run -d \
  --name "k3d-$AGENT_NAME-0" \
  --hostname "k3d-$AGENT_NAME-0" \
  --network "$CLUSTER_NETWORK" \
  --privileged \
  --tmpfs /run \
  --tmpfs /var/run \
  -e K3S_TOKEN="$K3S_TOKEN" \
  -e K3S_URL="$SERVER_URL" \
  -e K3S_KUBECONFIG_OUTPUT=/output/kubeconfig.yaml \
  -v "$MODEL_PATH:/models" \
  -v "k3d-$CLUSTER_NAME-images:/k3d/images" \
  --label k3d.cluster="$CLUSTER_NAME" \
  --label k3d.role=agent \
  --gpus all \
  --restart unless-stopped \
  "$IMAGE_ID" \
  agent --with-node-id

if [ $? -eq 0 ]; then
    echo "✅ Agent node created successfully"
else
    echo "❌ Failed to create agent node"
    exit 1
fi

# Wait for node to be ready
echo ""
echo "⏳ Waiting for node to be ready (this may take 30-60 seconds)..."
echo "   Note: Containerd initialization may take some time"
sleep 10

# Wait for node to appear in Kubernetes
MAX_WAIT=120
WAITED=0
while [ $WAITED -lt $MAX_WAIT ]; do
    # Check if node appears (with or without suffix)
    if kubectl get nodes 2>/dev/null | grep -q "k3d-$AGENT_NAME-0"; then
        echo "   ✅ Node detected in cluster!"
        break
    fi
    # Also check container logs for errors
    if [ $((WAITED % 20)) -eq 0 ] && [ $WAITED -gt 0 ]; then
        echo "   Checking container status..."
        CONTAINER_STATUS=$(docker inspect k3d-$AGENT_NAME-0 --format '{{.State.Status}}' 2>/dev/null || echo "not-found")
        if [ "$CONTAINER_STATUS" != "running" ]; then
            echo "   ⚠️  Container status: $CONTAINER_STATUS"
            echo "   Recent logs:"
            docker logs k3d-$AGENT_NAME-0 2>&1 | tail -5 | sed 's/^/      /'
        fi
    fi
    echo "   Waiting for node to appear in cluster... ($WAITED/$MAX_WAIT seconds)"
    sleep 2
    WAITED=$((WAITED + 2))
done

# Get the actual node name (with suffix from --with-node-id)
echo ""
echo "🔍 Detecting node name..."
NEW_NODE=$(kubectl get nodes -o json 2>/dev/null | \
    jq -r ".items[] | select(.metadata.name | startswith(\"k3d-$AGENT_NAME-0\")) | .metadata.name" | head -1)

if [ -z "$NEW_NODE" ]; then
    echo "⚠️  Could not detect node name automatically"
    echo "   Please check manually: kubectl get nodes"
    exit 1
fi

echo "✅ Node detected: $NEW_NODE"

# Wait for node to be ready
echo ""
echo "⏳ Waiting for node to be Ready..."
kubectl wait --for=condition=Ready node/"$NEW_NODE" --timeout=120s 2>/dev/null || true

# Check GPU capacity
echo ""
echo "🎮 Checking GPU capacity..."
sleep 5  # Wait for device plugin to detect GPU
GPU_CAPACITY=$(kubectl get node "$NEW_NODE" -o jsonpath='{.status.capacity.nvidia\.com/gpu}' 2>/dev/null || echo "0")
GPU_ALLOCATABLE=$(kubectl get node "$NEW_NODE" -o jsonpath='{.status.allocatable.nvidia\.com/gpu}' 2>/dev/null || echo "0")

echo "   GPU Capacity: $GPU_CAPACITY"
echo "   GPU Allocatable: $GPU_ALLOCATABLE"

if [ "$GPU_CAPACITY" = "0" ] || [ -z "$GPU_CAPACITY" ]; then
    echo "⚠️  Warning: GPU not detected yet. This may take 20-30 seconds."
    echo "   Run 'kubectl get node $NEW_NODE -o yaml' to check later"
fi

# Show node status
echo ""
echo "📊 Node status:"
kubectl get node "$NEW_NODE" -o wide

echo ""
echo "✅ Agent node added successfully!"
echo ""
echo "📝 Node name: $NEW_NODE"
echo ""

# Update TensorRT configuration if it exists
TENSORRT_YAML="$(dirname "$0")/tensorrt/llama-3.2-1b.yaml"
if [ -f "$TENSORRT_YAML" ]; then
    echo "🔧 Updating TensorRT configuration with node name..."
    
    # Check if nodeSelector already exists
    if grep -q "kubernetes.io/hostname:" "$TENSORRT_YAML"; then
        # Update existing nodeSelector (match the line with hostname, preserve indentation)
        if [[ "$OSTYPE" == "darwin"* ]]; then
            # macOS
            sed -i '' "s/\(.*kubernetes.io\/hostname:\).*/\1 $NEW_NODE/" "$TENSORRT_YAML"
        else
            # Linux
            sed -i "s/\(.*kubernetes.io\/hostname:\).*/\1 $NEW_NODE/" "$TENSORRT_YAML"
        fi
        echo "   ✅ Updated nodeSelector in $TENSORRT_YAML to: $NEW_NODE"
    else
        echo "   ⚠️  nodeSelector not found in $TENSORRT_YAML"
        echo "   Please manually add:"
        echo "     nodeSelector:"
        echo "       kubernetes.io/hostname: $NEW_NODE"
    fi
else
    echo "   ℹ️  TensorRT YAML not found at $TENSORRT_YAML"
    echo "   Please manually update the nodeSelector:"
    echo "     nodeSelector:"
    echo "       kubernetes.io/hostname: $NEW_NODE"
fi

echo ""
echo "💡 Next steps:"
echo "   1. Verify node is ready: kubectl get node $NEW_NODE"
echo "   2. Deploy TensorRT-LLM: kubectl apply -f tensorrt/llama-3.2-1b.yaml -n multi-engines"
