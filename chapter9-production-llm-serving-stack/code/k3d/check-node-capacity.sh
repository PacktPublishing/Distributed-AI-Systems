#!/bin/bash
# Check if k3d cluster has enough nodes and GPU capacity for TensorRT-LLM

set -e

echo "=== k3d cluster node and GPU capacity analysis ==="
echo ""

# Get all nodes
echo "Node list:"
kubectl get nodes -o wide
echo ""

# Get GPU capacity per node
echo "GPU capacity per node:"
for node in $(kubectl get nodes -o jsonpath='{.items[*].metadata.name}'); do
    gpu_capacity=$(kubectl get node "$node" -o jsonpath='{.status.capacity.nvidia\.com/gpu}' 2>/dev/null || echo "0")
    gpu_allocatable=$(kubectl get node "$node" -o jsonpath='{.status.allocatable.nvidia\.com/gpu}' 2>/dev/null || echo "0")
    role=$(kubectl get node "$node" -o jsonpath='{.metadata.labels.node-role\.kubernetes\.io/.*}' 2>/dev/null | head -1 || echo "agent")
    
    if [ -z "$role" ] || [ "$role" = "agent" ]; then
        role="agent"
    else
        role="control-plane"
    fi
    
    echo "  $node ($role):"
    echo "    GPU Capacity: $gpu_capacity"
    echo "    GPU Allocatable: $gpu_allocatable"
done
echo ""

# Get current GPU usage
echo "📈 当前 GPU 使用情况："
kubectl get pods --all-namespaces -o json | jq -r '
.items[] | 
select(.spec.containers[].resources.requests."nvidia.com/gpu") |
"\(.metadata.namespace)/\(.metadata.name): \(.spec.containers[].resources.requests."nvidia.com/gpu") GPU (node: \(.spec.nodeName // "pending"))"
' 2>/dev/null || echo "  (需要 jq 工具来解析)"
echo ""

# Count GPU requests
echo "🔢 GPU 资源统计："
TOTAL_GPU_REQUESTS=$(kubectl get pods --all-namespaces -o json 2>/dev/null | \
    jq '[.items[] | .spec.containers[]? | select(.resources.requests."nvidia.com/gpu") | .resources.requests."nvidia.com/gpu" | tonumber] | add' 2>/dev/null || echo "0")

echo "  Total GPU requests: $TOTAL_GPU_REQUESTS"
echo ""

# Check agent nodes
echo "Agent node details:"
AGENT_NODES=$(kubectl get nodes -o json | jq -r '.items[] | select(.metadata.labels."node-role.kubernetes.io/control-plane" != "true") | .metadata.name' 2>/dev/null)
AGENT_COUNT=$(echo "$AGENT_NODES" | wc -l)
echo "  Agent 节点数量: $AGENT_COUNT"
echo "  Agent 节点列表:"
echo "$AGENT_NODES" | while read -r node; do
    if [ -n "$node" ]; then
        echo "    - $node"
    fi
done
echo ""

# Analyze capacity for TensorRT-LLM
echo "TensorRT-LLM deployment analysis:"
echo ""
echo "TensorRT-LLM resource requirements:"
echo "  - GPU: 1 (requests: 1, limits: 1)"
echo "  - Memory: 6Gi request, 8Gi limit"
echo ""

# Check if we can schedule TensorRT-LLM
echo "Deployment recommendations:"
echo ""
echo "Option 1: Deploy on existing nodes (if GPU memory utilization is low)"
echo "  - With --gpu-memory-utilization 0.2, one GPU can run multiple pods"
echo "  - Kubernetes GPU allocation is exclusive (requests: 1 = 1 full GPU)"
echo "  - Pod will be Pending if node GPUs are full"
echo ""
echo "Option 2: Add new agent node (recommended)"
echo "  - See README.md step 4 for adding agent nodes"
echo "  - Use nodeSelector to deploy TensorRT-LLM on new node"
echo ""
echo "Option 3: GPU sharing (requires MIG or GPU time-slicing)"
echo "  - Configure GPU sharing (e.g. NVIDIA MIG)"
echo "  - Or use GPU time-slice scheduling (extra config required)"
echo ""

# Check current deployments
echo "Current model deployments:"
kubectl get deployments --all-namespaces -o wide | grep -E "vllm|sglang|tensorrt" || echo "  (none found)"
echo ""

echo "Analysis complete."
echo ""
echo "To check GPU usage:"
echo "   kubectl describe nodes | grep -A 10 nvidia.com/gpu"
echo "   kubectl get pods --all-namespaces -o wide | grep -E 'vllm|sglang|tensorrt'"
