#!/bin/bash
# Script to manage k3d cluster with multiple inference engines (vLLM + SGLang)
# Uses namespace 'multi-engines' to isolate from multi-models deployment
# 
# Usage:
#   ./manage-cluster-multi-engines.sh [stop|start|restart|status]
#   NAMESPACE=custom-ns ./manage-cluster-multi-engines.sh start  # Use custom namespace
#
# Examples:
#   ./manage-cluster-multi-engines.sh stop      # Stop the cluster
#   ./manage-cluster-multi-engines.sh start     # Start the cluster (deploys vLLM + SGLang in 'multi-engines' namespace)
#   ./manage-cluster-multi-engines.sh restart   # Restart the cluster
#   ./manage-cluster-multi-engines.sh status    # Show cluster status

set -e

CLUSTER_NAME="mycluster-gpu"
NAMESPACE="${NAMESPACE:-multi-engines}"  # Namespace for multi-engine deployment (vLLM + SGLang)

# Function to show usage
show_usage() {
    echo "Usage: $0 [stop|start|restart|status]"
    echo ""
    echo "Commands:"
    echo "  stop     - Stop the k3d cluster"
    echo "  start    - Start the k3d cluster"
    echo "  restart  - Restart the k3d cluster"
    echo "  status   - Show cluster status"
    exit 1
}

# Function to stop namespace resources (does NOT stop the cluster)
stop_cluster() {
    echo "=========================================="
    echo "Stopping resources in namespace: $NAMESPACE"
    echo "=========================================="
    echo ""
    echo "⚠️  Note: This will only delete resources in '$NAMESPACE' namespace."
    echo "   The cluster itself will remain running (other namespaces are unaffected)."
    echo ""
    
    # Check if cluster is running
    if ! k3d cluster list | grep -q "$CLUSTER_NAME"; then
        echo "⚠️  Cluster $CLUSTER_NAME not found"
        exit 1
    fi
    
    # Check if cluster is actually running
    if ! kubectl cluster-info &>/dev/null; then
        echo "⚠️  Cluster is not accessible. Please start the cluster first:"
        echo "   k3d cluster start $CLUSTER_NAME"
        exit 1
    fi
    
    # Stop port-forward processes for this namespace
    echo "Checking for active port-forward processes..."
    PF_PIDS=$(pgrep -f "kubectl port-forward.*$NAMESPACE" 2>/dev/null || true)
    if [ -n "$PF_PIDS" ]; then
        echo "  Found port-forward processes for namespace '$NAMESPACE', stopping them..."
        for pid in $PF_PIDS; do
            # Get the command line to show what we're stopping
            PF_CMD=$(ps -p $pid -o args= 2>/dev/null | head -1 || echo "unknown")
            echo "    Stopping port-forward (PID: $pid): $PF_CMD"
            kill $pid 2>/dev/null || true
        done
        sleep 1
        # Force kill if still running
        for pid in $PF_PIDS; do
            if ps -p $pid > /dev/null 2>&1; then
                echo "    Force killing port-forward (PID: $pid)..."
                kill -9 $pid 2>/dev/null || true
            fi
        done
        echo "  ✅ Port-forward processes stopped"
    else
        echo "  No port-forward processes found for namespace '$NAMESPACE'"
    fi
    echo ""
    
    # Clean up namespace resources (does NOT stop the cluster)
    echo "🧹 Cleaning up resources in namespace '$NAMESPACE'..."
    if kubectl get namespace "$NAMESPACE" &>/dev/null; then
        kubectl delete deployments,svc -n "$NAMESPACE" --all 2>/dev/null || echo "    ⚠️  Failed to delete some resources (may not exist)"
        echo "  ✅ Resources in namespace '$NAMESPACE' deleted"
    else
        echo "  ℹ️  Namespace '$NAMESPACE' does not exist, nothing to clean up"
    fi
    
    echo ""
    echo "✅ Resources in namespace '$NAMESPACE' stopped"
    echo ""
    echo "📊 Remaining resources in namespace '$NAMESPACE':"
    kubectl get all -n "$NAMESPACE" 2>/dev/null || echo "  (namespace is empty or does not exist)"
    echo ""
    echo "💡 The cluster is still running. Other namespaces are unaffected."
    echo "   To stop the entire cluster, use: k3d cluster stop $CLUSTER_NAME"
    echo ""
    echo "📊 Docker containers status:"
    docker ps -a --filter "name=k3d-$CLUSTER_NAME" --filter "label=k3d.cluster=$CLUSTER_NAME" --format "table {{.Names}}\t{{.Status}}"
}

# Function to start cluster
start_cluster() {
    echo "=========================================="
    echo "Starting k3d Cluster: $CLUSTER_NAME"
    echo "=========================================="
    echo ""
    
    if ! k3d cluster list | grep -q "$CLUSTER_NAME"; then
        echo "❌ Error: Cluster $CLUSTER_NAME not found"
        echo "   Please create the cluster first (see README)"
        exit 1
    fi
    
    echo "Starting cluster..."
    k3d cluster start "$CLUSTER_NAME"
    
    # Also start manually created agent nodes if they exist
    echo ""
    echo "Starting manually created agent nodes..."
    MANUAL_AGENTS=$(docker ps -a --filter "label=k3d.cluster=$CLUSTER_NAME" --filter "label=k3d.role=agent" --format "{{.Names}}" | grep -v "^k3d-$CLUSTER_NAME" || true)
    if [ -n "$MANUAL_AGENTS" ]; then
        for agent in $MANUAL_AGENTS; do
            if ! docker ps --format "{{.Names}}" | grep -q "^${agent}$"; then
                echo "  Starting $agent..."
                docker start "$agent" 2>/dev/null || true
            fi
        done
    fi
    
    echo ""
    echo "⏳ Waiting for cluster to be ready..."
    sleep 5
    
    # Wait for NVIDIA device plugin to be ready (important for GPU workloads)
    echo ""
    echo "⏳ Waiting for NVIDIA device plugin to register GPUs..."
    echo "   (This may take 20-30 seconds after cluster start)"
    for i in {1..30}; do
        if kubectl get nodes -o json | python3 -c "import sys, json; d=json.load(sys.stdin); nodes=[n for n in d['items'] if n['status'].get('allocatable', {}).get('nvidia.com/gpu')]; exit(0 if nodes else 1)" 2>/dev/null; then
            echo "   ✅ GPUs detected on nodes"
            break
        fi
        if [ $i -eq 30 ]; then
            echo "   ⚠️  GPUs not yet detected (device plugin may need more time)"
        else
            echo -n "."
            sleep 1
        fi
    done
    echo ""
    
    # Merge kubeconfig
    echo "🔗 Merging kubeconfig..."
    k3d kubeconfig merge "$CLUSTER_NAME" --kubeconfig-merge-default
    
    # Fix kubeconfig server address if needed
    export KUBECONFIG=$HOME/.kube/config
    KUBE_SERVER=$(kubectl config view -o jsonpath='{.clusters[?(@.name=="k3d-'$CLUSTER_NAME'")].cluster.server}' 2>/dev/null || echo "")
    if [[ "$KUBE_SERVER" == *"0.0.0.0"* ]]; then
        echo "🔧 Fixing kubeconfig server address..."
        kubectl config set-cluster "k3d-$CLUSTER_NAME" --server=$(echo $KUBE_SERVER | sed 's/0.0.0.0/127.0.0.1/')
    fi
    
    # Create namespace for multi-engine deployment
    echo ""
    echo "📦 Creating namespace: $NAMESPACE"
    kubectl create namespace "$NAMESPACE" 2>/dev/null || echo "  ✅ Namespace already exists"
    
    # Create HF_TOKEN secret in namespace if it doesn't exist
    if [ -n "$HF_TOKEN" ]; then
        echo "  🔐 Creating/updating HF_TOKEN secret in namespace..."
        kubectl create secret generic hf-token-secret \
          --from-literal=token="$HF_TOKEN" \
          --namespace "$NAMESPACE" \
          --dry-run=client -o yaml | kubectl apply -f - 2>/dev/null || echo "    ⚠️  Failed to create secret"
    else
        echo "  ⚠️  HF_TOKEN not set, secret may need to be created manually"
    fi
    
    # Auto-deploy LLM serving pods (multiple engines: vLLM + SGLang)
    # This script only deploys two services:
    #   1. vLLM Service (Llama-3.2-1B)
    #   2. SGLang Service (Llama-3.2-1B)
    echo ""
    echo "🚀 Checking and deploying LLM serving pods (multi-engine: vLLM + SGLang) in namespace: $NAMESPACE..."
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    
    # Clean up Pods in error states (UnexpectedAdmissionError, Failed, etc.)
    # This prevents multiple Pod instances when old error Pods aren't cleaned up
    echo "  🧹 Cleaning up Pods in error states..."
    # Use Python to properly detect pods with UnexpectedAdmissionError in conditions
    ERROR_PODS=$(kubectl get pods -n "$NAMESPACE" -o json 2>/dev/null | python3 -c "
import sys, json
data = json.load(sys.stdin)
pods_to_delete = []
for pod in data.get('items', []):
    name = pod['metadata']['name']
    phase = pod.get('status', {}).get('phase', '')
    conditions = pod.get('status', {}).get('conditions', [])
    
    # Check for Failed phase
    if phase == 'Failed':
        pods_to_delete.append(name)
        continue
    
    # Check for UnexpectedAdmissionError in PodScheduled condition
    for condition in conditions:
        if condition.get('type') == 'PodScheduled':
            reason = condition.get('reason', '')
            if reason == 'UnexpectedAdmissionError':
                pods_to_delete.append(name)
                break

for pod_name in pods_to_delete:
    print(pod_name)
" || true)
    
    if [ -n "$ERROR_PODS" ]; then
        echo "$ERROR_PODS" | while read -r pod; do
            if [ -n "$pod" ]; then
                echo "    🗑️  Deleting error Pod: $pod"
                kubectl delete pod "$pod" -n "$NAMESPACE" --grace-period=0 --force 2>/dev/null || true
            fi
        done
        sleep 2  # Wait for Pods to be deleted
    else
        echo "    ✅ No error Pods found"
    fi
    
    # Deploy vLLM Llama-3.2-1B deployment if it doesn't exist
    # Note: vLLM now uses Deployment (vllm-llama-32-1b-pod), not direct Pod
    if ! kubectl get deployment vllm-llama-32-1b-pod -n "$NAMESPACE" &>/dev/null; then
        echo "  📦 Deploying vLLM Llama-3.2-1B deployment..."
        if [ -f "$SCRIPT_DIR/vllm/llama-3.2-1b.yaml" ]; then
            kubectl apply -f "$SCRIPT_DIR/vllm/llama-3.2-1b.yaml" -n "$NAMESPACE" 2>/dev/null || echo "    ⚠️  Failed to deploy vLLM deployment (may need HF_TOKEN secret)"
        else
            echo "    ⚠️  vLLM YAML file not found: $SCRIPT_DIR/vllm/llama-3.2-1b.yaml"
        fi
    else
        echo "  ✅ vLLM Llama-3.2-1B deployment already exists"
    fi
    
    # Deploy SGLang Llama-3.2-1B deployment if it doesn't exist
    # Note: SGLang now uses Deployment (sglang-llama-32-1b-pod), not direct Pod
    if ! kubectl get deployment sglang-llama-32-1b-pod -n "$NAMESPACE" &>/dev/null; then
        echo "  📦 Deploying SGLang Llama-3.2-1B deployment..."
        if [ -f "$SCRIPT_DIR/sglang/llama-3.2-1b.yaml" ]; then
            kubectl apply -f "$SCRIPT_DIR/sglang/llama-3.2-1b.yaml" -n "$NAMESPACE" 2>/dev/null || echo "    ⚠️  Failed to deploy SGLang deployment (may need HF_TOKEN secret)"
        else
            echo "    ⚠️  SGLang YAML file not found: $SCRIPT_DIR/sglang/llama-3.2-1b.yaml"
        fi
    else
        echo "  ✅ SGLang Llama-3.2-1B deployment already exists"
    fi
    
    # Deploy TensorRT-LLM Llama-3.2-1B deployment if it doesn't exist (optional)
    # Note: TensorRT-LLM uses Deployment (tensorrt-llama-32-1b-pod)
    if ! kubectl get deployment tensorrt-llama-32-1b-pod -n "$NAMESPACE" &>/dev/null; then
        echo "  📦 Deploying TensorRT-LLM Llama-3.2-1B deployment (optional)..."
        if [ -f "$SCRIPT_DIR/tensorrt/llama-3.2-1b.yaml" ]; then
            kubectl apply -f "$SCRIPT_DIR/tensorrt/llama-3.2-1b.yaml" -n "$NAMESPACE" 2>/dev/null || echo "    ⚠️  Failed to deploy TensorRT deployment (may need HF_TOKEN secret or TensorRT-LLM model)"
        else
            echo "    ℹ️  TensorRT YAML file not found: $SCRIPT_DIR/tensorrt/llama-3.2-1b.yaml (optional, skipping)"
        fi
    else
        echo "  ✅ TensorRT-LLM Llama-3.2-1B deployment already exists"
    fi
    
    echo ""
    echo "✅ Cluster started"
    echo ""
    echo "📊 Cluster status:"
    k3d cluster list
    echo ""
    echo "📊 Node status:"
    kubectl get nodes
    echo ""
    echo "📊 Pod status in namespace $NAMESPACE:"
    kubectl get pods -n "$NAMESPACE"
    echo ""
    echo "📊 Services in namespace $NAMESPACE:"
    kubectl get svc -n "$NAMESPACE"
    
    # Deploy API Gateway and Ingress (optional, in default namespace)
    echo ""
    echo "🌐 Checking API Gateway and Ingress..."
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    
    # Deploy API Gateway if it doesn't exist
    if ! kubectl get pod vllm-api-gateway -n default &>/dev/null; then
        echo "  📦 Deploying API Gateway..."
        if [ -f "$SCRIPT_DIR/gateway/deploy-gateway.sh" ]; then
            cd "$SCRIPT_DIR/gateway" && ./deploy-gateway.sh 2>/dev/null || echo "    ⚠️  Failed to deploy Gateway"
            cd - > /dev/null
        else
            echo "    ⚠️  Gateway deployment script not found"
        fi
    else
        echo "  ✅ API Gateway already exists"
    fi
    
    # Deploy Traefik Ingress if it doesn't exist (optional, for production access)
    if ! kubectl get ingress vllm-api-gateway-ingress -n default &>/dev/null; then
        echo "  📦 Deploying Traefik Ingress (optional, for HTTPS access)..."
        # Create TLS secret if it doesn't exist
        if ! kubectl get secret vllm-api-tls -n default &>/dev/null; then
            echo "    🔐 Creating TLS certificate..."
            openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
              -keyout /tmp/tls.key \
              -out /tmp/tls.crt \
              -subj "/CN=localhost" \
              -addext "subjectAltName=DNS:localhost,DNS:*.localhost,IP:127.0.0.1,IP:::1" 2>/dev/null || echo "      ⚠️  OpenSSL not available, skipping TLS secret"
            
            if [ -f /tmp/tls.crt ] && [ -f /tmp/tls.key ]; then
                kubectl create secret tls vllm-api-tls \
                  --cert=/tmp/tls.crt \
                  --key=/tmp/tls.key \
                  -n default 2>/dev/null || echo "      ⚠️  Failed to create TLS secret"
                rm -f /tmp/tls.crt /tmp/tls.key
            fi
        fi
        
        # Apply Ingress configuration
        if [ -f "$SCRIPT_DIR/gateway/ingress-tls-traefik.yaml" ]; then
            kubectl apply -f "$SCRIPT_DIR/gateway/ingress-tls-traefik.yaml" 2>/dev/null || echo "    ⚠️  Failed to deploy Ingress"
            echo "    ✅ Ingress deployed (access via https://localhost)"
        else
            echo "    ⚠️  Ingress YAML not found: $SCRIPT_DIR/gateway/ingress-tls-traefik.yaml"
        fi
    else
        echo "  ✅ Traefik Ingress already exists"
    fi
}

# Function to restart cluster
restart_cluster() {
    echo "=========================================="
    echo "Restarting k3d Cluster: $CLUSTER_NAME"
    echo "=========================================="
    echo ""
    stop_cluster
    echo ""
    sleep 2
    start_cluster
}

# Function to show status
show_status() {
    echo "=========================================="
    echo "k3d Cluster Status: $CLUSTER_NAME"
    echo "=========================================="
    echo ""
    
    echo "📊 Cluster list:"
    k3d cluster list
    echo ""
    
    # Check if cluster is running by checking if servers/agents are running (not 0/1)
    CLUSTER_STATUS=$(k3d cluster list --no-headers | grep "^$CLUSTER_NAME" | awk '{print $2, $3}' || echo "")
    if echo "$CLUSTER_STATUS" | grep -qE "[1-9]/[0-9]+.*[1-9]/[0-9]+"; then
        # Switch to cluster context
        kubectl config use-context "k3d-$CLUSTER_NAME" 2>/dev/null || true
        
        echo "📊 Kubernetes nodes:"
        kubectl get nodes 2>/dev/null || echo "  Unable to connect to cluster"
        echo ""
        
        echo "📊 Namespaces:"
        kubectl get namespaces 2>/dev/null | grep -E "multi-engines|multi-models|NAME" || echo "  Unable to list namespaces"
        echo ""
        
        echo "📊 Pods in namespace $NAMESPACE:"
        kubectl get pods -n "$NAMESPACE" 2>/dev/null || echo "  Unable to list pods in namespace $NAMESPACE"
        echo ""
        
        echo "📊 Services in namespace $NAMESPACE:"
        kubectl get svc -n "$NAMESPACE" 2>/dev/null | head -10 || echo "  Unable to list services in namespace $NAMESPACE"
    else
        echo "⚠️  Cluster is not running (servers/agents: $CLUSTER_STATUS)"
    fi
}

# Main script logic
case "${1:-}" in
    stop)
        stop_cluster
        ;;
    start)
        start_cluster
        ;;
    restart)
        restart_cluster
        ;;
    status)
        show_status
        ;;
    *)
        show_usage
        ;;
esac
