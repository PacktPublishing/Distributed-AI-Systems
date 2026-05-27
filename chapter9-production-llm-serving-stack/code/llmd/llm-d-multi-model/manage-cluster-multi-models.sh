#!/bin/bash
# Script to manage llm-d cluster with multiple vLLM models
# Deploys Llama-3.2-1B and Qwen2.5-0.5B on vLLM
#
# This script is idempotent - safe to run multiple times.
#
# Usage:
#   ./manage-cluster-multi-models.sh [start|stop|restart|status] [--with-gateway]
#
# Examples:
#   ./manage-cluster-multi-models.sh start               # Deploy models (direct service access)
#   ./manage-cluster-multi-models.sh start --with-gateway # Deploy with Inference Gateway
#   ./manage-cluster-multi-models.sh stop                # Delete the cluster
#   ./manage-cluster-multi-models.sh restart             # Delete and recreate cluster
#   ./manage-cluster-multi-models.sh status              # Show cluster and pod status
#
# Prerequisites:
#   - Build the k3s-cuda image first: cd ../../../k3d && ./build.sh
#   - The default k3s image lacks NVIDIA container toolkit support
#
# IMPORTANT: Match the k3s-cuda image to your CUDA version!
#   - Check your CUDA version: nvidia-smi (look for "CUDA Version: X.Y")
#   - Build matching image: cd ../../../k3d && ./build.sh --cuda-version 13.0
#   - Or set: export K3S_IMAGE="k3s-cuda:1.35.1-cuda-13.0"
#
# Image versions are pinned for reproducibility. To use different versions:
#   export K3S_IMAGE="k3s-cuda:1.35.1-cuda-12.4.1"
#   export VLLM_IMAGE="vllm/vllm-openai:v0.14.1"
#
# Check available vLLM versions at: https://hub.docker.com/r/vllm/vllm-openai/tags
# llm-d v0.5.0 bundles vLLM v0.14.1

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLUSTER_NAME="llmd-cluster"
VLLM_IMAGE="${VLLM_IMAGE:-vllm/vllm-openai:v0.14.1}"
K3S_IMAGE="${K3S_IMAGE:-k3s-cuda:1.35.1-cuda-12.4.1}"
WITH_GATEWAY=false

# Function to show usage
show_usage() {
    echo "Usage: $0 [start|stop|restart|status] [--with-gateway]"
    echo ""
    echo "Commands:"
    echo "  start    - Create cluster and deploy models"
    echo "  stop     - Delete the cluster"
    echo "  restart  - Delete and recreate cluster"
    echo "  status   - Show cluster and pod status"
    echo ""
    echo "Options:"
    echo "  --with-gateway  Deploy with Inference Gateway for unified routing"
    echo ""
    echo "Environment variables:"
    echo "  HF_TOKEN    - HuggingFace token (required for start)"
    echo "  K3S_IMAGE   - Custom k3s-cuda image (default: $K3S_IMAGE)"
    echo "  VLLM_IMAGE  - vLLM image (default: $VLLM_IMAGE)"
    exit 1
}

# Helper function: wait for pod with fast-fail on errors
wait_for_pod() {
    local pod_name="$1"
    local timeout="${2:-600}"
    local interval=5
    local elapsed=0

    echo "Waiting for $pod_name to be ready (timeout: ${timeout}s)..."

    while [ $elapsed -lt $timeout ]; do
        # Get pod status
        local status=$(kubectl get pod "$pod_name" -o jsonpath='{.status.phase}' 2>/dev/null)
        local container_status=$(kubectl get pod "$pod_name" -o jsonpath='{.status.containerStatuses[0].state}' 2>/dev/null)

        # Check for failure states
        if echo "$container_status" | grep -q "CrashLoopBackOff\|Error\|OOMKilled\|ImagePullBackOff\|ErrImagePull\|InvalidImageName"; then
            echo ""
            echo "ERROR: Pod $pod_name failed to start!"
            echo "Status: $status"
            echo "Container state: $container_status"
            echo ""
            echo "Pod events:"
            kubectl describe pod "$pod_name" | grep -A 20 "Events:" || true
            echo ""
            echo "Recent logs:"
            kubectl logs "$pod_name" --tail=50 2>/dev/null || true
            return 1
        fi

        # Check if ready
        if kubectl get pod "$pod_name" -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null | grep -q "True"; then
            echo ""
            echo "Pod $pod_name is ready!"
            return 0
        fi

        # Show progress
        local reason=$(kubectl get pod "$pod_name" -o jsonpath='{.status.containerStatuses[0].state.waiting.reason}' 2>/dev/null)
        if [ -n "$reason" ]; then
            printf "\r  Status: %s (%ds elapsed)..." "$reason" "$elapsed"
        else
            printf "\r  Status: %s (%ds elapsed)..." "$status" "$elapsed"
        fi

        sleep $interval
        elapsed=$((elapsed + interval))
    done

    echo ""
    echo "ERROR: Timeout waiting for pod $pod_name"
    echo "Current status:"
    kubectl get pod "$pod_name" -o wide
    kubectl describe pod "$pod_name" | grep -A 10 "Events:" || true
    return 1
}

# Function to show status
show_status() {
    echo "=========================================="
    echo "  llm-d Multi-Model Status"
    echo "=========================================="
    echo ""

    echo "Cluster: $CLUSTER_NAME"
    echo ""

    # Check if cluster exists
    if ! k3d cluster list 2>/dev/null | grep -q "^${CLUSTER_NAME} "; then
        echo "Status: NOT FOUND"
        echo ""
        echo "To create the cluster, run:"
        echo "  $0 start"
        return
    fi

    # Get cluster status
    echo "Cluster list:"
    k3d cluster list
    echo ""

    # Check if we can connect to the cluster
    if ! kubectl cluster-info &>/dev/null 2>&1; then
        echo "Status: CLUSTER EXISTS BUT NOT ACCESSIBLE"
        echo ""
        echo "Docker containers:"
        docker ps -a --filter "name=k3d-$CLUSTER_NAME" --format "table {{.Names}}\t{{.Status}}"
        return
    fi

    # Switch context
    kubectl config use-context "k3d-$CLUSTER_NAME" &>/dev/null || true

    echo "Nodes:"
    kubectl get nodes -o wide 2>/dev/null || echo "  Unable to get nodes"
    echo ""

    echo "GPU Resources:"
    kubectl get nodes -o json 2>/dev/null | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    for node in data.get('items', []):
        name = node['metadata']['name']
        allocatable = node.get('status', {}).get('allocatable', {})
        gpus = allocatable.get('nvidia.com/gpu', '0')
        print(f'  {name}: {gpus} GPU(s)')
except:
    print('  Unable to parse GPU info')
" || echo "  Unable to get GPU info"
    echo ""

    echo "Pods:"
    kubectl get pods -o wide 2>/dev/null || echo "  Unable to get pods"
    echo ""

    echo "Services:"
    kubectl get svc 2>/dev/null || echo "  Unable to get services"
    echo ""

    # Check pod health
    echo "Pod Health:"
    for pod in vllm-llama-32-1b vllm-qwen2-5-0-5b; do
        if kubectl get pod "$pod" &>/dev/null 2>&1; then
            status=$(kubectl get pod "$pod" -o jsonpath='{.status.phase}' 2>/dev/null)
            ready=$(kubectl get pod "$pod" -o jsonpath='{.status.conditions[?(@.type=="Ready")].status}' 2>/dev/null)
            if [ "$ready" = "True" ]; then
                echo "  $pod: READY"
            else
                echo "  $pod: $status (not ready)"
            fi
        else
            echo "  $pod: NOT DEPLOYED"
        fi
    done
    echo ""

    # Check gateway status
    echo "Inference Gateway:"
    if kubectl get deployment llm-gateway &>/dev/null 2>&1; then
        ready=$(kubectl get deployment llm-gateway -o jsonpath='{.status.readyReplicas}' 2>/dev/null)
        if [ "$ready" = "1" ]; then
            echo "  llm-gateway: READY (unified routing enabled)"
            echo "  Access: kubectl port-forward svc/llm-gateway 8000:8000"
        else
            echo "  llm-gateway: NOT READY"
        fi
    else
        echo "  NOT DEPLOYED (using direct service access)"
        echo "  To enable: $0 start --with-gateway"
    fi
}

# Function to deploy Inference Gateway
deploy_gateway() {
    echo "Step 6: Deploying Inference Gateway"

    # Install Gateway API CRDs
    echo "Installing Gateway API CRDs..."
    if ! kubectl get crd gateways.gateway.networking.k8s.io &>/dev/null; then
        kubectl apply -f https://github.com/kubernetes-sigs/gateway-api/releases/download/v1.2.0/standard-install.yaml
        echo "Gateway API CRDs installed."
    else
        echo "Gateway API CRDs already installed."
    fi

    # Install Gateway API Inference Extension CRDs
    echo "Installing Gateway API Inference Extension CRDs..."
    if ! kubectl get crd inferencepools.inference.networking.x-k8s.io &>/dev/null; then
        kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/download/v0.3.0/manifests.yaml
        echo "Inference Extension CRDs installed."
    else
        echo "Inference Extension CRDs already installed."
    fi

    # Note: InferencePool not created - requires extensionRef (EPP) which we don't deploy.
    # llm-gateway routes directly to vLLM Services (vllm-llama-32-1b, vllm-qwen2-5-0-5b).

    # Deploy the Inference Gateway (Envoy-based)
    echo "Deploying Inference Gateway..."
    if ! kubectl get deployment llm-gateway &>/dev/null; then
        cat <<EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: llm-gateway
spec:
  replicas: 1
  selector:
    matchLabels:
      app: llm-gateway
  template:
    metadata:
      labels:
        app: llm-gateway
    spec:
      containers:
      - name: gateway
        image: envoyproxy/envoy:v1.31-latest
        ports:
        - containerPort: 8000
        command:
        - /bin/sh
        - -c
        - |
          cat > /tmp/envoy.yaml << 'ENVOY_CONFIG'
          static_resources:
            listeners:
            - name: listener_0
              address:
                socket_address:
                  address: 0.0.0.0
                  port_value: 8000
              filter_chains:
              - filters:
                - name: envoy.filters.network.http_connection_manager
                  typed_config:
                    "@type": type.googleapis.com/envoy.extensions.filters.network.http_connection_manager.v3.HttpConnectionManager
                    stat_prefix: ingress_http
                    codec_type: AUTO
                    route_config:
                      name: local_route
                      virtual_hosts:
                      - name: backend
                        domains: ["*"]
                        routes:
                        - match:
                            prefix: "/"
                            headers:
                            - name: x-model-name
                              string_match:
                                prefix: "meta-llama"
                          route:
                            cluster: llama_cluster
                        - match:
                            prefix: "/"
                            headers:
                            - name: x-model-name
                              string_match:
                                prefix: "Qwen"
                          route:
                            cluster: qwen_cluster
                        - match:
                            prefix: "/"
                          route:
                            cluster: llama_cluster
                    http_filters:
                    - name: envoy.filters.http.lua
                      typed_config:
                        "@type": type.googleapis.com/envoy.extensions.filters.http.lua.v3.Lua
                        inline_code: |
                          function envoy_on_request(request_handle)
                            local body = request_handle:body()
                            if body then
                              local body_str = body:getBytes(0, body:length())
                              local model = string.match(body_str, '"model"%s*:%s*"([^"]+)"')
                              if model then
                                request_handle:headers():add("x-model-name", model)
                              end
                            end
                          end
                    - name: envoy.filters.http.router
                      typed_config:
                        "@type": type.googleapis.com/envoy.extensions.filters.http.router.v3.Router
            clusters:
            - name: llama_cluster
              connect_timeout: 30s
              type: STRICT_DNS
              lb_policy: ROUND_ROBIN
              load_assignment:
                cluster_name: llama_cluster
                endpoints:
                - lb_endpoints:
                  - endpoint:
                      address:
                        socket_address:
                          address: vllm-llama-32-1b
                          port_value: 8000
            - name: qwen_cluster
              connect_timeout: 30s
              type: STRICT_DNS
              lb_policy: ROUND_ROBIN
              load_assignment:
                cluster_name: qwen_cluster
                endpoints:
                - lb_endpoints:
                  - endpoint:
                      address:
                        socket_address:
                          address: vllm-qwen2-5-0-5b
                          port_value: 8000
          ENVOY_CONFIG
          envoy -c /tmp/envoy.yaml
        resources:
          requests:
            memory: "128Mi"
            cpu: "100m"
          limits:
            memory: "256Mi"
            cpu: "500m"
---
apiVersion: v1
kind: Service
metadata:
  name: llm-gateway
spec:
  selector:
    app: llm-gateway
  ports:
  - port: 8000
    targetPort: 8000
EOF
        echo "Inference Gateway deployed."
    else
        echo "Inference Gateway already exists."
    fi

    # Wait for gateway to be ready
    echo "Waiting for gateway to be ready..."
    kubectl wait --for=condition=available deployment/llm-gateway --timeout=120s || \
        echo "Gateway may still be starting. Check: kubectl logs -l app=llm-gateway"
    echo ""
}

# Function to stop/delete cluster
stop_cluster() {
    echo "=========================================="
    echo "  Stopping llm-d Cluster"
    echo "=========================================="
    echo ""

    if ! k3d cluster list 2>/dev/null | grep -q "^${CLUSTER_NAME} "; then
        echo "Cluster '$CLUSTER_NAME' not found."
        return
    fi

    echo "Deleting cluster: $CLUSTER_NAME"
    k3d cluster delete "$CLUSTER_NAME"
    echo ""
    echo "Cluster deleted."
}

# Function to start/create cluster and deploy models
start_cluster() {
    echo "=========================================="
    echo "  llm-d Multi-Model Deployment"
    echo "=========================================="
    echo "Using k3s image: $K3S_IMAGE"
    echo "Using vLLM image: $VLLM_IMAGE"
    echo ""

    # Check prerequisites
    echo "Checking prerequisites..."

    if ! command -v k3d &> /dev/null; then
        echo "Error: k3d not found. Please install k3d first."
        exit 1
    fi

    if ! command -v kubectl &> /dev/null; then
        echo "Error: kubectl not found. Please install kubectl first."
        exit 1
    fi

    if [ -z "$HF_TOKEN" ]; then
        echo "Error: HF_TOKEN environment variable not set"
        echo "Please run: export HF_TOKEN='your_token_here'"
        exit 1
    fi

    echo "All prerequisites met."
    echo ""

    # Step 1: Create or use existing k3d cluster
    echo "Step 1: Setting up k3d cluster"

    # Check for existing k3d clusters
    existing_clusters=$(k3d cluster list --no-headers 2>/dev/null | awk '{print $1}')
    if [ -n "$existing_clusters" ]; then
        echo "Found existing k3d cluster(s):"
        k3d cluster list
        echo ""
        
        if echo "$existing_clusters" | grep -q "^${CLUSTER_NAME}$"; then
            echo "Target cluster '$CLUSTER_NAME' already exists."
            read -p "Delete and recreate it? (yes/no) [no]: " response
            if [ "$response" = "yes" ]; then
                echo "Deleting cluster: $CLUSTER_NAME"
                k3d cluster delete "$CLUSTER_NAME"
            fi
        else
            echo "WARNING: Other k3d cluster(s) are running."
            echo "This may cause resource conflicts (GPU, ports, memory)."
            read -p "Delete all existing clusters? (yes/no) [no]: " response
            if [ "$response" = "yes" ]; then
                for cluster in $existing_clusters; do
                    echo "Deleting cluster: $cluster"
                    k3d cluster delete "$cluster"
                done
            else
                echo "Proceeding with existing clusters running..."
            fi
        fi
    fi

    # Create new cluster if needed
    if ! k3d cluster list 2>/dev/null | grep -q "^${CLUSTER_NAME} "; then
        # Check if k3s-cuda image exists
        if ! docker images --format "{{.Repository}}:{{.Tag}}" | grep -q "^${K3S_IMAGE}$"; then
            echo "Error: k3s-cuda image not found: $K3S_IMAGE"
            echo ""
            echo "The default k3s image lacks NVIDIA container toolkit support."
            echo "You need to build the k3s-cuda image matching your CUDA version."
            echo ""
            echo "1. Check your CUDA version:"
            echo "   nvidia-smi  # Look for 'CUDA Version: X.Y'"
            echo ""
            echo "2. Build the k3s-cuda image:"
            echo "   cd ../../../k3d && ./build.sh"
            echo "   # Or with specific CUDA version:"
            echo "   cd ../../../k3d && ./build.sh --cuda-version 13.0"
            echo ""
            echo "3. Or use an existing image:"
            echo "   export K3S_IMAGE='k3s-cuda:1.35.1-cuda-13.0'"
            echo ""
            echo "Available k3s-cuda images:"
            docker images --format "  {{.Repository}}:{{.Tag}}" | grep "k3s-cuda" || echo "  (none found)"
            exit 1
        fi

        # Check if port 8080 is available before creating cluster
        if lsof -i :8080 &>/dev/null; then
            echo "Port 8080 is in use. Attempting to free it..."
            # Kill processes using port 8080
            fuser -k 8080/tcp 2>/dev/null || sudo fuser -k 8080/tcp 2>/dev/null || true
            sleep 1
            if lsof -i :8080 &>/dev/null; then
                echo "Error: Could not free port 8080. Please manually stop the process using it."
                echo "Run: lsof -i :8080"
                exit 1
            fi
            echo "Port 8080 freed."
        fi
        echo "Creating k3d cluster: $CLUSTER_NAME"
        k3d cluster create "$CLUSTER_NAME" \
          --image "$K3S_IMAGE" \
          --gpus=all \
          --servers 1 \
          --agents 1 \
          --port "8080:80@loadbalancer"
    else
        echo "Using existing cluster: $CLUSTER_NAME"
    fi

    k3d kubeconfig merge "$CLUSTER_NAME" --kubeconfig-merge-default
    kubectl config use-context "k3d-$CLUSTER_NAME"
    kubectl wait --for=condition=Ready nodes --all --timeout=60s
    echo ""

    # Step 2: Install NVIDIA device plugin
    echo "Step 2: Installing NVIDIA device plugin"

    if kubectl get daemonset nvidia-device-plugin-daemonset -n kube-system &>/dev/null; then
        echo "NVIDIA device plugin already installed."
    else
        kubectl apply -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.14.1/nvidia-device-plugin.yml
        kubectl wait --for=condition=ready pod \
          -l name=nvidia-device-plugin-ds \
          -n kube-system \
          --timeout=60s
        echo "NVIDIA device plugin installed."
    fi
    echo ""

    # Step 3: Create HF token secret (only if not exists)
    echo "Step 3: Creating HuggingFace token secret"

    if kubectl get secret hf-token-secret &>/dev/null; then
        echo "HF token secret already exists."
    else
        kubectl create secret generic hf-token-secret --from-literal=token="$HF_TOKEN"
        echo "HF token secret created."
    fi
    echo ""

    # Step 4: Deploy Llama-3.2-1B on vLLM
    echo "Step 4: Deploying Llama-3.2-1B on vLLM"

    if kubectl get pod vllm-llama-32-1b &>/dev/null; then
        echo "Llama pod already exists."
    else
        kubectl apply -f "$SCRIPT_DIR/vllm-pod.yaml"
        echo "Llama pod created."
    fi

    wait_for_pod vllm-llama-32-1b 600 || exit 1
    echo ""

    # Step 5: Deploy Qwen2.5-0.5B on vLLM
    echo "Step 5: Deploying Qwen2.5-0.5B on vLLM"

    if kubectl get pod vllm-qwen2-5-0-5b &>/dev/null; then
        echo "Qwen pod already exists."
    else
        cat <<EOF | kubectl apply -f -
apiVersion: v1
kind: Pod
metadata:
  name: vllm-qwen2-5-0-5b
  labels:
    app: vllm
    model: qwen2-5-0-5b
spec:
  runtimeClassName: nvidia
  containers:
  - name: vllm-server
    image: ${VLLM_IMAGE}
    args:
    - --model
    - Qwen/Qwen2.5-0.5B-Instruct
    - --host
    - "0.0.0.0"
    - --port
    - "8000"
    - --gpu-memory-utilization
    - "0.3"
    ports:
    - containerPort: 8000
    resources:
      limits:
        nvidia.com/gpu: 1
    livenessProbe:
      httpGet:
        path: /health
        port: 8000
      initialDelaySeconds: 120
      periodSeconds: 30
    readinessProbe:
      httpGet:
        path: /health
        port: 8000
      initialDelaySeconds: 60
      periodSeconds: 10
  restartPolicy: Always
---
apiVersion: v1
kind: Service
metadata:
  name: vllm-qwen2-5-0-5b
spec:
  selector:
    app: vllm
    model: qwen2-5-0-5b
  ports:
  - port: 8000
    targetPort: 8000
EOF
        echo "Qwen pod created."
    fi

    wait_for_pod vllm-qwen2-5-0-5b 600 || exit 1
    echo ""

    # Step 6: Deploy Inference Gateway (if requested)
    if [ "$WITH_GATEWAY" = true ]; then
        deploy_gateway
    fi

    # Summary
    echo "=========================================="
    echo "  Deployment Summary"
    echo "=========================================="
    echo ""
    kubectl get pods
    echo ""
    kubectl get svc
    echo ""
    echo "Deployment complete!"
    echo ""
    if [ "$WITH_GATEWAY" = true ]; then
        echo "Next steps (with Inference Gateway):"
        echo "  1. Port forward gateway:  kubectl port-forward svc/llm-gateway 8000:8000"
        echo "  2. Test Llama: curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{\"model\": \"meta-llama/Llama-3.2-1B-Instruct\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello!\"}]}'"
        echo "  3. Test Qwen:  curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{\"model\": \"Qwen/Qwen2.5-0.5B-Instruct\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello!\"}]}'"
    else
        echo "Next steps (direct service access):"
        echo "  1. Port forward Llama:  kubectl port-forward svc/vllm-llama-32-1b 8001:8000"
        echo "  2. Port forward Qwen:   kubectl port-forward svc/vllm-qwen2-5-0-5b 8002:8000"
        echo "  3. Test: curl http://localhost:8001/v1/chat/completions -H 'Content-Type: application/json' -d '{\"model\": \"meta-llama/Llama-3.2-1B-Instruct\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello!\"}]}'"
        echo ""
        echo "For unified routing, restart with: $0 start --with-gateway"
    fi
    echo ""
}

# Function to restart cluster
restart_cluster() {
    echo "=========================================="
    echo "  Restarting llm-d Cluster"
    echo "=========================================="
    echo ""
    stop_cluster
    echo ""
    start_cluster
}

# Parse arguments
COMMAND=""
for arg in "$@"; do
    case "$arg" in
        --with-gateway)
            WITH_GATEWAY=true
            ;;
        start|stop|restart|status)
            COMMAND="$arg"
            ;;
        *)
            echo "Unknown argument: $arg"
            show_usage
            ;;
    esac
done

# Main script logic
case "${COMMAND}" in
    start)
        start_cluster
        ;;
    stop)
        stop_cluster
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
