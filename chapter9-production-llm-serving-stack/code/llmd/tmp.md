# Hands-On Examples: LLM serving in Kubernetes with llm-d

This section demonstrates how to achieve the same architecture as Example 1 (different models, same inference engine) using llm-d's production-ready Helm charts and intelligent inference scheduling.

## Testing Guide: llm-d Multi-Model Setup with k3d

This guide helps you test this document using your existing k3d setup.

### ✅ Yes, k3d works with llm-d!

Your existing setup already demonstrates this:
- `09/code/llmd/` - Simple llm-d setup with LLMInferenceService CRDs
- `09/code/k3d/` - GPU-enabled k3d cluster setup

### 🎯 Cluster Strategy

**Important:** This guide creates a **NEW separate cluster** (`llmd-cluster`) for testing llm-d production setup. This keeps it isolated from your existing `mycluster-gpu` cluster which is used for:
- Manual vLLM deployments (`09/code/vllm/`)
- Manual SGLang deployments (`09/code/sglang/`)

This separation ensures:
- ✅ No conflicts between manual deployments and llm-d production setup
- ✅ You can test llm-d without affecting your working deployments
- ✅ Both clusters can run simultaneously (if you have enough resources)
- ✅ Easy cleanup: delete `llmd-cluster` when done testing

### 📍 Quick Path Reference

When following this document, replace paths like this:

| Document references | Actual path on your system |
|-------------------|---------------------------|
| `resources/llm-d/guides/prereq/client-setup` | `${LLMD_HOME}/guides/prereq/client-setup` |
| `resources/llm-d/guides/prereq/gateway-provider` | `${LLMD_HOME}/guides/prereq/gateway-provider` |
| `resources/llm-d/guides/inference-scheduling` | `${LLMD_HOME}/guides/inference-scheduling` |

**Good news!** The llm-d repository should be available at `${LLMD_HOME}` (set via `export LLMD_HOME=/path/to/llm-d`), so all the guides and scripts referenced in this document are available.

### Cluster Separation Summary

You now have (or will have) **two separate k3d clusters**:

| Cluster Name | Purpose | Location | Gateway |
|--------------|---------|----------|---------|
| `mycluster-gpu` | Manual deployments (vLLM, SGLang) | `09/code/vllm/`, `09/code/sglang/` | Traefik or custom API gateway |
| `llmd-cluster` | llm-d production setup | This guide | Istio/kGateway |

### Switching Between Clusters

```bash
# Work with your existing manual deployments
kubectl config use-context k3d-mycluster-gpu
kubectl get pods -n default

# Switch to llm-d cluster for testing
kubectl config use-context k3d-llmd-cluster
kubectl get pods -n llm-d-multi-model

# List all contexts
kubectl config get-contexts
```

### Managing Both Clusters

```bash
# List all clusters
k3d cluster list

# Start/stop clusters independently
k3d cluster start mycluster-gpu
k3d cluster start llmd-cluster

k3d cluster stop mycluster-gpu
k3d cluster stop llmd-cluster

# Delete llm-d cluster when done testing (keeps mycluster-gpu intact)
k3d cluster delete llmd-cluster
```

### Key Differences: Manual Setup vs llm-d Production

| Component | llm-d (Production) | Manual Setup |
|-----------|---------------------|-------------------|
| **Cluster** | New `llmd-cluster` | Existing `mycluster-gpu` |
| **Gateway** | Istio/kGateway (Gateway API) | Traefik (k3d default) or custom API gateway |
| **Model Deployment** | ModelService Helm charts | LLMInferenceService CRDs or direct Pods |
| **Routing** | InferencePool (automatic discovery) | Manual routing config in api-gateway.py |
| **Complexity** | High (production-ready) | Medium (development/testing) |

### Troubleshooting

#### Issue: Can't find llm-d guides

The guides should be available at `${LLMD_HOME}`. If you need to update the repository:
```bash
cd ${LLMD_HOME}
git pull
```

#### Issue: Gateway provider installation fails

For k3d, Istio can be resource-intensive. Consider:
- Using kGateway instead (lighter)
- Or using Traefik with manual routing (simpler, but less features)

#### Issue: ModelService Helm charts not found

The ModelService charts might be in a different repository. Check:
- llm-d official documentation
- GitHub releases for chart locations
- OCI registry: `oci://us-central1-docker.pkg.dev/k8s-staging-images/...`

#### Issue: GPU not detected

```bash
# Verify NVIDIA device plugin
kubectl get pods -n kube-system -l name=nvidia-device-plugin-ds

# Check node GPU capacity
kubectl describe node | grep -i gpu
```

**Important Notes:**
- All resources are deployed in the `llm-d-multi-model` namespace, not the default namespace
- Use `kubectl get pods -n llm-d-multi-model` to see deployed pods (not `kubectl get pods` which queries default namespace)
- The cluster must have GPU support and `nvidia` RuntimeClass configured
- `RELEASE_NAME_POSTFIX` cannot contain dots (.) - use hyphens instead (e.g., `llama-32-1b` not `llama-3.2-1b`)
- For k3d clusters, use `kubectl port-forward` to access services from outside the cluster

### Architecture Overview

llm-d provides a production-grade solution using:
- **Inference Gateway (IGW)**: Kubernetes-native gateway with intelligent load balancing
- **InferencePool**: Routes requests to appropriate ModelService instances
- **ModelService**: Helm chart for deploying vLLM model servers
- **Intelligent Scheduler**: Load-aware and prefix-cache aware routing

**Architecture:**

```
┌─────────────────────────────────────────────────────────────┐
│                    Client Applications                      │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            │ HTTP Request with 'model' field
                            │ (Unified Interface: Single Endpoint)
                            ▼
┌─────────────────────────────────────────────────────────────┐
│         Inference Gateway (Kubernetes Gateway API)          │
│  - Load balancing with prefix-cache awareness               │
│  - Intelligent request scheduling                           │
│  - Traffic routing to InferencePool                         │
│  - Unified Interface: /v1/chat/completions                  │
└───────────────┬─────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────┐
│              InferencePool (Routing Layer)                  │
│  - Routes requests to appropriate ModelService              │
│  - Model-aware request distribution                         │
│  - Health checking and load balancing                       │
│  - Unified Interface: Routes by 'model' field               │
└───────────────┬───────────────────────────┬─────────────────┘
                │                           │
                │                           │
    ┌───────────▼──────────┐    ┌───────────▼─────────┐
    │  ModelService 1      │    │  ModelService 2     │
    │  (Llama-3.2-1B)      │    │  (Qwen2.5-0.5B)     │
    │  vLLM Pods (1x)      │    │  vLLM Pods (1x)     │
    └──────────────────────┘    └─────────────────────┘

Alternative: Custom API Gateway (Advanced Unified Interface with owned_by support)
┌─────────────────────────────────────────────────────────────┐
│              Custom API Gateway (Optional)                  │
│  - Unified Interface: /v1/chat/completions                  │
│  - Routes by 'model' + 'owned_by' fields                    │
│  - Supports header-based routing (x-owned-by)               │
│  - Admin API for dynamic routing management                 │
└───────────────┬─────────────────────────────────────────────┘
                │
                │ Routes to InferencePool Gateway
                │ (preserves intelligent scheduling)
                ▼
┌─────────────────────────────────────────────────────────────┐
│         InferencePool Gateway (Kubernetes Gateway API)      │
│  - Intelligent routing, load balancing                      │
│  - Prefix-cache awareness                                   │
└───────────────┬─────────────────────────────────────────────┘
                │
                ▼
┌─────────────────────────────────────────────────────────────┐
│              InferencePool (Routing Layer)                  │
│  - Routes requests to appropriate ModelService              │
└───────────────┬───────────────────────────┬─────────────────┘
                │                           │
    ┌───────────▼──────────┐    ┌───────────▼─────────┐
    │  ModelService 1      │    │  ModelService 2     │
    │  (Llama-3.2-1B)      │    │  (Qwen2.5-0.5B)     │
    └──────────────────────┘    └─────────────────────┘
```

### Prerequisites

**Set environment variables:**
```bash
export LLMD_HOME=/path/to/llm-d  # Set to your llm-d repository path
export LLMD_CONFIG_DIR=<path-to-your-config-directory>  # Directory containing values YAML files
```

1. **Install Client Tools:**
   ```bash
   cd ${LLMD_HOME}/guides/prereq/client-setup
   ./install-deps.sh
   ```
   This installs: `helm`, `helmfile`, `kubectl`, `yq`, `git`

```
 🌈 $   export LLMD_HOME=/path/to/llm-d  # Set to your llm-d repository path
   cd ${LLMD_HOME}/guides/prereq/client-setup
   ./install-deps.sh
Installing yq...
[sudo] password for xxx: 
📦 helm-diff plugin not found. Installing v3.11.0...
Downloading https://github.com/databus23/helm-diff/releases/download/v3.11.0/helm-diff-linux-amd64.tgz
Preparing to install into /home/wukong/.local/share/helm/plugins/helm-diff
Installed plugin: diff
📦 helmfile not found. Installing 1.2.1...
✅ All tools installed successfully.
```

2. **Create k3d Cluster for llm-d:**
   ```bash
   # Clean up existing cluster if it exists (optional, if recreating)
   # Check if cluster exists
   k3d cluster list | grep llmd-cluster && echo "Cluster exists, deleting..." && k3d cluster delete llmd-cluster || echo "Cluster does not exist, proceeding to create..."
   
   # Create a new k3d cluster specifically for llm-d
   # Note: The cluster currently has 2 agents (agent-0 and agent-1) to schedule different models to different nodes
   k3d cluster create llmd-cluster \
     --image k3s-cuda:v1.33.6-cuda-12.2.0-working \
     --gpus=all \
     --servers 1 \
     --agents 2 \
     --volume /raid/models:/models \
     --k3s-arg '--disable=traefik@server:0' \
     --port "8080:80@loadbalancer" \
     --port "8443:443@loadbalancer"
   
   # Merge kubeconfig
   k3d kubeconfig merge llmd-cluster --kubeconfig-merge-default
   
   # Switch to the new cluster context
   kubectl config use-context k3d-llmd-cluster
   
   # Verify cluster is ready
   kubectl get nodes
   kubectl cluster-info
   ```
   
   **Note:** 
   - k3d doesn't support assigning specific GPUs to specific nodes during cluster creation. All nodes will have access to all GPUs. To ensure Pods use specific GPUs, use `nodeSelector` in your deployment values (see Step 1 and Step 2 below).
   - The `llmd-cluster` currently has 2 agents (`agent-0` and `agent-1`), allowing different models to be scheduled to different nodes. Models are assigned using `nodeSelector` with `kubernetes.io/hostname` (e.g., `k3d-llmd-cluster-agent-0` for the first model, `k3d-llmd-cluster-agent-1` for the second model).

3. **Create Namespace:**
   ```bash
   export NAMESPACE=llm-d-multi-model
   kubectl create namespace ${NAMESPACE}
   ```

4. **Create HuggingFace Token Secret:**
   ```bash
   export HF_TOKEN='your_huggingface_token_here'
   kubectl create secret generic llm-d-hf-token \
     --from-literal="HF_TOKEN=${HF_TOKEN}" \
     --namespace "${NAMESPACE}" \
     --dry-run=client -o yaml | kubectl apply -f -
   ```

5. **Deploy Gateway Provider:**
   ```bash
   cd ${LLMD_HOME}/guides/prereq/gateway-provider
   ./install-gateway-provider-dependencies.sh
   
   # Deploy Istio Gateway Provider
   helmfile apply -f istio.helmfile.yaml
   ```

**Note:** Based on actual deployment experience, the cluster was created with `--agents 1` (not 2) for the initial setup. The commands used were:

```bash
# Create cluster with 1 agent
k3d cluster create llmd-cluster \
  --image k3s-cuda:v1.33.6-cuda-12.2.0-working \
  --gpus=all \
  --servers 1 \
  --agents 1 \
  --volume /raid/models:/models \
  --k3s-arg '--disable=traefik@server:0' \
  --port "8080:80@loadbalancer" \
  --port "8443:443@loadbalancer"

# Merge kubeconfig and switch context
k3d kubeconfig merge llmd-cluster --kubeconfig-merge-default
kubectl config use-context k3d-llmd-cluster

# Verify cluster
kubectl get nodes
kubectl cluster-info

# Deploy Gateway Provider (from gateway-provider directory)
cd ${LLMD_HOME}/guides/prereq/gateway-provider
helmfile apply -f istio.helmfile.yaml

# Navigate to inference-scheduling directory for model deployment
cd ${LLMD_HOME}/guides/inference-scheduling
```

Results:
```
 🌈 $   ./install-gateway-provider-dependencies.sh
✅ 📜 Base CRDs: Installing...
customresourcedefinition.apiextensions.k8s.io/gatewayclasses.gateway.networking.k8s.io created
customresourcedefinition.apiextensions.k8s.io/gateways.gateway.networking.k8s.io created
customresourcedefinition.apiextensions.k8s.io/grpcroutes.gateway.networking.k8s.io created
customresourcedefinition.apiextensions.k8s.io/httproutes.gateway.networking.k8s.io created
customresourcedefinition.apiextensions.k8s.io/referencegrants.gateway.networking.k8s.io created
✅ 🚪 GAIE CRDs: Installing...
customresourcedefinition.apiextensions.k8s.io/inferenceobjectives.inference.networking.x-k8s.io created
customresourcedefinition.apiextensions.k8s.io/inferencepoolimports.inference.networking.x-k8s.io created
customresourcedefinition.apiextensions.k8s.io/inferencepools.inference.networking.k8s.io created
customresourcedefinition.apiextensions.k8s.io/inferencepools.inference.networking.x-k8s.io created
```

### Actual Deployment Workflow (Based on Real Experience)

**Note:** The following commands were actually used to create and deploy `llmd-cluster`:

```bash
# 1. Create k3d cluster with 2 agents (for scheduling different models to different nodes)
k3d cluster create llmd-cluster \
  --image k3s-cuda:v1.33.6-cuda-12.2.0-working \
  --gpus=all \
  --servers 1 \
  --agents 2 \
  --volume /raid/models:/models \
  --k3s-arg '--disable=traefik@server:0' \
  --port "8080:80@loadbalancer" \
  --port "8443:443@loadbalancer"

# 2. Merge kubeconfig and switch context
k3d kubeconfig merge llmd-cluster --kubeconfig-merge-default
kubectl config use-context k3d-llmd-cluster

# 3. Verify cluster
kubectl get nodes
kubectl cluster-info

# 4. Set namespace
export NAMESPACE=llm-d-multi-model

# 5. Deploy Gateway Provider (Istio)
cd ${LLMD_HOME}/guides/prereq/gateway-provider
helmfile apply -f istio.helmfile.yaml

# 6. Navigate to inference-scheduling directory
cd ${LLMD_HOME}/guides/inference-scheduling

# 7. Prepare values file for first model
mkdir -p ms-inference-scheduling
cp ${LLMD_CONFIG_DIR}/llama-3.2-1b-values.yaml ms-inference-scheduling/values.yaml

# 8. Deploy first model using helmfile
RELEASE_NAME_POSTFIX=llama-32-1b helmfile apply -n ${NAMESPACE}

# 9. Verify deployment
kubectl get pods -n ${NAMESPACE} -o wide
kubectl get pods -n ${NAMESPACE} -l llm-d.ai/role=decode

# 10. Fix permissions for model cache (if needed)
sudo chmod -R 775 /raid/models/hub

# 11. Test the deployment
# Port-forward to ModelService pod
POD_NAME=$(kubectl get pods -n ${NAMESPACE} -l llm-d.ai/role=decode | grep llama-32-1b | awk '{print $1}')
kubectl port-forward -n ${NAMESPACE} pod/${POD_NAME} 8002:8000

# Or access via Gateway service
GATEWAY_IP=$(kubectl get svc -n ${NAMESPACE} infra-llama-32-1b-inference-gateway-istio -o jsonpath='{.spec.clusterIP}')
curl http://${GATEWAY_IP}/v1/models
```

**Key Points:**
- Cluster was created with `--agents 1` (single agent node)
- Used `helmfile apply` from `${LLMD_HOME}/guides/inference-scheduling` directory
- Model values file was copied to `ms-inference-scheduling/values.yaml` as expected by helmfile
- Gateway provider (Istio) was deployed first before deploying models

### Supported Inference Server Types

**llm-d supports multiple inference server types**, including:

1. **vLLM** (`type: vllm`)
   - Image: `vllm/vllm-openai:v0.12.0` or `ghcr.io/llm-d/llm-d-cuda:v0.4.0`
   - Command: `python3 -m vllm.entrypoints.openai.api_server`
   - Used in: `llm-d-multi-model`, `llm-d-multi-engine`

2. **SGLang** (`type: sglang`) ✅
   - Image: `lmsysorg/sglang:v0.5.6.post2-runtime`
   - Command: `python3 -m sglang.launch_server`
   - Used in: `llm-d-multi-model`, `llm-d-multi-engine`
   - **Example**: See `09/code/llmd/llm-d-multi-model/sglang-llminference.yaml` and `09/code/llmd/llm-d-multi-engine/sglang-llminference.yaml`

**Deploying SGLang with llm-d:**

SGLang can be deployed using the `LLMInferenceService` CRD with `inferenceServer.type: sglang`:

```yaml
apiVersion: llm-d.ai/v1alpha1
kind: LLMInferenceService
metadata:
  name: sglang-model-name
spec:
  inferenceServer:
    type: sglang
    image: lmsysorg/sglang:v0.5.6.post2-runtime
    command:
      - python3
      - -m
      - sglang.launch_server
    args:
      - --model-path
      - <model-name>
      - --port
      - "8000"
      - --mem-fraction-static
      - "0.1"
```

**Note:** The existing deployments in `llm-d-multi-model` and `llm-d-multi-engine` already include working SGLang configurations that can be used as reference.

### Step 1: Deploy First Model (Llama-3.2-1B-Instruct)

Create a custom values file for the first model. The file should be created in your config directory:

**File:** `${LLMD_CONFIG_DIR}/llama-3.2-1b-values.yaml`

**Note:** Ensure `LLMD_CONFIG_DIR` environment variable is set (see Prerequisites) before creating the file.

```yaml
multinode: false

modelArtifacts:
  uri: "hf://meta-llama/Llama-3.2-1B-Instruct"
  name: "meta-llama/Llama-3.2-1B-Instruct"
  size: 20Gi
  authSecretName: "llm-d-hf-token"

routing:
  servicePort: 8000
  proxy:
    image: ghcr.io/llm-d/llm-d-routing-sidecar:v0.4.0-rc.1
    connector: nixlv2
    secure: false

decode:
  create: true
  replicas: 1
  extraConfig:
    runtimeClassName: nvidia  # Required for k3d GPU access
    nodeSelector:
      kubernetes.io/hostname: k3d-llmd-cluster-agent-0  # Schedule to agent-0
    securityContext:
      fsGroup: 0  # Set group to root so pod can write to /models/hub (llm-d runs as uid=2000, needs fsGroup to write)
  containers:
  - name: "vllm"
    image: ghcr.io/llm-d/llm-d-cuda:v0.4.0  # Updated to v0.4.0
    modelCommand: vllmServe
    args:
      - "--disable-uvicorn-access-log"
    env:
      - name: CUDA_VISIBLE_DEVICES
        value: "0"  # Use GPU 0 on agent-0
    resources:
      limits:
        nvidia.com/gpu: "1"
        memory: 8Gi
      requests:
        nvidia.com/gpu: "1"
        memory: 6Gi
    mountModelVolume: false  # Using hostPath volume from k3d mount (/raid/models on host → /models in container)
    volumeMounts:
    - name: models
      mountPath: /models
      readOnly: false  # Allow vLLM to write cache to /models/hub
  volumes:
  - name: models
    hostPath:
      path: /models  # k3d maps /raid/models (host) to /models (container)
      type: Directory

prefill:
  create: false
```

**Deploy:**

```bash
cd ${LLMD_HOME}/guides/inference-scheduling

# Create directory and copy custom values file
# Note: helmfile expects ms-inference-scheduling/values.yaml, so we'll use that path
mkdir -p ms-inference-scheduling
cp ${LLMD_CONFIG_DIR}/llama-3.2-1b-values.yaml ms-inference-scheduling/values.yaml

# Disable Prometheus monitoring to avoid ServiceMonitor CRD requirement
# Backup original file and disable Prometheus monitoring
cp gaie-inference-scheduling/values.yaml gaie-inference-scheduling/values.yaml.bak
# Use yq to disable Prometheus (yq should be installed from prerequisites)
yq eval '.inferenceExtension.monitoring.prometheus.enabled = false' -i gaie-inference-scheduling/values.yaml

# Deploy first model with custom release name
# Note: RELEASE_NAME_POSTFIX cannot contain dots (.) - use hyphens instead
RELEASE_NAME_POSTFIX=llama-32-1b \
helmfile apply -n ${NAMESPACE}

# Wait for pods to be ready (this may take several minutes for model download)
kubectl wait --for=condition=ready pod -l llm-d.ai/role=decode -n ${NAMESPACE} --timeout=600s
```

### Step 2: Deploy Second Model (Qwen2.5-0.5B-Instruct)

**Important:** The second model should use the **same** Gateway and InferencePool as the first model. Only deploy the ModelService for the second model, not a new Gateway/InferencePool.

Create a custom values file for the second model. The file should be created in your config directory:

**File:** `${LLMD_CONFIG_DIR}/qwen2.5-0.5b-values.yaml`

```yaml
multinode: false

modelArtifacts:
  uri: "hf://Qwen/Qwen2.5-0.5B-Instruct"  # HuggingFace model path
  name: "Qwen/Qwen2.5-0.5B-Instruct"
  size: 2Gi
  authSecretName: ""  # Qwen2.5-0.5B-Instruct is not gated, no token needed

routing:
  servicePort: 8000
  proxy:
    image: ghcr.io/llm-d/llm-d-routing-sidecar:v0.4.0-rc.1
    connector: nixlv2
    secure: false

decode:
  create: true
  replicas: 1
  extraConfig:
    runtimeClassName: nvidia  # Required for k3d GPU access
    nodeSelector:
      kubernetes.io/hostname: k3d-llmd-cluster-agent-1  # Schedule to agent-1
    securityContext:
      fsGroup: 0  # Set group to root so pod can write to /models/hub (llm-d runs as uid=2000, needs fsGroup to write)
    volumes:
    - name: models
      hostPath:
        path: /models  # k3d maps /raid/models (host) to /models (container)
        type: Directory
  containers:
  - name: "vllm"
    image: ghcr.io/llm-d/llm-d-cuda:v0.4.0  # Updated to v0.4.0
    modelCommand: vllmServe
    args:
      - "--disable-uvicorn-access-log"
      # Note: Qwen2.5-0.5B-Instruct doesn't require --trust-remote-code
      - "--gpu-memory-utilization"
      - "0.1"  # Use 10% of GPU memory (target: ~14GB out of 140GB)
      - "--max-model-len"
      - "32768"  # Set to 32K as model supports 32K
    env:
      - name: NVIDIA_VISIBLE_DEVICES
        value: "1"  # Only expose GPU 1 to container (host GPU 1)
      - name: CUDA_VISIBLE_DEVICES
        value: "0"  # Use GPU 0 within container (which maps to host GPU 1)
      - name: CUDA_DEVICE_ORDER
        value: "PCI_BUS_ID"  # Ensure consistent GPU ordering
      - name: HF_HOME
        value: "/models/hub"  # Now writable after permission fix
      - name: TRANSFORMERS_CACHE
        value: "/models/hub"
      - name: HF_HUB_CACHE
        value: "/models/hub"
    resources:
      limits:
        nvidia.com/gpu: "1"
        memory: 8Gi
      requests:
        nvidia.com/gpu: "1"
        memory: 4Gi
    mountModelVolume: true  # Use default model volume mount for HuggingFace models

prefill:
  create: false
```

**Deploy:**

```bash
# Update values file for the second model
# Note: helmfile expects ms-inference-scheduling/values.yaml
cp ${LLMD_CONFIG_DIR}/qwen2.5-0.5b-values.yaml ms-inference-scheduling/values.yaml

# Prometheus should already be disabled from previous step
# Deploy second model with different release name
# Note: RELEASE_NAME_POSTFIX cannot contain dots (.) - use hyphens instead
# IMPORTANT: This will deploy infra, gaie, and ms, but we only need ms.
# The existing Gateway and InferencePool will work with both models.
RELEASE_NAME_POSTFIX=qwen2-5-0-5b \
helmfile -e default -f helmfile.yaml.gotmpl --namespace ${NAMESPACE} apply

# Clean up duplicate Gateway and InferencePool (keep only one shared instance)
helm uninstall infra-qwen2-5-0-5b -n ${NAMESPACE} 2>/dev/null || true
helm uninstall gaie-qwen2-5-0-5b -n ${NAMESPACE} 2>/dev/null || true

# Wait for pods to be ready (this may take several minutes for model download)
kubectl wait --for=condition=ready pod -l llm-d.ai/role=decode -n ${NAMESPACE} --timeout=600s
```

### Step 3: Configure InferencePool for Multi-Model Routing

The InferencePool automatically discovers ModelService instances and routes requests based on the `model` field. 

**Note:** llm-d's InferencePool supports plugin-based customization through `pluginsCustomConfig`. The default configuration routes by `model` field only. To route by both `model` and `owned_by` (inference engine type), you would need to:

1. **Use custom plugins** - Create a custom filter plugin that reads the `owned_by` field from:
   - **Request body** (JSON field) - requires parsing request body
   - **Request header** (e.g., `x-owned-by` or `X-Owned-By`) - requires header reading capability
   - Then filter endpoints based on labels matching the `owned_by` value
   - This requires modifying the `llm-d-inference-scheduler` source code

2. **Use a custom API Gateway** - Deploy a custom API Gateway (like the one in `09/code/k3d/ref.md`) that routes based on both `model` and `owned_by` fields (from body or header) before requests reach the InferencePool.

**Available Plugin Types:**
- `prefill-header-handler` - Handles prefill-specific headers (adds `X-Prefiller-Host-Port` for P/D disaggregation)
- `prefill-filter` / `decode-filter` - Filters endpoints based on request type
- `queue-scorer` - Scores endpoints based on queue depth
- `kv-cache-utilization-scorer` - Scores based on KV cache usage
- `prefix-cache-scorer` - Scores based on prefix cache hit rate
- `max-score-picker` / `random-picker` - Picks endpoint based on scores
- `slo-aware-profile-handler` - Switches profile based on SLO headers (e.g., `x-prediction-based-scheduling: true`)

**Using `owned_by` in Request Headers:**

While `prefill-header-handler` is designed for P/D disaggregation (adding `X-Prefiller-Host-Port`), the plugin system does support reading custom headers. However:

1. **Default plugins don't support `owned_by` header routing** - The existing filter plugins (`prefill-filter`, `decode-filter`) filter by request type, not by custom headers like `x-owned-by`.

2. **Possible approach**: You could potentially create a custom filter plugin that:
   - Reads `x-owned-by` or `X-Owned-By` header from the request
   - Filters endpoints based on labels matching the `owned_by` value
   - This would require modifying `llm-d-inference-scheduler` source code

3. **✅ Custom API Gateway (Implemented)**: A custom API Gateway has been created at `09/code/llmd/llmd-api-gateway.yaml` that:
   - Reads `owned_by` from request body, HTTP header (`x-owned-by`), or query parameters
   - Automatically discovers ModelService instances and InferencePool Gateway services from Kubernetes
   - **Routes through InferencePool Gateway services** (e.g., `infra-llama-32-1b-inference-gateway-istio`) to leverage llm-d's intelligent routing, load balancing, and prefix-cache awareness
   - Falls back to direct pod IP routing only if InferencePool Gateway service is not available
   - Routes to different ModelServices based on `model` and `owned_by`
   - Works without modifying llm-d source code
   - **Important**: The Custom API Gateway uses InferencePool Gateway services (not the IP headless services), ensuring you get all the benefits of llm-d's intelligent scheduling, load balancing, and prefix-cache awareness
   - See **Step 5: Deploy Custom API Gateway** below for deployment instructions

**Example**: See `llm-d/guides/predicted-latency-based-scheduling/README.md` for how custom headers (`x-prediction-based-scheduling`, `x-slo-ttft-ms`) are used with `slo-aware-profile-handler`.

See `llm-d/guides/wide-ep-lws/manifests/inferencepool.values.yaml` for a custom plugin configuration example.

Create or update the InferencePool configuration:

**File:** `inferencepool-multi-model-values.yaml`

```yaml
provider:
  name: istio  # or kgateway, gke, etc.

inferencePool:
  apiVersion: inference.networking.k8s.io/v1
  metadata:
    name: multi-model-pool
  spec:
    # InferencePool automatically discovers ModelService instances
    # and routes based on model field in requests
```

**Deploy InferencePool:**

```bash
helm install multi-model-pool \
  -n ${NAMESPACE} \
  -f inferencepool-multi-model-values.yaml \
  --set "provider.name=istio" \
  --set "inferenceExtension.monitoring.prometheus.enable=true" \
  oci://us-central1-docker.pkg.dev/k8s-staging-images/gateway-api-inference-extension/charts/inferencepool \
  --version v1.2.0-rc.1
```

### Step 4: Deploy HTTPRoute

Create an HTTPRoute to expose the InferencePool:

**File:** `httproute-multi-model.yaml`

```yaml
apiVersion: gateway.networking.k8s.io/v1
kind: HTTPRoute
metadata:
  name: multi-model-route
  namespace: ${NAMESPACE}
spec:
  parentRefs:
  - name: gateway  # Your Gateway name
    namespace: istio-system
  rules:
  - matches:
    - path:
        type: PathPrefix
        value: /
    backendRefs:
    - name: multi-model-pool-epp
      port: 8000
```

**Deploy:**

```bash
kubectl apply -f httproute-multi-model.yaml -n ${NAMESPACE}
```

### Step 5: Deploy Custom API Gateway (Optional - for owned_by routing)

**Note:** This step is optional. If you only need routing by `model` field, the InferencePool (deployed in Step 1) is sufficient. Deploy this custom API Gateway only if you need to route by both `model` and `owned_by` fields.

The custom API Gateway provides:
- ✅ Routing by both `model` and `owned_by` fields
- ✅ Automatic service discovery from Kubernetes
- ✅ Support for `owned_by` in request body, HTTP header (`x-owned-by`), or query parameters
- ✅ Unified interface for all models

**Deploy the Custom API Gateway:**

**Option A: Using deployment script (Recommended)**

```bash
export NAMESPACE=llm-d-multi-model
cd ${LLMD_CONFIG_DIR}

# Deploy using the script
./deploy-gateway.sh
```

**Option B: Manual deployment**

```bash
export NAMESPACE=llm-d-multi-model

# Deploy the Gateway (includes ConfigMap, Deployment, ServiceAccount, Role, RoleBinding, and Service)
kubectl apply -f ${LLMD_CONFIG_DIR}/llmd-api-gateway.yaml

# Wait for Gateway to be ready
kubectl wait --for=condition=ready pod -l app=llmd-gateway -n ${NAMESPACE} --timeout=60s

# Verify Gateway is running
kubectl get pod,svc -l app=llmd-gateway -n ${NAMESPACE}
```

**Trigger Service Discovery:**

The Gateway automatically discovers services on startup, but you can manually trigger discovery:

```bash
# Port-forward to Gateway (using standard port 8000)
kubectl port-forward -n ${NAMESPACE} svc/llmd-api-gateway 8000:8000

# In another terminal, trigger service discovery
curl -X POST http://localhost:8000/admin/discover

# Check routing configuration
curl http://localhost:8000/admin/api/routing
```

**Test the Gateway:**

```bash
# List available models
curl http://localhost:8000/v1/models

# Test with owned_by in request body
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "owned_by": "vllm",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "max_tokens": 50
  }'

# Test with owned_by in HTTP header
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "x-owned-by: vllm" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "max_tokens": 50
  }'
```

**Gateway Features:**

1. **Automatic Service Discovery**: 
   - Discovers ModelService instances from Kubernetes using label selector `llm-d.ai/role=decode`
   - Automatically finds InferencePool Gateway services (e.g., `infra-*-inference-gateway-istio`)
   - Prefers InferencePool Gateway services over direct pod IP routing
2. **InferencePool Integration**: 
   - Routes requests through InferencePool Gateway services to leverage intelligent scheduling
   - Preserves all llm-d features: load balancing, prefix-cache awareness, health checking
   - Falls back to direct pod IP routing only if InferencePool Gateway is unavailable
3. **Multiple Input Sources**: Reads `owned_by` from:
   - Request body JSON field: `{"model": "...", "owned_by": "vllm"}`
   - HTTP header: `x-owned-by: vllm` or `X-Owned-By: vllm`
   - Query parameter: `?owned_by=vllm`
4. **Backward Compatibility**: Also supports `engine` and `inference_server` fields
5. **Admin API**: 
   - `GET /admin/api/routing` - List all routing mappings
   - `POST /admin/api/routing` - Add a routing mapping
   - `DELETE /admin/api/routing?model=...&owned_by=...` - Delete a routing mapping
   - `POST /admin/discover` - Manually trigger service discovery

**Integration with InferencePool:**

The custom API Gateway is designed to **work with InferencePool Gateway**, not bypass it:
- **✅ Recommended**: Custom API Gateway → InferencePool Gateway → InferencePool → ModelServices
  - Custom API Gateway routes requests to InferencePool Gateway services (e.g., `infra-llama-32-1b-inference-gateway-istio`)
  - InferencePool Gateway provides intelligent routing, load balancing, prefix-cache awareness, and health checking
  - This ensures you get all the benefits of llm-d's production-ready features
- **Fallback**: Custom API Gateway → Direct Pod IP (only if InferencePool Gateway service is not available)
  - Used as a last resort when InferencePool Gateway service discovery fails
  - Not recommended for production use

**Why use InferencePool Gateway?**
- ✅ Intelligent request scheduling based on queue depth and KV cache utilization
- ✅ Prefix-cache aware routing for better performance
- ✅ Automatic health checking and failover
- ✅ Load balancing across multiple ModelService replicas
- ✅ Production-ready and battle-tested
- ✅ Kubernetes Gateway API integration for advanced traffic management

```
 🚀 $k get pods
NAME                                                         READY   STATUS    RESTARTS   AGE
gaie-llama-32-1b-epp-5f8479848-7tlk8                         1/1     Running   0          3h17m
infra-llama-32-1b-inference-gateway-istio-56b9656d76-4pjj5   1/1     Running   0          3h17m
llmd-api-gateway-794ffd5f74-tm4zz                            1/1     Running   0          12m
ms-llama-32-1b-llm-d-modelservice-decode-74485b6976-fnt2v    2/2     Running   0          113m
ms-qwen2-5-0-5b-llm-d-modelservice-decode-74d49c849f-4xjq7   2/2     Running   0          113m
```

**Pod Architecture Explanation:**

When you run `kubectl get pods -n llm-d-multi-model`, you will see 5 pods (or more if you scale up). This is the expected llm-d production architecture:

1. **InferencePool Gateway** (`infra-*-inference-gateway-istio`):
   - Kubernetes Gateway API implementation
   - Provides unified entry point for all requests
   - Handles intelligent routing and load balancing
   - Routes requests to InferencePool based on `model` field

2. **InferencePool EPP** (`gaie-*-epp`):
   - Extension Point Processor for InferencePool
   - Implements intelligent scheduling plugins (queue scoring, KV cache utilization, prefix-cache awareness)
   - Processes routing decisions based on real-time metrics
   - Communicates with ModelService pods via routing sidecars

3. **Custom API Gateway** (`llmd-api-gateway`) - Optional:
   - Provides `owned_by` field routing support
   - Routes requests to InferencePool Gateway (preserves intelligent scheduling)
   - Automatically discovers services from Kubernetes
   - Admin API for dynamic routing management

4. **ModelService Pods** (`ms-*-decode`) - 2 pods, each with 2/2 Ready:
   - Each pod contains:
     - **vLLM container**: Actual model inference engine (runs the LLM model)
     - **routing-proxy sidecar**: Communicates with InferencePool for intelligent scheduling
   - `2/2 Ready` means both containers in the pod are running successfully
   - One pod per model (Llama and Qwen in this example)

**Why so many pods?**

This is llm-d's production-ready architecture that provides:
- ✅ **Intelligent request scheduling** (not just round-robin) - routes based on queue depth, KV cache utilization
- ✅ **Load balancing** - distributes requests across replicas based on real-time metrics
- ✅ **Prefix-cache awareness** - routes requests to pods with matching prefix cache for better performance
- ✅ **Health checking and automatic failover** - automatically routes away from unhealthy pods
- ✅ **Kubernetes-native integration** - uses Gateway API for standard Kubernetes traffic management
- ✅ **Scalability** - easy to add more models or scale replicas
- ✅ **High availability** - multiple components ensure system resilience

All components work together to provide a production-grade LLM serving solution with intelligent scheduling and optimal resource utilization.


### Step 6: Using the Unified Interface

**✅ Yes, llm-d cluster supports unified interface!**

Once deployed, all requests can go through a unified interface (single endpoint) that automatically routes to the correct model service. You have **two options** for unified interface:

#### Option 1: InferencePool Gateway (Default - Basic Unified Interface)

The InferencePool Gateway provides a unified interface that routes based on `model` field only.

**Access via InferencePool Gateway:**

```bash
export NAMESPACE=llm-d-multi-model

# Port-forward to InferencePool Gateway
kubectl port-forward -n ${NAMESPACE} svc/infra-llama-32-1b-inference-gateway-istio 8080:80

# In another terminal, use the unified interface
# 1. List all available models
curl http://localhost:8080/v1/models
```

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "meta-llama/Llama-3.2-1B-Instruct",
      "object": "model",
      "created": 0,
      "owned_by": "vllm"
    },
    {
      "id": "Qwen/Qwen2.5-0.5B-Instruct",
      "object": "model",
      "created": 0,
      "owned_by": "vllm"
    }
  ]
}
```

**2. Chat Completion with Llama Model (using unified interface):**

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "messages": [
      {"role": "user", "content": "What is machine learning?"}
    ],
    "max_tokens": 100
  }'
```

**3. Chat Completion with Qwen Model (using unified interface):**

```bash
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [
      {"role": "user", "content": "What is machine learning?"}
    ],
    "max_tokens": 100
  }'
```

**Key Features:**
- ✅ **Unified Interface**: Single endpoint (`/v1/chat/completions`) for all models
- ✅ **Automatic Routing**: Routes based on `model` field in request body
- ✅ **Model Discovery**: Aggregates model lists from all ModelServices (`/v1/models`)
- ⚠️ **Limitation**: Only routes by `model` field, not by `owned_by` (inference engine type)

#### Option 2: Custom API Gateway (Advanced Unified Interface with owned_by support)

The Custom API Gateway provides a unified interface that routes based on both `model` and `owned_by` fields.

**Access via Custom API Gateway:**

```bash
export NAMESPACE=llm-d-multi-model

# Port-forward to Custom API Gateway (using standard port 8000)
kubectl port-forward -n ${NAMESPACE} svc/llmd-api-gateway 8000:8000

# In another terminal, use the unified interface
# 1. List all available models
curl http://localhost:8000/v1/models
```

**2. Chat Completion with owned_by in request body:**

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "owned_by": "vllm",
    "messages": [
      {"role": "user", "content": "What is machine learning?"}
    ],
    "max_tokens": 100
  }'
```

**3. Chat Completion with owned_by in HTTP header:**

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "x-owned-by: vllm" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [
      {"role": "user", "content": "What is machine learning?"}
    ],
    "max_tokens": 100
  }'
```

**Key Features:**
- ✅ **Unified Interface**: Single endpoint (`/v1/chat/completions`) for all models
- ✅ **Advanced Routing**: Routes based on both `model` and `owned_by` fields
- ✅ **Multiple Input Sources**: Reads `owned_by` from request body, HTTP header, or query parameters
- ✅ **Model Discovery**: Aggregates model lists from all ModelServices (`/v1/models`)
- ✅ **Admin API**: Dynamic routing configuration management

#### Comparison: Unified Interface Options

| Feature | InferencePool Gateway | Custom API Gateway |
|---------|----------------------|-------------------|
| **Unified Interface** | ✅ | ✅ |
| **Single Endpoint** | ✅ | ✅ |
| **Route by `model`** | ✅ | ✅ |
| **Route by `owned_by`** | ❌ | ✅ |
| **Header-based routing** | ❌ | ✅ |
| **Admin API** | ❌ | ✅ |
| **Auto Service Discovery** | ✅ | ✅ |

**Recommendation:**
- Use **InferencePool Gateway** if you only need routing by `model` field (simpler, production-ready)
- Use **Custom API Gateway** if you need routing by both `model` and `owned_by` fields (advanced use cases)

### Step 7: Test Multi-Model Routing

**For k3d clusters, use port-forward to access the service:**

```bash
export NAMESPACE=llm-d-multi-model

# Option 1: Port-forward to Custom API Gateway (recommended if deployed)
# Use standard port 8000 for consistency
kubectl port-forward -n ${NAMESPACE} svc/llmd-api-gateway 8000:8000

# In another terminal, test the API
curl -X GET http://localhost:8000/v1/models

# Test with owned_by in request body
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "owned_by": "vllm",
    "messages": [
      {"role": "user", "content": "Hello!"}
    ],
    "max_tokens": 50
  }'

# Option 2: Port-forward directly to ModelService pod (for direct testing)
# Note: Use a different port (e.g., 8001) to avoid conflicts
POD_NAME=$(kubectl get pods -n ${NAMESPACE} -l llm-d.ai/role=decode | grep llama-32-1b | awk '{print $1}')
kubectl port-forward -n ${NAMESPACE} pod/${POD_NAME} 8001:8000

# In another terminal, test the API
curl -X GET http://localhost:8001/v1/models

# Option 3: Port-forward to InferencePool Gateway (through InferencePool routing)
kubectl port-forward -n ${NAMESPACE} svc/infra-llama-32-1b-inference-gateway-istio 8080:80
curl -X GET http://localhost:8080/v1/models
```

**Or use the Gateway service directly from within cluster:**

```bash
# Option A: InferencePool Gateway (basic unified interface)
export NAMESPACE=llm-d-multi-model
GATEWAY_IP=$(kubectl get svc -n ${NAMESPACE} infra-llama-32-1b-inference-gateway-istio -o jsonpath='{.spec.clusterIP}')
echo "InferencePool Gateway IP: ${GATEWAY_IP}"

# List Available Models
curl http://${GATEWAY_IP}/v1/models

# Request to any model (unified interface)
curl http://${GATEWAY_IP}/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "messages": [
      {"role": "user", "content": "What is machine learning?"}
    ],
    "max_tokens": 100
  }'

# Option B: Custom API Gateway (advanced unified interface with owned_by support)
CUSTOM_GATEWAY_IP=$(kubectl get svc -n ${NAMESPACE} llmd-api-gateway -o jsonpath='{.spec.clusterIP}')
echo "Custom API Gateway IP: ${CUSTOM_GATEWAY_IP}"

# List Available Models
curl http://${CUSTOM_GATEWAY_IP}:8000/v1/models

# Request with owned_by in body
curl http://${CUSTOM_GATEWAY_IP}:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "owned_by": "vllm",
    "messages": [
      {"role": "user", "content": "What is machine learning?"}
    ],
    "max_tokens": 100
  }'
```

### Key Advantages of llm-d Approach

1. **Production-Ready**: Tested and benchmarked configurations
2. **Unified Interface**: Single endpoint for all models - both InferencePool Gateway and Custom API Gateway provide unified interfaces
3. **Intelligent Routing**: Prefix-cache aware and load-aware balancing
4. **Automatic Discovery**: InferencePool automatically discovers ModelService instances
5. **Monitoring**: Built-in Prometheus metrics and Grafana dashboards
6. **Scalability**: Easy to add more models or scale replicas
7. **Multi-Hardware Support**: Works with NVIDIA, AMD, Intel XPU, Google TPU
8. **Advanced Features**: Supports prefill/decode disaggregation, expert parallelism, etc.
9. **Flexible Routing**: Choose between basic (model-only) or advanced (model + owned_by) unified interface

**Customization Options:**
- **Plugin System**: llm-d's InferencePool supports plugin-based customization via `pluginsCustomConfig`. You can configure custom filters, scorers, and pickers to customize routing behavior.
- **Available Plugins**: Filters (`prefill-filter`, `decode-filter`), Scorers (`queue-scorer`, `kv-cache-utilization-scorer`, `prefix-cache-scorer`), Pickers (`max-score-picker`, `random-picker`), and Handlers (`prefill-header-handler`, `pd-profile-handler`).
- **Example**: See `llm-d/guides/wide-ep-lws/manifests/inferencepool.values.yaml` for custom plugin configuration.

**Limitations:**
- **Default Routing**: The default InferencePool configuration routes based on `model` field only, not by inference engine type (`owned_by` field).
- **Multi-Engine Routing**: To route by both `model` and `owned_by` (e.g., same model with vLLM vs SGLang), you have three options:
  1. **✅ Custom API Gateway (Recommended)**: Use the custom API Gateway deployed in Step 5 (`09/code/llmd/llmd-api-gateway.yaml`) that automatically discovers services and routes based on both `model` and `owned_by` fields. This works without modifying llm-d source code.
  2. **Custom Plugin**: Create a custom filter plugin that reads `owned_by` from request body or header (requires modifying `llm-d-inference-scheduler` source code).
  3. **Reference Implementation**: See `09/code/k3d/ref.md` Example 2 for another custom API Gateway implementation.

## Summary

**Current Status:**
- ✅ Cluster `llmd-cluster` is running with 3 nodes (1 server + 2 agents)
- ✅ All pods are deployed in `llm-d-multi-model` namespace
- ✅ **Correct Architecture**: 1 InferencePool Gateway + 1 InferencePool EPP + 2 ModelService pods + 1 Custom API Gateway
- ✅ First model (llama-32-1b) is running (2/2 Ready) and **accepting curl requests** ✓
- ✅ Second model (qwen2-5-0-5b) is running (2/2 Ready) and **accepting curl requests** ✓
- ✅ InferencePool Gateway and HTTPRoute are configured
- ✅ InferencePool automatically discovers ModelService instances via label selector `llm-d.ai/inferenceServing=true`
- ✅ Custom API Gateway automatically discovers and routes through InferencePool Gateway services
- ✅ Models accessible: `/raid/models` (host) → `/models` (k3d container) → `/models` (pod via hostPath)
- ✅ **API Testing**: Both models successfully accept curl requests via port-forward through Custom API Gateway
- ✅ **InferencePool Integration**: All requests benefit from InferencePool's intelligent scheduling, load balancing, and prefix-cache awareness
- ✅ **Unified Interface**: Both InferencePool Gateway and Custom API Gateway provide unified interfaces
  - InferencePool Gateway: Basic unified interface (routes by `model` field only) - Access via port-forward to `infra-llama-32-1b-inference-gateway-istio` service on port 8080
  - Custom API Gateway: Advanced unified interface (routes by `model` + `owned_by` fields) - Access via port-forward to `llmd-api-gateway` service on port 8000
  - **Important**: Custom API Gateway routes through InferencePool Gateway, ensuring all requests benefit from intelligent scheduling
- ✅ **Custom API Gateway**: Successfully deployed and tested, returns both models with complete metadata (`created` timestamp and `max_model_len`)
- ✅ **InferencePool Integration**: Custom API Gateway correctly uses InferencePool Gateway services, preserving all llm-d intelligent scheduling features

**Model Storage Configuration:**
- **Host path**: `/raid/models` (where models are stored on host machine)
- **k3d mount**: `/raid/models:/models` (mounted when creating cluster with `--volume /raid/models:/models`)
- **Pod access**: Use HuggingFace path `hf://Qwen/Qwen2.5-0.5B-Instruct` or local path `/models/<model-name>` for local models
- **Volume config**: Add `hostPath` volume in `decode.volumes` section of values.yaml to mount `/models` from k3d container
- **Volume mount**: Set `readOnly: false` to allow vLLM to write cache files to `/models/hub`

**Testing the API:**
- ✅ Both models (llama-32-1b and qwen2-5-0-5b) successfully accept curl requests
- ✅ Tested endpoints: `/v1/models` and `/v1/chat/completions`
- ✅ Custom API Gateway accessible on `http://localhost:8000` (via port-forward)
- ✅ Custom API Gateway `/v1/models` returns both models with complete metadata (`created` and `max_model_len`)
- ✅ Direct pod port-forward works: `kubectl port-forward pod/<pod-name> 8001:8000` (use 8001 to avoid conflict with Gateway)

**Reference:** 
- See `/home/wukong/workspace/coderepo/09/code/k3d/ref.md` for manual deployment examples using kubectl (without llm-d)
- See `/home/wukong/workspace/coderepo/09/code/vllm/` for manual vLLM deployment examples
- See `/home/wukong/workspace/coderepo/09/code/vllm/llama-3.2-1b.yaml` for reference configurations

**To check pods:**
```bash
export NAMESPACE=llm-d-multi-model
kubectl get pods -n ${NAMESPACE}  # NOT just "kubectl get pods" (which queries default namespace)
```

**To test the API:**

**Option 1: Via Custom API Gateway (Recommended - Unified Interface):**
```bash
export NAMESPACE=llm-d-multi-model

# Port-forward to Custom API Gateway (standard port 8000)
kubectl port-forward -n ${NAMESPACE} svc/llmd-api-gateway 8000:8000

# In another terminal, test unified interface
# List all models (returns both models with complete metadata)
curl http://localhost:8000/v1/models | jq

# Test chat completion
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.2-1B-Instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'
```

**Option 2: Direct Pod Port-Forward (For Direct Testing):**
```bash
export NAMESPACE=llm-d-multi-model
# Get pod name
POD_NAME=$(kubectl get pods -n ${NAMESPACE} -l llm-d.ai/role=decode | grep llama-32-1b | awk '{print $1}')

# Port-forward to pod (use 8001 to avoid conflict with Gateway on 8000)
kubectl port-forward -n ${NAMESPACE} pod/${POD_NAME} 8001:8000

# In another terminal, test
curl -X GET http://localhost:8001/v1/models
curl -X POST http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "meta-llama/Llama-3.2-1B-Instruct", "messages": [{"role": "user", "content": "Hello!"}], "max_tokens": 10}'
```

**Key Fixes Applied:**
1. ✅ Added `runtimeClassName: nvidia` to `extraConfig` for GPU access in k3d
2. ✅ Changed `RELEASE_NAME_POSTFIX` from `llama-3.2-1b` to `llama-32-1b` (no dots allowed in DNS names)
3. ✅ Moved `nodeSelector` to `extraConfig.nodeSelector` (correct ModelService chart schema)
4. ✅ Disabled Prometheus monitoring to avoid ServiceMonitor CRD requirement
5. ✅ Configured `hostPath` volume for local model access (`/raid/models` on host → `/models` in pods)
6. ✅ Set `readOnly: false` on volume mount to allow vLLM cache writes
7. ✅ Cleaned up duplicate Gateway/InferencePool deployments (keep only one shared instance)
8. ✅ Custom API Gateway deployed and configured for `owned_by` routing
9. ✅ Custom API Gateway returns both models with complete metadata (`created` timestamp and `max_model_len`)
10. ✅ Port-forward configured to use standard port 8000 for Custom API Gateway
11. ✅ Custom API Gateway now correctly routes through InferencePool Gateway services (e.g., `infra-llama-32-1b-inference-gateway-istio`) instead of bypassing them
12. ✅ Fixed service discovery to automatically find and use InferencePool Gateway services
13. ✅ Fixed Kubernetes API access issues (clusterIP attribute access)
14. ✅ All requests now benefit from InferencePool's intelligent scheduling, load balancing, and prefix-cache awareness

**Successfully Tested:**
- ✅ Both models (llama-32-1b and qwen2-5-0-5b) accept curl requests
- ✅ Custom API Gateway `/v1/models` endpoint returns both models with complete metadata
- ✅ Custom API Gateway `/v1/chat/completions` endpoint routes correctly to both models through InferencePool Gateway
- ✅ Custom API Gateway automatically discovers and uses InferencePool Gateway services
- ✅ All requests benefit from InferencePool's intelligent scheduling and load balancing
- ✅ Direct pod port-forward works: `kubectl port-forward pod/<pod-name> 8001:8000`
- ✅ Custom API Gateway accessible on `http://localhost:8000` (standard port)

### Comparison: k3d vs llm-d

| Feature | k3d (Manual) | llm-d (Production) |
|---------|--------------|---------------------|
| **Setup Complexity** | Manual YAML files | Helm charts (automated) |
| **Unified Interface** | ✅ Custom API Gateway | ✅ InferencePool Gateway + Optional Custom Gateway |
| **Routing** | Custom API Gateway | Inference Gateway (K8s native) |
| **Load Balancing** | Basic round-robin | Intelligent (prefix-cache aware) |
| **Monitoring** | Manual setup | Built-in Prometheus/Grafana |
| **Scaling** | Manual pod management | HPA-ready, autoscaling support |
| **Multi-Model** | Manual service mapping | Automatic discovery |
| **Production Features** | Limited | Full production stack |
| **owned_by Routing** | ✅ Supported | ✅ Supported (via Custom Gateway) |


