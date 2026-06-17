# Root Causes of vLLM Triggering disk-pressure in Kubernetes

## 1. vLLM I/O Characteristics

### 1. Image Layer Pressure
- **Image size**: `vllm/vllm-openai:latest` is about 19.5 GB
- **Pull behavior**: the first pull downloads all layers
- **Storage location**: `/var/lib/docker/overlay2` (`nodefs`)
- **Impact**: on a disk-pressure node, the image pull may fail or trigger eviction

### 2. Model Download and Caching
vLLM uses HuggingFace, and by default:

```
/root/.cache/huggingface/
├── hub/                    # model files (GB scale)
├── transformers/           # tokenizer cache
└── datasets/               # dataset cache (if any)
```

**Problems**:
- The default path is inside the container overlayfs -> writes to `nodefs`
- Llama-3.2-1B-Instruct is about 2-3 GB
- Larger models (7B, 13B) can be 10-30 GB

### 3. KV Cache and Temporary Files
At runtime, vLLM generates:
- KV cache (in GPU memory, but swap may exist)
- Temporary token files
- Log files (stderr/stdout)

### 4. Log Writing
- kubelet collects container logs
- Logs are written to `/var/log/pods/` or `/var/lib/containers/`
- Verbose vLLM logs can become very large

## 2. kubelet Eviction Mechanism

### 1. Monitored Paths
kubelet monitors disk usage on these paths:

```
nodefs (root filesystem):
  - /var/lib/docker (overlay2, volumes)
  - /var/lib/containers
  - /var/log/pods
  - /tmp (if used)

imagefs (if separate):
  - /var/lib/docker/images
```

### 2. Default Thresholds
kubelet's default eviction thresholds:

```yaml
evictionHard:
  nodefs.available: "10%"      # or 15%
  imagefs.available: "15%"
  nodefs.inodesFree: "5%"
```

When disk usage exceeds the threshold:
1. The node is marked `DiskPressure=True`
2. New Pods stop being scheduled unless they have a toleration
3. Pod eviction begins based on priority

### 3. Eviction Order
kubelet evicts pods in this order:

1. **BestEffort** Pods (no requests)
2. **Burstable** Pods (requests < limits)
3. **Guaranteed** Pods (requests == limits)

The test-vLLM Pod is **Burstable**, so:
- It can be scheduled if it has a toleration
- But if pressure continues to rise, it may still be evicted

## 3. Why vLLM Is Especially Prone to disk-pressure

### 1. Combined Image and Model Pressure
```
Image pull: 19.5 GB -> nodefs
Model download: 2-3 GB -> nodefs (if cache lives in overlayfs)
Total: about 22 GB written
```

On a node already at 90% usage, this easily triggers the threshold.

### 2. Concentrated Write Bursts
When vLLM starts:
1. Pull the image (if it is not already local)
2. Download the model (if the cache is missing)
3. Load the model into GPU memory
4. Start serving

The first two steps write large amounts to `nodefs` in a short period.

### 3. Cache Strategy Issues
The default HuggingFace cache is under `/root/.cache`:
- It lives inside the container overlayfs
- Writes land on `nodefs`
- Even when the model is already downloaded, startup may still verify or refresh the cache

## 4. Solutions

### Option 1: Use PVC Storage for the Cache (Recommended)
```yaml
volumes:
- name: hf-cache
  persistentVolumeClaim:
    claimName: vllm-cache-pvc
volumeMounts:
- name: hf-cache
  mountPath: /root/.cache/huggingface
```

**Benefits**:
- Cache is stored on a separate partition such as `/raid`
- It does not add pressure to `nodefs`
- It can be shared across Pods

### Option 2: Use hostPath to /raid
```yaml
volumes:
- name: hf-cache
  hostPath:
    path: /raid/tmpdata/vllm-cache
    type: DirectoryOrCreate
```

**Benefits**:
- Simple and direct
- Does not depend on PVCs
- Uses `/raid` directly, which usually has plenty of space

### Option 3: Pre-pull Images and Models
On a node that is not under disk-pressure:
1. Pre-pull the image
2. Pre-download the model into a shared cache
3. Then schedule onto the target node

### Option 4: Adjust kubelet Eviction Thresholds
Modify the k3d cluster configuration to raise the threshold:

```yaml
# Use --k3s-arg when creating the k3d cluster
--k3s-arg '--kubelet-arg=eviction-hard=nodefs.available<5%'
```

**Risk**: This may delay issue detection and lead to a worse disk-full condition

## 5. Best Practices Summary

### For vLLM Serving Pods:

1. **Pin the image version** instead of using `latest`
2. **Use PVC storage for the cache** to avoid `nodefs`
3. **Remove disk-pressure toleration** in production
4. **Set reasonable resource requests** to avoid OOM and eviction
5. **Add health checks** to detect problems early
6. **Use Guaranteed QoS** to lower eviction priority

### For Debug Pods:

1. **Use `emptyDir`** to limit size
2. **Use temporary tolerations** only for debugging
3. **Delete the pod after validation**
4. **Do not use it for real inference**
