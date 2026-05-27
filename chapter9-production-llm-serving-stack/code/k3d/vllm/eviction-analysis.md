# vLLM 在 Kubernetes 中触发 disk-pressure 的深层原因

## 一、vLLM 的 I/O 特征

### 1. 镜像层压力
- **镜像大小**: vllm/vllm-openai:latest 约 19.5GB
- **拉取行为**: 首次拉取需要下载所有 layer
- **存储位置**: `/var/lib/docker/overlay2` (nodefs)
- **影响**: 在 disk-pressure 节点上，image pull 可能失败或触发 eviction

### 2. 模型下载和缓存
vLLM 使用 HuggingFace，默认行为：

```
/root/.cache/huggingface/
├── hub/                    # 模型文件（GB 级别）
├── transformers/           # tokenizer cache
└── datasets/               # 数据集 cache（如果有）
```

**问题**:
- 默认路径在容器 overlayfs 内 → 写入 nodefs
- Llama-3.2-1B-Instruct 约 2-3GB
- 如果使用更大的模型（7B, 13B），可能 10-30GB

### 3. KV Cache 和临时文件
vLLM 运行时会产生：
- KV cache（在 GPU memory，但可能有 swap）
- 临时 token 文件
- 日志文件（stderr/stdout）

### 4. 日志写入
- kubelet 会收集容器日志
- 写入 `/var/log/pods/` 或 `/var/lib/containers/`
- vLLM 的 verbose 日志可能很大

## 二、kubelet eviction 机制

### 1. 监控的路径
kubelet 监控以下路径的磁盘使用：

```
nodefs (根分区):
  - /var/lib/docker (overlay2, volumes)
  - /var/lib/containers
  - /var/log/pods
  - /tmp (如果使用)

imagefs (如果独立):
  - /var/lib/docker/images
```

### 2. 默认阈值
kubelet 默认 eviction 阈值：

```yaml
evictionHard:
  nodefs.available: "10%"      # 或 15%
  imagefs.available: "15%"
  nodefs.inodesFree: "5%"
```

当磁盘使用率超过阈值：
1. 节点标记为 `DiskPressure=True`
2. 停止调度新 Pod（除非有 toleration）
3. 开始驱逐 Pod（按优先级）

### 3. 驱逐顺序
kubelet 按以下顺序驱逐：

1. **BestEffort** Pod（无 requests）
2. **Burstable** Pod（requests < limits）
3. **Guaranteed** Pod（requests == limits）

你的 test-vllm Pod 是 **Burstable**，所以：
- 有 toleration → 可以被调度
- 但一旦压力继续上升 → 仍可能被驱逐

## 三、为什么 vLLM 特别容易触发 disk-pressure

### 1. 镜像 + 模型双重压力
```
镜像拉取: 19.5GB → nodefs
模型下载: 2-3GB → nodefs (如果 cache 在 overlayfs)
总计: ~22GB 写入
```

在 90% 使用率的节点上，这很容易触发阈值。

### 2. 写入时机集中
vLLM 启动时：
1. 拉取镜像（如果本地没有）
2. 下载模型（如果 cache 没有）
3. 加载模型到 GPU memory
4. 开始 serving

前两步都在短时间内大量写入 nodefs。

### 3. 缓存策略问题
默认 HuggingFace cache 在 `/root/.cache`：
- 在容器 overlayfs 内
- 写入会落到 nodefs
- 即使模型已下载，每次启动可能还会验证/更新 cache

## 四、解决方案

### 方案 1: 使用 PVC 存储 cache（推荐）
```yaml
volumes:
- name: hf-cache
  persistentVolumeClaim:
    claimName: vllm-cache-pvc
volumeMounts:
- name: hf-cache
  mountPath: /root/.cache/huggingface
```

**优点**:
- cache 存储在独立分区（如 /raid）
- 不加重 nodefs
- 可以跨 Pod 共享

### 方案 2: 使用 hostPath 到 /raid
```yaml
volumes:
- name: hf-cache
  hostPath:
    path: /raid/tmpdata/vllm-cache
    type: DirectoryOrCreate
```

**优点**:
- 简单直接
- 不依赖 PVC
- 直接使用 /raid（空间充足）

### 方案 3: 预拉取镜像和模型
在非 disk-pressure 节点上：
1. 预拉取镜像
2. 预下载模型到共享 cache
3. 再调度到目标节点

### 方案 4: 调整 kubelet eviction 阈值
修改 k3d 集群配置，提高阈值：

```yaml
# 在 k3d 创建时通过 --k3s-arg
--k3s-arg '--kubelet-arg=eviction-hard=nodefs.available<5%'
```

**风险**: 可能延迟发现问题，导致更严重的磁盘满

## 五、最佳实践总结

### 对于 vLLM Serving Pod：

1. **固定镜像版本**（不用 latest）
2. **使用 PVC 存储 cache**（避免 nodefs）
3. **移除 disk-pressure toleration**（生产环境）
4. **设置合理的资源 requests**（避免 OOM + eviction）
5. **添加健康检查**（及时发现问题）
6. **使用 Guaranteed QoS**（降低被驱逐优先级）

### 对于 Debug Pod：

1. **使用 emptyDir**（限制大小）
2. **临时 toleration**（仅用于调试）
3. **验证后立即删除**
4. **不要用于实际 inference**
