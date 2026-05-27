# llm-d 应用情况说明

## 当前状态

### vLLM
- ✅ **完全使用 llm-d**
  - 镜像: `ghcr.io/llm-d/llm-d-cuda:v0.4.0` (llm-d 包装镜像)
  - modelCommand: `vllmServe` (llm-d 支持的命令)
  - routing-proxy sidecar: ✅ 有
  - 通过 ModelService Helm chart 部署

### SGLang
- ⚠️ **部分使用 llm-d**
  - 镜像: `lmsysorg/sglang:v0.5.6.post2-runtime` (SGLang 官方镜像)
  - modelCommand: `custom` (因为 ModelService chart 不支持 `sglangServe`)
  - routing-proxy sidecar: ✅ 有
  - 通过 ModelService Helm chart 部署

## 原因分析

### ModelService Helm Chart 限制
ModelService Helm chart (`llm-d-modelservice/llm-d-modelservice`) 的 `modelCommand` 字段只支持：
- `vllmServe` - 用于 vLLM
- `imageDefault` - 使用镜像默认命令
- `custom` - 自定义命令

**当前部署的 chart 版本 (v0.3.8) 不支持 `sglangServe`**，所以 SGLang 必须使用 `custom` 模式，并直接使用 SGLang 官方镜像。

### 重要发现
虽然在 `${LLMD_HOME}/guides/inference-scheduling/ms-inference-scheduling/values.yaml` 中找到了 `sglangServe` 的示例配置：
```yaml
containers:
- name: "sglang"
  image: ghcr.io/llm-d/llm-d-cuda:v0.4.0
  modelCommand: sglangServe  # Use llm-d's sglangServe command
```

但当前部署的 Helm chart 版本还不支持这个功能。这可能意味着：
- `sglangServe` 是计划中的功能，但尚未在 chart 中实现
- 或者需要更新到更新的 chart 版本
- 或者这个功能还在开发中

### SGLang 仍然获得的部分 llm-d 功能
虽然 SGLang 主容器没有使用 llm-d 包装镜像，但它仍然通过以下方式获得了 llm-d 的部分功能：

1. **routing-proxy sidecar**
   - 提供智能路由和负载均衡
   - 支持 prefix-cache aware routing
   - 提供统一的 API 接口

2. **ModelService 管理**
   - 通过 llm-d ModelService Helm chart 部署
   - 获得 llm-d 的 Pod 管理、健康检查等功能

3. **InferencePool 集成**
   - 可以集成到 llm-d 的 InferencePool 中
   - 通过 InferencePool Gateway 访问

## 如何让 SGLang 完全使用 llm-d？

### 选项 1: 使用 LLMInferenceService CRD（如果支持）
如果 llm-d 的 LLMInferenceService CRD 支持 SGLang，可以使用 CRD 方式部署：
```yaml
apiVersion: llm-d.ai/v1alpha1
kind: LLMInferenceService
spec:
  inferenceServer:
    type: sglang
    image: lmsysorg/sglang:v0.5.6.post2-runtime
```

### 选项 2: 等待 ModelService Chart 支持 sglangServe
如果未来 ModelService Helm chart 添加了 `sglangServe` 支持，可以：
```yaml
containers:
- name: "sglang"
  image: ghcr.io/llm-d/llm-d-cuda:v0.4.0
  modelCommand: sglangServe  # 如果支持的话
```

### 选项 3: 使用 llm-d 包装镜像 + custom 命令
可以尝试使用 llm-d 包装镜像，但需要手动指定 SGLang 命令：
```yaml
containers:
- name: "sglang"
  image: ghcr.io/llm-d/llm-d-cuda:v0.4.0
  modelCommand: custom
  command:
    - python3
    - -m
    - sglang.launch_server
```

## 当前部署的影响

### 功能差异
- **vLLM**: 完全使用 llm-d 的所有功能（包装镜像 + routing-proxy）
- **SGLang**: 使用 llm-d 的部分功能（routing-proxy + ModelService 管理），但主容器使用官方镜像

### 性能影响
- 两者都能正常工作
- SGLang 可能无法使用 llm-d 包装镜像中的某些优化
- 但 routing-proxy 提供的智能路由等功能仍然可用

## 总结

**是的，llm-d 主要应用于 vLLM，对 SGLang 的应用是部分的。**

- vLLM: 完全使用 llm-d（包装镜像 + routing-proxy）
- SGLang: 部分使用 llm-d（routing-proxy + ModelService 管理，但主容器使用官方镜像）

这是由于 ModelService Helm chart 的限制导致的。如果需要完全使用 llm-d，可以考虑使用 LLMInferenceService CRD 方式部署（如果支持的话）。
