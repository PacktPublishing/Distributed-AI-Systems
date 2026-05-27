# SGLang 使用 llm-d 包装镜像的发现

## 发现

在 `${LLMD_HOME}/guides/inference-scheduling/ms-inference-scheduling/values.yaml` 中找到了使用 `sglangServe` 的示例：

```yaml
containers:
- name: "sglang"
  image: ghcr.io/llm-d/llm-d-cuda:v0.4.0  # Use llm-d wrapper image (same as vLLM)
  modelCommand: sglangServe  # Use llm-d's sglangServe command
  args:
    - "--disable-uvicorn-access-log"
    - "--mem-fraction-static"
    - "0.2"
    - "--model-path"
    - Qwen/Qwen2.5-0.5B-Instruct
    - "--port"
    - "8200"
```

## 当前状态

### 尝试使用 sglangServe 的结果

当尝试使用 `sglangServe` 部署时，遇到了以下错误：

```
Error: execution error at (llm-d-modelservice/templates/decode-deployment.yaml:43:13): 
.container.modelCommand is not as expected. 
Valid values are `vllmServe`, `imageDefault` and `custom`.
```

### 分析

1. **llm-d 示例文件**展示了 `sglangServe` 的用法
2. **当前部署的 Helm chart 版本 (v0.3.8)** 不支持 `sglangServe`
3. **Chart 模板验证**只允许：`vllmServe`, `imageDefault`, `custom`

## 可能的原因

根据 GitHub issue #403 ([llm-d/llm-d#403](https://github.com/llm-d/llm-d/issues/403))：

1. **功能正在开发中**：这是一个 EPIC issue，创建于 2025年10月29日，专门用于跟踪 SGLang 支持的工作
2. **多仓库协作**：需要在多个 llm-d repos 中进行更改：
   - llm-d/llm-d (主仓库)
   - llm-d-inference-scheduler (推理调度器)
   - gateway-api-inference-extension (需要基本支持)
3. **正在进行的工作**：
   - PR #527 "Add SGLang option for inference-scheduling well-lit path" (2025年12月3日)
   - 多个子任务正在开发中（#519, #520, #521）
4. **示例文件是前瞻性的**：示例文件展示了未来功能的使用方式，但当前 chart 版本尚未实现

## 当前解决方案

由于 chart 不支持 `sglangServe`，当前必须使用：

```yaml
containers:
- name: "sglang"
  image: lmsysorg/sglang:v0.5.6.post2-runtime  # 使用官方镜像
  modelCommand: custom  # 使用 custom 模式
  command:
    - python3
    - -m
    - sglang.launch_server
  args:
    - --model-path
    - Qwen/Qwen2.5-0.5B-Instruct
    - --port
    - "8200"
    - --mem-fraction-static
    - "0.2"
```

## 未来方向

如果未来 chart 支持 `sglangServe`，可以这样配置：

```yaml
containers:
- name: "sglang"
  image: ghcr.io/llm-d/llm-d-cuda:v0.4.0
  modelCommand: sglangServe  # 如果支持的话
  args:
    - "--disable-uvicorn-access-log"
    - "--mem-fraction-static"
    - "0.2"
    - "--model-path"
    - Qwen/Qwen2.5-0.5B-Instruct
    - "--port"
    - "8200"
```

## 相关资源

- **GitHub Issue**: [llm-d/llm-d#403 - [EPIC] Support sglang](https://github.com/llm-d/llm-d/issues/403)
- **相关 PR**: [PR #527 - Add SGLang option for inference-scheduling well-lit path](https://github.com/llm-d/llm-d/pull/527)
- **Inference Scheduler Issue**: [llm-d-inference-scheduler#394](https://github.com/llm-d/llm-d-inference-scheduler/issues/394)
- **Gateway API Extension**: [kubernetes-sigs/gateway-api-inference-extension#1141](https://github.com/kubernetes-sigs/gateway-api-inference-extension/issues/1141)

## 总结

- ✅ llm-d 示例文件中展示了 `sglangServe` 的用法
- ❌ 当前部署的 chart 版本 (v0.3.8) 不支持 `sglangServe`
- ✅ SGLang 仍然通过 routing-proxy sidecar 获得 llm-d 的部分功能
- 🔄 **SGLang 支持是一个正在进行的 EPIC 工作**（GitHub issue #403，创建于 2025年10月29日）
- ⏳ 需要等待相关 PR 合并和 chart 更新
