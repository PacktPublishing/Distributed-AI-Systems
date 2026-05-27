# llm-d 支持的推理引擎

## 当前支持的推理引擎

### 1. vLLM ✅
- **状态**: 完全支持（默认推理引擎）
- **镜像**: `ghcr.io/llm-d/llm-d-cuda:v0.4.0` 或 `vllm/vllm-openai:v0.12.0`
- **modelCommand**: `vllmServe`
- **说明**: llm-d 的主要推理引擎，所有 well-lit paths 都围绕 vLLM 设计

### 2. SGLang 🔄
- **状态**: 正在开发中
- **GitHub Issue**: [llm-d/llm-d#403](https://github.com/llm-d/llm-d/issues/403)
- **镜像**: `lmsysorg/sglang:v0.5.6.post2-runtime`（当前使用官方镜像）
- **modelCommand**: `sglangServe`（计划中，但当前 chart 不支持）
- **当前方案**: 使用 `custom` 模式 + 官方镜像
- **进展**: PR #527 正在添加 SGLang 支持

## 不支持的推理引擎

### TensorRT / TensorRT-LLM ❌
- **状态**: 未找到支持
- **搜索范围**: 
  - llm-d 主仓库
  - 文档和提案
  - GitHub issues
- **结果**: 没有找到任何关于 TensorRT 或 TensorRT-LLM 的引用或支持

## llm-d 的设计原则

根据 `docs/proposals/modelservice.md`：

> **Prioritize non-vLLM serving engines (initially):** llm-d follows a **vLLM-first but not vLLM-only** design principle. ModelService follows the same.

这意味着：
- llm-d 采用 "vLLM-first but not vLLM-only" 的设计原则
- 未来可能会支持其他推理引擎
- 但目前主要围绕 vLLM 构建

## 架构说明

根据 `README.md`：

> llm-d accelerates distributed inference by integrating industry-standard open technologies: **vLLM as default model server and engine**, Inference Gateway as request scheduler and balancer, and Kubernetes as infrastructure orchestrator and workload control plane.

llm-d 的核心架构：
- **默认模型服务器**: vLLM
- **请求调度器**: Inference Gateway
- **基础设施编排**: Kubernetes

## 如何添加新的推理引擎支持

如果需要添加 TensorRT-LLM 或其他推理引擎支持，可以参考：

1. **GitHub Issue #403** - SGLang 支持的实现方式
2. **ModelService Helm Chart** - 需要添加新的 `modelCommand` 类型
3. **Inference Gateway Extension** - 可能需要添加相应的支持

## 总结

| 推理引擎 | 状态 | 说明 |
|---------|------|------|
| vLLM | ✅ 完全支持 | 默认推理引擎，所有功能都围绕 vLLM |
| SGLang | 🔄 开发中 | issue #403，PR #527 进行中 |
| TensorRT | ❌ 不支持 | 未找到相关支持 |
| TensorRT-LLM | ❌ 不支持 | 未找到相关支持 |

## 相关资源

- [llm-d README](https://github.com/llm-d/llm-d)
- [ModelService Proposal](https://github.com/llm-d/llm-d/blob/main/docs/proposals/modelservice.md)
- [SGLang Support Issue #403](https://github.com/llm-d/llm-d/issues/403)
- [TensorRT-LLM GitHub](https://github.com/NVIDIA/TensorRT-LLM)
