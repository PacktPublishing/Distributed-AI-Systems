# Inference Engines Supported by llm-d

## Currently Supported Inference Engines

### 1. vLLM ✅
- **Status**: Fully supported (default inference engine)
- **Image**: `ghcr.io/llm-d/llm-d-cuda:v0.4.0` or `vllm/vllm-openai:v0.12.0`
- **modelCommand**: `vllmServe`
- **Notes**: llm-d's main inference engine; all well-lit paths are designed around vLLM

### 2. SGLang 🔄
- **Status**: Under development
- **GitHub Issue**: [llm-d/llm-d#403](https://github.com/llm-d/llm-d/issues/403)
- **Image**: `lmsysorg/sglang:v0.5.6.post2-runtime` (currently using the official image)
- **modelCommand**: `sglangServe` (planned, but the current chart does not support it)
- **Current approach**: use `custom` mode + official image
- **Progress**: PR #527 is adding SGLang support

## Unsupported Inference Engines

### TensorRT / TensorRT-LLM ❌
- **Status**: No support found
- **Search scope**:
  - llm-d main repository
  - documentation and proposals
  - GitHub issues
- **Result**: No references or support for TensorRT or TensorRT-LLM were found

## llm-d Design Principles

According to `docs/proposals/modelservice.md`:

> **Prioritize non-vLLM serving engines (initially):** llm-d follows a **vLLM-first but not vLLM-only** design principle. ModelService follows the same.

This means:
- llm-d follows a "vLLM-first but not vLLM-only" design principle
- Other inference engines may be supported in the future
- But the project is currently built primarily around vLLM

## Architecture Overview

According to `README.md`:

> llm-d accelerates distributed inference by integrating industry-standard open technologies: **vLLM as default model server and engine**, Inference Gateway as request scheduler and balancer, and Kubernetes as infrastructure orchestrator and workload control plane.

llm-d's core architecture:
- **Default model server**: vLLM
- **Request scheduler**: Inference Gateway
- **Infrastructure orchestration**: Kubernetes

## How to Add Support for a New Inference Engine

If you need to add TensorRT-LLM or another inference engine, you can refer to:

1. **GitHub Issue #403** - SGLang implementation approach
2. **ModelService Helm Chart** - add a new `modelCommand` type
3. **Inference Gateway Extension** - add the corresponding support

## Summary

| Inference engine | Status | Notes |
|---------|------|------|
| vLLM | ✅ Fully supported | Default inference engine; all features are built around vLLM |
| SGLang | 🔄 In development | issue #403, PR #527 in progress |
| TensorRT | ❌ Not supported | No related support found |
| TensorRT-LLM | ❌ Not supported | No related support found |

## Related Resources

- [llm-d README](https://github.com/llm-d/llm-d)
- [ModelService Proposal](https://github.com/llm-d/llm-d/blob/main/docs/proposals/modelservice.md)
- [SGLang Support Issue #403](https://github.com/llm-d/llm-d/issues/403)
- [TensorRT-LLM GitHub](https://github.com/NVIDIA/TensorRT-LLM)
