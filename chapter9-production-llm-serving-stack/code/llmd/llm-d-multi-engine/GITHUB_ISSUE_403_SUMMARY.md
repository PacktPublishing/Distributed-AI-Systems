# GitHub Issue #403 总结

## Issue 信息

- **标题**: [EPIC] Support sglang
- **链接**: https://github.com/llm-d/llm-d/issues/403
- **创建时间**: 2025年10月29日
- **状态**: Open
- **Assignee**: ezrasilvera

## 描述

这是一个 EPIC issue，用于跟踪所有 llm-d/llm-d 相关的 SGLang 支持任务，同时也是所有其他 llm-d repos 所需更改的占位符。

## 任务清单

### 1. Inference Scheduler 支持
- [ ] [EPIC] Support sglang in the inference scheduler
  - 相关 issue: [llm-d-inference-scheduler#394](https://github.com/llm-d/llm-d-inference-scheduler/issues/394)

### 2. Well-lit Path Guides 支持
- [ ] Support sglang in all well-lit path guides
  - [ ] [Feat] Sglang support for well-lit path of approximate prefix cache aware scorer
    - 相关 issue: [llm-d/llm-d#519](https://github.com/llm-d/llm-d/issues/519)
  - [ ] [Feat] Sglang support for well-lit path of precise prefix cache aware scorer
    - 相关 issue: [llm-d/llm-d#520](https://github.com/llm-d/llm-d/issues/520)
  - [ ] [Feat] Sglang support for well-lit path of Prefill/Decode Disaggregation
    - 相关 issue: [llm-d/llm-d#521](https://github.com/llm-d/llm-d/issues/521)

## 相关引用

### Gateway API Extension
需要在 `gateway-api-inference-extension` 中添加基本支持：
- 相关 issue: [kubernetes-sigs/gateway-api-inference-extension#1141](https://github.com/kubernetes-sigs/gateway-api-inference-extension/issues/1141)

## 进展

### 2025年12月3日
- PR #527: "Add SGLang option for inference-scheduling well-lit path"
  - 链接: https://github.com/llm-d/llm-d/pull/527
  - 状态: Open

## 影响

这个 EPIC issue 证实了：

1. **SGLang 支持是计划中的功能**，但仍在开发中
2. **需要多个仓库的协作**才能完全支持 SGLang
3. **示例文件中的 `sglangServe`** 是前瞻性的，展示了未来功能的使用方式
4. **当前 ModelService Helm chart 不支持 `sglangServe`** 是因为功能尚未完全实现

## 当前状态

- ✅ llm-d 示例文件展示了 `sglangServe` 的用法
- ❌ ModelService Helm chart (v0.3.8) 不支持 `sglangServe`
- 🔄 相关工作正在进行中（PR #527 等）
- ⏳ 需要等待相关 PR 合并和 chart 更新

## 建议

1. **监控 PR #527** 的进展，这可能是添加 `sglangServe` 支持的关键 PR
2. **关注 issue #403** 的更新，了解整体进展
3. **当前使用 `custom` 模式**部署 SGLang，通过 routing-proxy sidecar 获得部分 llm-d 功能
4. **等待 chart 更新**后再尝试使用 `sglangServe`
