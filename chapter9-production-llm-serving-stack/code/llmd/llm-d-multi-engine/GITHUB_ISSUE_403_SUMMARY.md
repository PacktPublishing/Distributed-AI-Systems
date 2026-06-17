# GitHub Issue #403 Summary

## Issue Details

- **Title**: [EPIC] Support sglang
- **Link**: https://github.com/llm-d/llm-d/issues/403
- **Created**: October 29, 2025
- **Status**: Open
- **Assignee**: ezrasilvera

## Description

This is an EPIC issue used to track all SGLang support tasks related to llm-d/llm-d, and it also serves as a placeholder for changes needed across the other llm-d repositories.

## Task List

### 1. Inference Scheduler Support
- [ ] [EPIC] Support sglang in the inference scheduler
  - Related issue: [llm-d-inference-scheduler#394](https://github.com/llm-d/llm-d-inference-scheduler/issues/394)

### 2. Well-lit Path Guides Support
- [ ] Support sglang in all well-lit path guides
  - [ ] [Feat] Sglang support for well-lit path of approximate prefix cache aware scorer
    - Related issue: [llm-d/llm-d#519](https://github.com/llm-d/llm-d/issues/519)
  - [ ] [Feat] Sglang support for well-lit path of precise prefix cache aware scorer
    - Related issue: [llm-d/llm-d#520](https://github.com/llm-d/llm-d/issues/520)
  - [ ] [Feat] Sglang support for well-lit path of Prefill/Decode Disaggregation
    - Related issue: [llm-d/llm-d#521](https://github.com/llm-d/llm-d/issues/521)

## Related References

### Gateway API Extension
Basic support needs to be added in `gateway-api-inference-extension`:
- Related issue: [kubernetes-sigs/gateway-api-inference-extension#1141](https://github.com/kubernetes-sigs/gateway-api-inference-extension/issues/1141)

## Progress

### December 3, 2025
- PR #527: "Add SGLang option for inference-scheduling well-lit path"
  - Link: https://github.com/llm-d/llm-d/pull/527
  - Status: Open

## Impact

This EPIC issue confirms that:

1. **SGLang support is a planned feature**, but it is still under development
2. **Multiple repositories must work together** to fully support SGLang
3. **`sglangServe` in the example files** is forward-looking and demonstrates how the future feature will be used
4. **The current ModelService Helm chart does not support `sglangServe`** because the feature is not fully implemented yet

## Current Status

- ✅ The llm-d example files show how to use `sglangServe`
- ❌ The ModelService Helm chart (v0.3.8) does not support `sglangServe`
- 🔄 Related work is in progress (including PR #527)
- ⏳ We need to wait for the relevant PRs to merge and for the chart to be updated

## Recommendations

1. **Watch PR #527** closely, as it may be the key PR for adding `sglangServe` support
2. **Follow issue #403** for updates on the overall progress
3. **Use `custom` mode for SGLang today**, and rely on the routing-proxy sidecar to get some llm-d functionality
4. **Wait for the chart update** before trying `sglangServe`
