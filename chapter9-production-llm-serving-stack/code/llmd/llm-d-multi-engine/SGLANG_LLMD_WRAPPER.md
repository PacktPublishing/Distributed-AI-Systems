# Discovery: SGLang Using the llm-d Wrapper Image

## Discovery

An example using `sglangServe` was found in `${LLMD_HOME}/guides/inference-scheduling/ms-inference-scheduling/values.yaml`:

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

## Current Status

### Result of Trying `sglangServe`

When attempting to deploy with `sglangServe`, the following error occurred:

```
Error: execution error at (llm-d-modelservice/templates/decode-deployment.yaml:43:13):
.container.modelCommand is not as expected.
Valid values are `vllmServe`, `imageDefault` and `custom`.
```

### Analysis

1. **The llm-d example files** show how to use `sglangServe`
2. **The currently deployed Helm chart version (v0.3.8)** does not support `sglangServe`
3. **Chart template validation** only allows `vllmServe`, `imageDefault`, and `custom`

## Possible Reasons

Based on GitHub issue #403 ([llm-d/llm-d#403](https://github.com/llm-d/llm-d/issues/403)):

1. **The feature is still under development**: this EPIC issue was created on October 29, 2025, specifically to track SGLang support work
2. **Multi-repo collaboration**: changes are needed across multiple llm-d repositories:
   - llm-d/llm-d (main repository)
   - llm-d-inference-scheduler (inference scheduler)
   - gateway-api-inference-extension (needs basic support)
3. **Work in progress**:
   - PR #527 "Add SGLang option for inference-scheduling well-lit path" (December 3, 2025)
   - Multiple sub-tasks are under development (#519, #520, #521)
4. **The example files are forward-looking**: they show how the future feature will be used, but the current chart version has not implemented it yet

## Current Workaround

Because the chart does not support `sglangServe`, the current deployment must use:

```yaml
containers:
- name: "sglang"
  image: lmsysorg/sglang:v0.5.6.post2-runtime  # use the official image
  modelCommand: custom  # use custom mode
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

## Future Direction

If the chart supports `sglangServe` in the future, it can be configured like this:

```yaml
containers:
- name: "sglang"
  image: ghcr.io/llm-d/llm-d-cuda:v0.4.0
  modelCommand: sglangServe  # if supported
  args:
    - "--disable-uvicorn-access-log"
    - "--mem-fraction-static"
    - "0.2"
    - "--model-path"
    - Qwen/Qwen2.5-0.5B-Instruct
    - "--port"
    - "8200"
```

## Related Resources

- **GitHub Issue**: [llm-d/llm-d#403 - [EPIC] Support sglang](https://github.com/llm-d/llm-d/issues/403)
- **Related PR**: [PR #527 - Add SGLang option for inference-scheduling well-lit path](https://github.com/llm-d/llm-d/pull/527)
- **Inference Scheduler Issue**: [llm-d-inference-scheduler#394](https://github.com/llm-d/llm-d-inference-scheduler/issues/394)
- **Gateway API Extension**: [kubernetes-sigs/gateway-api-inference-extension#1141](https://github.com/kubernetes-sigs/gateway-api-inference-extension/issues/1141)

## Summary

- ✅ The llm-d example files show how to use `sglangServe`
- ❌ The currently deployed chart version (v0.3.8) does not support `sglangServe`
- ✅ SGLang still gets some llm-d functionality through the routing-proxy sidecar
- 🔄 **SGLang support is an ongoing EPIC effort** (GitHub issue #403, created on October 29, 2025)
- ⏳ We need to wait for the relevant PRs to merge and for the chart to be updated
