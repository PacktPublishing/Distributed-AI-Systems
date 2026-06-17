# llm-d Application Status

## Current Status

### vLLM
- ✅ **Fully uses llm-d**
  - Image: `ghcr.io/llm-d/llm-d-cuda:v0.4.0` (llm-d wrapper image)
  - modelCommand: `vllmServe` (supported llm-d command)
  - routing-proxy sidecar: ✅ present
  - Deployed through the ModelService Helm chart

### SGLang
- ⚠️ **Partially uses llm-d**
  - Image: `lmsysorg/sglang:v0.5.6.post2-runtime` (SGLang official image)
  - modelCommand: `custom` (because the ModelService chart does not support `sglangServe`)
  - routing-proxy sidecar: ✅ present
  - Deployed through the ModelService Helm chart

## Root Cause Analysis

### ModelService Helm Chart Limitation
The `modelCommand` field in the ModelService Helm chart (`llm-d-modelservice/llm-d-modelservice`) only supports:
- `vllmServe` - for vLLM
- `imageDefault` - use the image default command
- `custom` - custom command

**The currently deployed chart version (v0.3.8) does not support `sglangServe`**, so SGLang must use `custom` mode and the official SGLang image directly.

### Important Finding
Although an example `sglangServe` configuration was found in `${LLMD_HOME}/guides/inference-scheduling/ms-inference-scheduling/values.yaml`:
```yaml
containers:
- name: "sglang"
  image: ghcr.io/llm-d/llm-d-cuda:v0.4.0
  modelCommand: sglangServe  # Use llm-d's sglangServe command
```

the currently deployed Helm chart version still does not support this feature. That may mean:
- `sglangServe` is a planned feature, but it has not been implemented in the chart yet
- The deployment needs a newer chart version
- Or the feature is still under development

### llm-d Features SGLang Still Gets
Although the SGLang main container does not use the llm-d wrapper image, it still gets some llm-d functionality through:

1. **routing-proxy sidecar**
   - Provides intelligent routing and load balancing
   - Supports prefix-cache-aware routing
   - Provides a unified API surface

2. **ModelService management**
   - Deployed through the llm-d ModelService Helm chart
   - Gets llm-d Pod management, health checks, and related features

3. **InferencePool integration**
   - Can be integrated into llm-d InferencePools
   - Accessed through the InferencePool Gateway

## How Can SGLang Use llm-d Fully?

### Option 1: Use the LLMInferenceService CRD (if supported)
If llm-d's LLMInferenceService CRD supports SGLang, you can deploy it that way:
```yaml
apiVersion: llm-d.ai/v1alpha1
kind: LLMInferenceService
spec:
  inferenceServer:
    type: sglang
    image: lmsysorg/sglang:v0.5.6.post2-runtime
```

### Option 2: Wait for the ModelService Chart to Support `sglangServe`
If the ModelService Helm chart adds `sglangServe` support in the future, you can:
```yaml
containers:
- name: "sglang"
  image: ghcr.io/llm-d/llm-d-cuda:v0.4.0
  modelCommand: sglangServe  # if supported
```

### Option 3: Use the llm-d Wrapper Image + Custom Command
You can try the llm-d wrapper image, but you must manually specify the SGLang command:
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

## Impact on the Current Deployment

### Feature Differences
- **vLLM**: Fully uses all llm-d features (wrapper image + routing-proxy)
- **SGLang**: Uses some llm-d features (routing-proxy + ModelService management), but the main container uses the official image

### Performance Impact
- Both work correctly
- SGLang may not be able to use some optimizations from the llm-d wrapper image
- The routing-proxy still provides intelligent routing and related features

## Summary

**Yes, llm-d is primarily applied to vLLM, and its application to SGLang is partial.**

- vLLM: fully uses llm-d (wrapper image + routing-proxy)
- SGLang: partially uses llm-d (routing-proxy + ModelService management, but the main container uses the official image)

This is due to limitations in the ModelService Helm chart. If you need full llm-d integration, consider deploying with the LLMInferenceService CRD approach, if supported.
