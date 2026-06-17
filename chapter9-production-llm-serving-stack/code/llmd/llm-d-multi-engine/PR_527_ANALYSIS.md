# PR #527 Analysis: Add SGLang option for inference-scheduling well-lit path

## PR Overview

- **PR number**: #527
- **Title**: Add SGLang option for inference-scheduling well-lit path
- **Author**: andreyod
- **Status**: Open (needs review)
- **Related issues**: #519, #403
- **Change summary**: +147 −4 (4 files)

## Purpose

Add SGLang as an inference server option in the "Intelligent Inference Scheduling" well-lit path, as an alternative to vLLM.

## Scope of Changes

### 1. Supported Scope
- **Profile**: "approximate prefix cache aware"
- **Gateway**: default Istio gateway
- **Hardware**: GPU hardware
- **Limitations**: only the above configuration combination is supported for now

### 2. File Changes

#### New Files

1. **`guides/inference-scheduling/ms-inference-scheduling/values_sglang.yaml`**
   - SGLang ModelService configuration values
   - Uses the `docker.io/lmsysorg/sglang:v0.5.5.post1` image
   - Configures a routing-proxy sidecar, with the connector set to `sglang`
   - Port: 8200 (to avoid conflicting with the routing-proxy on port 8000)
   - Model: Qwen/Qwen3-0.6B

2. **`guides/inference-scheduling/gaie-inference-scheduling/values_sglang.yaml`**
   - SGLang configuration for GAIE (Gateway API Inference Extension)

#### Modified Files

1. **`guides/inference-scheduling/helmfile.yaml.gotmpl`**
   - Added a `sglang: &SGL` environment configuration
   - Supports selecting the SGLang environment with `-e sglang`

2. **`guides/inference-scheduling/README.md`**
   - Added an "Inference Server Selection" section
   - Explains how to use SGLang:
     ```bash
     helmfile apply -e sglang -n ${NAMESPACE}
     ```

## Key Configuration Details

### SGLang Configuration Characteristics

1. **Image**: `docker.io/lmsysorg/sglang:v0.5.5.post1` (official image)
2. **Port**: 8200 (to avoid conflicting with the routing-proxy on port 8000)
3. **Connector**: `sglang` (uses the dedicated SGLang connector)
4. **Command**: `python3 -m sglang.launch_server`
5. **Health checks**:
   - Startup: `/v1/models` on port 8200
   - Liveness: `/health` on port 8200
   - Readiness: `/v1/models` on port 8200

### Comparison with the Current Deployment

| Feature | PR #527 | Current deployment (llm-d-multi-engine) |
|------|---------|-------------------------------|
| Image | `lmsysorg/sglang:v0.5.5.post1` | `lmsysorg/sglang:v0.5.6.post2-runtime` |
| Port | 8200 | 8200 |
| Connector | `sglang` | `nixlv2` |
| modelCommand | `custom` (manual command) | `custom` |
| Model | Qwen/Qwen3-0.6B | Qwen/Qwen2.5-0.5B-Instruct |

## Review Feedback

### 1. Comment from liu-cong
- **Concern**: SGLang is not an environment; it is a component of the environment
- **Suggestion**: Consider a better organization, perhaps without automation, and instead add a tab in the user guide so users can choose

### 2. Reply from ezrasilvera
- **View**: Agreed it may not be the best approach, but it should still be integrated into automation
- **Reasoning**:
  - There will be regression tests in the future
  - SGLang should be treated as a first-class citizen alongside vLLM
  - Automatic validation is needed to ensure the feature is not broken

### 3. Suggestion from hhk7734
- **Issue**: the routing-proxy connector should use `sglang` rather than `nixlv2`
- **Reference**: [connector_sglang.go](https://github.com/llm-d/llm-d-inference-scheduler/blob/main/pkg/sidecar/proxy/connector_sglang.go)
- **Status**: ✅ Fixed (the author updated it to `connector: sglang`)

## Relationship to Issue #403

This PR is part of [Issue #403](https://github.com/llm-d/llm-d/issues/403) (EPIC: Support sglang), specifically:
- Issue #519: Sglang support for well-lit path of approximate prefix cache aware scorer

## Future Plans

According to the PR description, future work will also explore:
- P/D disaggregation scenarios
- Precise prefix scenarios

These are tracked in the Issue #403 EPIC.

## Evaluation

### Strengths ✅

1. **Minimal code change**: adds a new environment configuration rather than refactoring the whole framework
2. **Clear documentation**: README explains how to use it
3. **Correct connector**: uses the dedicated SGLang connector
4. **Complete configuration**: includes health checks, monitoring, and other production details

### Limitations ⚠️

1. **Limited scope**: only supports a specific configuration combination (Istio + GPU + approximate prefix cache aware)
2. **Not a first-class citizen**: selected through environment variables rather than a `modelCommand` option
3. **Version mismatch**: the SGLang version used (`v0.5.5.post1`) is different from the current deployment

### Impact on the Current Deployment

This PR **does not directly affect** the current `llm-d-multi-engine` deployment because:
1. It only affects the `guides/inference-scheduling` path
2. The current deployment uses the ModelService Helm chart, not this well-lit path
3. It is still a useful reference for how to configure SGLang correctly

## Recommendations

1. **Wait for the PR to merge**: this PR is still under review, so wait for it to merge before adopting it
2. **Watch the connector**: confirm whether the current deployment should use the `sglang` connector instead of `nixlv2`
3. **Align versions**: consider whether to use the SGLang version from the PR

## Summary

PR #527 is a **gradual improvement** that lays the groundwork for SGLang support. Although the environment-variable approach may not be the cleanest, it is a **minimally invasive** implementation that fits the current architecture.

This PR shows that:
- ✅ SGLang support is actively being developed
- ✅ A dedicated SGLang connector is available
- ✅ The community is working to integrate SGLang as a first-class citizen

For the current deployment, this PR is mainly a **reference point** showing how to configure SGLang integration with llm-d correctly.
