# InferencePool Gateway 不工作的原因分析

## 问题

InferencePool Gateway 返回 404 "route_not_found"，无法路由请求到 ModelService pods。

## 根本原因

InferencePool Gateway 需要两个关键资源才能工作：

1. **InferencePool CRD 实例** - 定义哪些 ModelService pods 属于这个 pool
2. **HTTPRoute 资源** - 配置如何路由请求到 InferencePool

当前部署中：
- ✅ InferencePool Gateway pods 正在运行
- ✅ Gateway API 资源已创建
- ✅ ModelService pods 有正确的标签 (`llm-d.ai/inferenceServing=true`)
- ❌ **但是没有 InferencePool CRD 实例**
- ❌ **没有 HTTPRoute 资源**

## InferencePool Gateway 的工作原理

```
Client Request
    ↓
InferencePool Gateway (Istio Envoy)
    ↓
HTTPRoute (配置路由规则)
    ↓
InferencePool CRD (定义哪些 pods 属于 pool)
    ↓
ModelService Pods (通过标签匹配)
```

## 为什么 helmfile 没有创建这些资源？

从检查结果看，helmfile 部署了：
- `gaie-vllm-qwen2-5-0-5b` - InferencePool Extension (Endpoint Picker)
- `infra-vllm-qwen2-5-0-5b-inference-gateway-istio` - Gateway

但是：
- InferencePool CRD 实例可能需要在 `gaie-inference-scheduling/values.yaml` 中正确配置
- HTTPRoute 可能需要单独创建或通过 helmfile 模板生成

## 解决方案

### 方案 1: 使用 Custom API Gateway (当前推荐)

Custom API Gateway 已经实现了直接 pod 访问和 fallback 机制，不依赖 InferencePool Gateway 的自动发现。

### 方案 2: 修复 InferencePool Gateway (需要额外配置)

需要：
1. 创建 InferencePool CRD 实例，匹配 ModelService pods
2. 创建 HTTPRoute 资源，将请求路由到 InferencePool
3. 确保 InferencePool Extension 正确发现 ModelService pods

## 为什么这是 llm-d 的优点？

InferencePool Gateway 是 llm-d 的核心优势，提供：
- ✅ 智能路由和负载均衡
- ✅ Prefix-cache 感知
- ✅ 自动服务发现
- ✅ 生产级可靠性

但是需要正确配置 InferencePool 和 HTTPRoute 资源才能工作。
