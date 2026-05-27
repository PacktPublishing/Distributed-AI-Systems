# 路由和对比测试指南

## 1. 资源竞争和 Race Condition 分析

### ✅ 当前配置的资源隔离情况

**GPU 隔离（完全隔离）**：
- vLLM: `NVIDIA_VISIBLE_DEVICES=0` → 使用 GPU 0
- SGLang: `NVIDIA_VISIBLE_DEVICES=1` → 使用 GPU 1
- ✅ **无 GPU 竞争**：每个服务使用独立的 GPU

**内存隔离（部分隔离）**：
- vLLM: `--gpu-memory-utilization=0.1` → 使用 GPU 0 的 10% 内存
- SGLang: `--mem-fraction-static=0.1` → 使用 GPU 1 的 10% 内存
- ✅ **GPU 内存完全隔离**：不同 GPU，无竞争
- ⚠️ **系统内存共享**：两个 pod 共享节点的系统内存（但每个 pod 有 limits: 8Gi）

**CPU 资源（共享但有 limits）**：
- ⚠️ **CPU 共享**：两个 pod 共享节点的 CPU
- ✅ **有资源限制**：每个 pod 有 CPU requests/limits（如果设置了）
- 💡 **建议**：如果发现 CPU 竞争，可以设置 CPU limits

**磁盘 I/O（共享）**：
- ⚠️ **模型存储共享**：两个 pod 都访问 `/models` 目录
- ✅ **只读访问**：模型文件是只读的，不会有写冲突
- ⚠️ **缓存写入**：HuggingFace 缓存写入到 `/models/hub`，可能有轻微竞争

**网络带宽（共享）**：
- ⚠️ **共享网络**：两个服务共享节点的网络带宽
- 💡 **影响较小**：对于推理服务，网络带宽通常不是瓶颈

### Race Condition 分析

**❌ 不会有 Race Condition**：
- 两个 pod 是**独立的进程**，不共享内存空间
- 每个服务运行在自己的容器中，有独立的进程空间
- 模型文件是**只读的**，不会有写冲突
- GPU 内存完全隔离，不会有内存竞争

**可能的资源竞争点**：
1. **CPU 竞争**：如果两个服务同时高负载，可能竞争 CPU
2. **系统内存竞争**：如果两个服务同时加载大量数据到系统内存
3. **磁盘 I/O 竞争**：如果同时读取模型文件（但模型通常已加载到 GPU 内存）

### 对比测试的公平性

**✅ 当前配置适合对比测试**：
- 相同的硬件环境（同一节点）
- GPU 完全隔离（不同 GPU）
- 相同的模型（Qwen2.5-0.5B-Instruct）
- 相同的 GPU 内存使用率（都是 10%）

**⚠️ 需要注意的变量**：
- CPU 竞争可能影响结果（但可以通过 CPU limits 控制）
- 系统内存竞争（通常影响较小）
- 网络延迟（同一节点，影响很小）

## 2. 路由方案

### 方案 1: 直接通过 Service 访问（最简单）

每个 LLMInferenceService 会自动创建 Kubernetes Service：

```bash
# 访问 vLLM
kubectl port-forward svc/vllm-qwen2-5-0-5b 8001:8000
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [{"role": "user", "content": "Hello"}]
  }'

# 访问 SGLang
kubectl port-forward svc/sglang-qwen2-5-0-5b 8002:8000
curl http://localhost:8002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-0.5B-Instruct",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

### 方案 2: 通过 API Gateway 路由（推荐用于对比测试）

创建一个 API Gateway，根据请求中的 `inference_server` 字段路由到不同的服务。

**优点**：
- 统一的入口点
- 可以通过请求参数选择引擎
- 方便进行 A/B 测试

**实现方式**：见下面的 `api-gateway.yaml`

### 方案 3: 通过 llm-d InferencePool（如果使用 llm-d 完整功能）

如果部署了 llm-d 的 InferencePool，可以通过 `owned_by` 标签进行路由。

## 3. 对比测试脚本示例

```bash
#!/bin/bash
# benchmark-comparison.sh

# 测试参数
MODEL="Qwen/Qwen2.5-0.5B-Instruct"
NUM_REQUESTS=100
CONCURRENT=10

# 测试 vLLM
echo "Testing vLLM..."
kubectl port-forward svc/vllm-qwen2-5-0-5b 8001:8000 &
VLLM_PF=$!
sleep 2

time for i in $(seq 1 $NUM_REQUESTS); do
  curl -s http://localhost:8001/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"$MODEL\", \"messages\": [{\"role\": \"user\", \"content\": \"Test $i\"}]}" \
    > /dev/null &
  if [ $((i % $CONCURRENT)) -eq 0 ]; then
    wait
  fi
done
wait

kill $VLLM_PF

# 测试 SGLang
echo "Testing SGLang..."
kubectl port-forward svc/sglang-qwen2-5-0-5b 8002:8000 &
SGLANG_PF=$!
sleep 2

time for i in $(seq 1 $NUM_REQUESTS); do
  curl -s http://localhost:8002/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\": \"$MODEL\", \"messages\": [{\"role\": \"user\", \"content\": \"Test $i\"}]}" \
    > /dev/null &
  if [ $((i % $CONCURRENT)) -eq 0 ]; then
    wait
  fi
done
wait

kill $SGLANG_PF
```

## 4. 优化建议

### 如果发现 CPU 竞争

在 LLMInferenceService 中添加 CPU limits：

```yaml
resources:
  requests:
    nvidia.com/gpu: 1
    memory: 6Gi
    cpu: "2"  # 添加 CPU request
  limits:
    nvidia.com/gpu: 1
    memory: 8Gi
    cpu: "4"  # 添加 CPU limit
```

### 如果发现系统内存竞争

增加内存 limits 或减少并发请求数。

### 监控资源使用

```bash
# 监控节点资源
kubectl top node

# 监控 pod 资源
kubectl top pod -l app=vllm
kubectl top pod -l app=sglang

# 监控 GPU 使用
nvidia-smi
```
