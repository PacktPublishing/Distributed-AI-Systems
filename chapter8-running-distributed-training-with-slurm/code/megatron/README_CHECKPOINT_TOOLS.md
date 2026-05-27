# Megatron-LM tools/checkpoint/ 目录文件说明

## 概述

`tools/checkpoint/` 目录包含 Megatron-LM 的 checkpoint 转换工具，使用**插件系统**（loader/saver）来实现不同格式之间的转换。

## 核心文件

### 1. `convert.py` - 主转换工具
**作用**：通用的 checkpoint 转换框架

**工作原理**：
- 使用 **loader** 插件加载源格式的 checkpoint
- 使用 **saver** 插件保存为目标格式的 checkpoint
- 通过 multiprocessing.Queue 在 loader 和 saver 之间传递数据

**使用方式**：
```bash
python convert.py \
    --model-type GPT \
    --loader <loader_name> \      # 指定 loader 插件
    --saver <saver_name> \        # 指定 saver 插件
    --load-dir <source_dir> \
    --save-dir <target_dir>
```

**支持的转换方向**：
- HF/Meta → Megatron（有现成的 loader）
- Megatron → Megatron（改变并行度）
- Megatron → HF（需要自定义 saver，目前只有 LLaVA 的）

## Loader 插件（加载器）

### 2. `loader_base.py` - Loader 基类
**作用**：定义所有 loader 的基础接口和通用功能

**关键方法**：
- `load_checkpoint()`: 从队列接收数据的主函数
- `send_metadata_over_queue()`: 发送模型元数据
- `send_llm_over_queue()`: 发送 LLM 模型权重

### 3. `loader_core.py` - Megatron Core 格式加载器
**作用**：加载 Megatron-LM 的 core 格式 checkpoint（torch_dist 格式）

**使用场景**：
- 从 Megatron checkpoint 加载模型
- 改变 tensor/pipeline parallel 大小

**示例**：
```bash
python convert.py \
    --loader core \
    --load-dir checkpoints/gpt_8b/iter_0000010
```

### 4. `loader_legacy.py` - Legacy 格式加载器
**作用**：加载旧版 Megatron 格式的 checkpoint

**使用场景**：
- 迁移旧版 checkpoint 到新格式

### 5. `loader_llama_mistral.py` - Llama/Mistral 加载器
**作用**：从 HuggingFace 或 Meta 格式加载 Llama/Mistral 模型

**支持的格式**：
- HuggingFace 格式（`--checkpoint-type hf`）
- Meta 格式（`--checkpoint-type meta`）

**示例**：
```bash
python convert.py \
    --loader llama_mistral \
    --checkpoint-type hf \
    --load-dir /path/to/hf_checkpoint \
    --model-size llama2-7B
```

### 6. `loader_mixtral_hf.py` - Mixtral HF 加载器
**作用**：从 HuggingFace 格式加载 Mixtral 模型

### 7. `loader_llava.py` - LLaVA 加载器
**作用**：加载 LLaVA 多模态模型的 checkpoint

## Saver 插件（保存器）

### 8. `saver_base.py` - Saver 基类
**作用**：定义所有 saver 的基础接口和通用功能

**关键方法**：
- `save_checkpoint()`: 从队列接收数据并保存的主函数
- `receive_checkpoint_metadata()`: 接收模型元数据
- `receive_lm()`: 接收 LLM 模型权重

### 9. `saver_core.py` - Megatron Core 格式保存器
**作用**：保存为 Megatron-LM 的 core 格式（torch_dist 格式）

**使用场景**：
- 将其他格式转换为 Megatron 格式
- 改变 checkpoint 的并行度配置

**示例**：
```bash
python convert.py \
    --saver core \
    --save-dir checkpoints/gpt_8b_converted \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1
```

### 10. `saver_legacy.py` - Legacy 格式保存器
**作用**：保存为旧版 Megatron 格式

**使用场景**：
- 兼容旧版系统

### 11. `saver_hf_llava.py` - HuggingFace LLaVA 保存器
**作用**：将 LLaVA 模型保存为 HuggingFace 格式

**注意**：这是**唯一**的 HF saver，只支持 LLaVA，不支持通用 GPT 模型

### 12. `saver_llava.py` - LLaVA 保存器
**作用**：保存 LLaVA 模型为 Megatron 格式

## Schema 文件（模式定义）

### 13. `schema_base.py` - Schema 基类
**作用**：定义模型参数的组织方式和访问接口

**功能**：
- 定义如何从模型中提取参数
- 定义如何将参数设置到模型中

### 14. `schema_core.py` - Core Schema
**作用**：定义 Megatron Core 模型的参数结构

**支持的模型类型**：
- GPT
- BERT
- 支持 MoE（Mixture of Experts）

### 15. `schema_hf.py` - HuggingFace Schema
**作用**：定义 HuggingFace 格式的参数结构

**功能**：
- 提供 Megatron → HF 的层名称映射
- 目前主要用于 LLaVA

## 工具文件

### 16. `checkpoint_inspector.py` - Checkpoint 检查器
**作用**：检查和验证 checkpoint 的内容

**功能**：
- 查看 checkpoint 的元数据
- 验证 checkpoint 的完整性
- 转换 checkpoint 格式（torch_dist ↔ fsdp_dtensor）

### 17. `hybrid_conversion.py` - 混合转换
**作用**：处理混合格式的转换（可能用于特殊场景）

### 18. `utils.py` - 工具函数
**作用**：提供转换过程中使用的工具函数

**主要功能**：
- `chunk_weight()`: 分割权重用于 tensor parallel
- `chunk_bias()`: 分割 bias 用于 tensor parallel
- `_ConverterFakeProcessGroup`: 模拟进程组用于转换

## 数据流

```
源 Checkpoint (HF/Meta/Megatron)
    ↓
[Loader 插件]
    ↓ (通过 Queue)
[convert.py]
    ↓ (通过 Queue)
[Saver 插件]
    ↓
目标 Checkpoint (Megatron/HF)
```

## 使用示例

### 示例 1: HF → Megatron
```bash
python convert.py \
    --model-type GPT \
    --loader llama_mistral \
    --saver core \
    --checkpoint-type hf \
    --load-dir /path/to/hf_checkpoint \
    --save-dir /path/to/megatron_checkpoint \
    --model-size llama2-7B \
    --target-tensor-parallel-size 1
```

### 示例 2: Megatron → Megatron (改变并行度)
```bash
python convert.py \
    --model-type GPT \
    --loader core \
    --saver core \
    --load-dir checkpoints/gpt_8b/iter_0000010 \
    --save-dir checkpoints/gpt_8b_tp1 \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1
```

### 示例 3: Megatron → PyTorch（需要自定义 saver）
```bash
# 将 megatron_saver_pytorch.py 复制到 tools/checkpoint/ 目录
cp megatron_saver_pytorch.py /path/to/Megatron-LM/tools/checkpoint/

python convert.py \
    --model-type GPT \
    --loader core \
    --saver pytorch \
    --load-dir checkpoints/gpt_8b/iter_0000010 \
    --save-dir exported \
    --target-tensor-parallel-size 1
```

## 限制

1. **没有通用的 HF saver**：只有 `saver_hf_llava.py`，只支持 LLaVA
2. **没有 PyTorch saver**：需要自己实现（如 `megatron_saver_pytorch.py`）
3. **转换方向**：主要是 HF/Meta → Megatron，反向转换支持有限

## 总结

- **convert.py**：核心转换框架
- **loader_***：各种格式的加载器
- **saver_***：各种格式的保存器
- **schema_***：参数结构的定义
- **工具文件**：辅助功能

这个插件系统设计得很好，可以轻松扩展新的 loader/saver 来支持更多格式。
