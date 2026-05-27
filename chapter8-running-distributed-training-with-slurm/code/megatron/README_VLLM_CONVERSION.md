# 将 Megatron Checkpoint 转换为 vLLM 格式

## vLLM 的要求

**vLLM 不能直接使用单独的 `.pt` 文件**，它需要：

1. **模型架构定义**（代码）
2. **配置文件**（`config.json`）
3. **权重文件**（`.pt`, `.safetensors`, 或 `.bin`）
4. **HuggingFace 兼容的目录结构**

## 为什么需要这些？

vLLM 主要支持：
- ✅ **HuggingFace Transformers 格式**（最常见，推荐）
- ✅ **自定义模型类**（需要实现模型架构代码）

一个单独的 `.pt` 文件（只有权重）是不够的，因为：
- vLLM 需要知道**模型架构**来初始化模型
- 需要知道**层名称映射**（Megatron 格式 → HuggingFace 格式）
- 需要知道**配置参数**（层数、隐藏层大小等）

## 转换方案

### 方案 1: 转换为 HuggingFace 格式（推荐）

这是最兼容的方式，vLLM 原生支持 HuggingFace 格式。

#### 步骤 1: 转换为 HuggingFace 格式

**选项 A: 使用 Megatron-Bridge（如果有）**

```bash
pip install megatron-bridge

python -c "
from megatron.bridge import AutoBridge
AutoBridge.export_ckpt(
    'checkpoints/gpt_8b/iter_0000010',
    'hf_output'
)
"
```

**选项 B: 手动转换**

需要：
1. 将 Megatron 层名称映射到 HuggingFace 格式
2. 创建 `config.json` 文件
3. 保存权重为 HuggingFace 格式

#### 步骤 2: 创建 HuggingFace 目录结构

```
hf_model/
├── config.json          # 模型配置
├── pytorch_model.bin    # 或 model.safetensors
├── tokenizer_config.json
└── tokenizer.json       # 如果需要 tokenizer
```

#### 步骤 3: 创建 config.json

根据你的模型配置创建 `config.json`：

```json
{
  "vocab_size": 128256,
  "hidden_size": 4096,
  "intermediate_size": 14336,
  "num_hidden_layers": 32,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "max_position_embeddings": 2048,
  "torch_dtype": "bfloat16",
  "model_type": "gpt2",  // 或你的模型类型
  "architectures": ["GPT2LMHeadModel"]
}
```

#### 步骤 4: 转换权重名称

需要将 Megatron 的层名称转换为 HuggingFace 格式：

```python
# Megatron 格式 → HuggingFace 格式映射示例
mapping = {
    'embedding.word_embeddings.weight': 'transformer.wte.weight',
    'decoder.layers.0.self_attention.linear_proj.weight': 'transformer.h.0.attn.c_proj.weight',
    # ... 更多映射
}
```

### 方案 2: 使用自定义模型类（高级）

如果你不想转换为 HuggingFace 格式，可以实现自定义模型类：

```python
from vllm import LLM, SamplingParams
from vllm.model_executor.models import ModelRegistry

# 注册自定义模型
@ModelRegistry.register("custom_gpt")
class CustomGPTModel:
    # 实现模型架构
    # 加载你的 .pt 文件
    pass

# 使用
llm = LLM(
    model="custom_gpt",
    load_format="pt",
    # ... 其他参数
)
```

这需要：
1. 实现完整的模型架构代码
2. 处理权重加载逻辑
3. 确保与 vLLM 的接口兼容

## 实际转换脚本

创建一个转换脚本，将导出的 `.pt` 文件转换为 HuggingFace 格式：

```python
# convert_to_hf_for_vllm.py
import torch
import json
from pathlib import Path

def convert_megatron_to_hf(checkpoint_path, output_dir, config):
    """将 Megatron checkpoint 转换为 HuggingFace 格式"""
    
    # 加载 checkpoint
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckpt['model_state_dict']
    
    # 创建输出目录
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 1. 保存 config.json
    config_path = output_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # 2. 转换层名称并保存权重
    hf_state_dict = {}
    for key, value in state_dict.items():
        # 转换层名称（需要根据实际模型调整）
        hf_key = convert_key_megatron_to_hf(key)
        hf_state_dict[hf_key] = value
    
    # 3. 保存权重
    model_path = output_dir / 'pytorch_model.bin'
    torch.save(hf_state_dict, model_path)
    
    print(f"✓ Converted to HuggingFace format: {output_dir}")
    print(f"  Config: {config_path}")
    print(f"  Model: {model_path}")

def convert_key_megatron_to_hf(key):
    """将 Megatron 层名称转换为 HuggingFace 格式"""
    # 这里需要根据你的模型架构实现具体的映射
    # 示例映射（需要根据实际情况调整）
    if key.startswith('embedding.word_embeddings'):
        return key.replace('embedding.word_embeddings', 'transformer.wte')
    elif 'decoder.layers' in key:
        # 转换 decoder layers
        return key.replace('decoder.layers', 'transformer.h')
    # ... 更多映射规则
    return key

if __name__ == '__main__':
    # 从导出的 checkpoint 读取配置
    ckpt = torch.load('exported_model.pt', map_location='cpu')
    model_config = ckpt['model_config']
    
    # 创建 HuggingFace config
    hf_config = {
        "vocab_size": model_config['vocab_size'],
        "n_embd": model_config['hidden_size'],
        "n_layer": model_config['num_layers'],
        "n_head": model_config['num_attention_heads'],
        "n_inner": model_config['ffn_hidden_size'],
        "n_positions": model_config['max_position_embeddings'],
        "torch_dtype": "bfloat16",
        "model_type": "gpt2",
        "architectures": ["GPT2LMHeadModel"]
    }
    
    convert_megatron_to_hf(
        'exported_model.pt',
        'hf_model',
        hf_config
    )
```

## 使用 vLLM 加载

转换完成后，使用 vLLM 加载：

```bash
python -m vllm.entrypoints.openai.api_server \
    --model hf_model \
    --load-format pt \
    --tensor-parallel-size 1
```

或者使用 Python API：

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="hf_model",  # HuggingFace 格式目录
    load_format="pt",
    dtype="bfloat16",
    tensor_parallel_size=1
)

# 使用
prompts = ["Hello, my name is"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
outputs = llm.generate(prompts, sampling_params)
```

## 总结

| 方案 | 难度 | 推荐度 | 说明 |
|------|------|--------|------|
| HuggingFace 格式 | 中等 | ⭐⭐⭐⭐⭐ | 最兼容，vLLM 原生支持 |
| 自定义模型类 | 高 | ⭐⭐ | 需要实现完整模型架构 |

**推荐流程**：
1. 导出为 `.pt` 文件（已完成）
2. 转换为 HuggingFace 格式（需要层名称映射）
3. 使用 vLLM 加载 HuggingFace 格式

**注意**：层名称映射是关键步骤，需要根据你的具体模型架构来实现。
