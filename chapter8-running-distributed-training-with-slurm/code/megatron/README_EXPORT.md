# 导出 Megatron Checkpoint 到其他格式

## 概述

Megatron-LM 的 checkpoint 是分布式格式，需要转换为标准格式才能在 SGLang、LLM 或其他框架中使用。

## 方法 1: 使用 convert_megatron_to_pytorch.py（推荐，最简单）

这个脚本直接加载并导出，最简单直接：

```bash
cd /home/wukong/workspace/coderepo/08/code/megatron

source ~/miniconda3/etc/profile.d/conda.sh
conda activate research

python convert_megatron_to_pytorch.py \
    --checkpoint-dir checkpoints/gpt_8b/iter_0000010 \
    --output-path model.pt \
    --num-layers 32 \
    --hidden-size 4096 \
    --num-attention-heads 32 \
    --vocab-size 128256 \
    --max-position-embeddings 2048 \
    --ffn-hidden-size 14336 \
    --num-query-groups 8 \
    --kv-channels 128
```

## 方法 2: 使用 convert.py + megatron_saver_pytorch.py

使用 Megatron-LM 自带的 `convert.py` 工具配合自定义的 PyTorch saver：

```bash
cd /home/wukong/workspace/coderepo/Megatron-LM/tools/checkpoint

# 确保 megatron_saver_pytorch.py 在 Python path 中
export PYTHONPATH=/home/wukong/workspace/coderepo/08/code/megatron:$PYTHONPATH

python convert.py \
    --model-type GPT \
    --loader core \
    --saver pytorch \
    --load-dir /home/wukong/workspace/coderepo/08/code/megatron/checkpoints/gpt_8b/iter_0000010 \
    --save-dir /home/wukong/workspace/coderepo/08/code/megatron/exported \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --output-filename model.pt
```

**注意**：需要将 `megatron_saver_pytorch.py` 放在 `tools/checkpoint/` 目录下，或者确保它在 Python path 中。

## 方法 3: 使用 convert_megatron_checkpoint.py（支持多种格式）

这个脚本支持导出为 PyTorch 或 HuggingFace 格式：

```bash
python convert_megatron_checkpoint.py \
    --checkpoint-dir checkpoints/gpt_8b/iter_0000010 \
    --output-dir exported \
    --format pytorch  # 或 'huggingface'
```

## 方法 4: 使用 Megatron-Bridge（HuggingFace 格式）

如果需要 HuggingFace 格式：

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

## 方法 5: 直接使用分布式 checkpoint

一些框架（如 SGLang）可能支持直接加载 Megatron 的分布式 checkpoint，但需要：
- 确保框架支持 Megatron 的 checkpoint 格式
- 可能需要指定 checkpoint 路径和配置

## 文件大小说明

- **完整 checkpoint**（含优化器）：~108 GB（4个文件 × 27 GB）
- **仅模型权重**（bf16）：~16 GB
- **PyTorch 格式**（仅权重）：~16 GB

## 示例：加载导出的 checkpoint

```python
import torch

# 加载导出的 checkpoint
checkpoint = torch.load('model.pt')
state_dict = checkpoint['model_state_dict']
config = checkpoint['model_config']

print(f"Model config: {config}")
print(f"State dict keys: {list(state_dict.keys())[:5]}...")

# 使用 state_dict 初始化你的模型
# model.load_state_dict(state_dict)
```

## 注意事项

1. **模型配置**：确保导出时使用的模型配置与训练时一致
2. **层名称映射**：如果需要 HuggingFace 格式，需要手动转换层名称
3. **分布式 checkpoint**：原始 checkpoint 是分布式的，导出时会合并所有分片
