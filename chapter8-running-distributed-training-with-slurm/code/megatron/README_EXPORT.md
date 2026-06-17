# Export Megatron Checkpoint to Other Formats

## Overview

Megatron-LM checkpoints are distributed formats and must be converted to a standard format before they can be used in SGLang, LLM, or other frameworks.

## Method 1: Use convert_megatron_to_pytorch.py (Recommended, Easiest)

This script loads and exports directly, which is the simplest approach:

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

## Method 2: Use convert.py + megatron_saver_pytorch.py

Use Megatron-LM's built-in `convert.py` tool together with a custom PyTorch saver:

```bash
cd /home/wukong/workspace/coderepo/Megatron-LM/tools/checkpoint

# Make sure megatron_saver_pytorch.py is on the Python path
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

**Note**: You need to place `megatron_saver_pytorch.py` in `tools/checkpoint/`, or ensure it is on the Python path.

## Method 3: Use convert_megatron_checkpoint.py (Supports Multiple Formats)

This script supports exporting to PyTorch or HuggingFace format:

```bash
python convert_megatron_checkpoint.py \
    --checkpoint-dir checkpoints/gpt_8b/iter_0000010 \
    --output-dir exported \
    --format pytorch  # or 'huggingface'
```

## Method 4: Use Megatron-Bridge (HuggingFace Format)

If you need HuggingFace format:

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

## Method 5: Use the Distributed Checkpoint Directly

Some frameworks, such as SGLang, may support loading Megatron's distributed checkpoints directly, but you need to:
- Ensure the framework supports the Megatron checkpoint format
- Potentially specify the checkpoint path and configuration

## File Size Notes

- **Full checkpoint** (including optimizer): ~108 GB (4 files × 27 GB)
- **Model weights only** (bf16): ~16 GB
- **PyTorch format** (weights only): ~16 GB

## Example: Loading the Exported Checkpoint

```python
import torch

# Load the exported checkpoint
checkpoint = torch.load('model.pt')
state_dict = checkpoint['model_state_dict']
config = checkpoint['model_config']

print(f"Model config: {config}")
print(f"State dict keys: {list(state_dict.keys())[:5]}...")

# Use state_dict to initialize your model
# model.load_state_dict(state_dict)
```

## Notes

1. **Model configuration**: Make sure the model configuration used for export matches the training configuration
2. **Layer-name mapping**: If you need HuggingFace format, you must manually convert the layer names
3. **Distributed checkpoint**: The original checkpoint is distributed, and export merges all shards
