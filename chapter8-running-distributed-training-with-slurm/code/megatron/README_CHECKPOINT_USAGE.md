# Using Exported Checkpoints

The exported PyTorch checkpoint (`exported_model.pt`) is **completely independent** and does NOT require Megatron to load or inspect.

## Quick Start (No Megatron Required)

### Simple Python Script

```python
import torch

# Load checkpoint - NO MEGATRON NEEDED!
checkpoint = torch.load('exported_model.pt', map_location='cpu')

# View model configuration
print(checkpoint['model_config'])

# Access state dict
state_dict = checkpoint['model_state_dict']
print(f"Total keys: {len(state_dict)}")
print(f"First key: {list(state_dict.keys())[0]}")
```

### Using the Example Script

```bash
# No Megatron installation needed - only PyTorch!
python load_checkpoint_standalone.py --checkpoint-path exported_model.pt
```

This script will show:
- Model configuration
- State dict keys and shapes
- Parameter statistics
- File size information

## Checkpoint Structure

The exported checkpoint contains:

```python
{
    'model_state_dict': {
        # All model weights in standard PyTorch format
        'embedding.word_embeddings.weight': tensor(...),
        'decoder.layers.0.self_attention.linear_proj.weight': tensor(...),
        # ... etc
    },
    'model_config': {
        'num_layers': 32,
        'hidden_size': 4096,
        'num_attention_heads': 32,
        'vocab_size': 128256,
        'max_position_embeddings': 2048,
        'ffn_hidden_size': 14336,
        'num_query_groups': 8,
        'kv_channels': 128,
    }
}
```

## Loading into Your Own Model

If you have a PyTorch model with matching architecture:

```python
import torch

# Load checkpoint
checkpoint = torch.load('exported_model.pt', map_location='cpu')

# Load into your model
model.load_state_dict(checkpoint['model_state_dict'], strict=False)
```

## Converting to Other Formats

### HuggingFace Transformers

You'll need to:
1. Map the layer names from Megatron format to HuggingFace format
2. Reshape tensors if needed (e.g., attention heads)
3. Handle any architecture differences

### SGLang / vLLM

These frameworks typically accept:
- HuggingFace format checkpoints
- Or standard PyTorch state dicts (with proper model initialization)

## Scripts Available

1. **`load_checkpoint_standalone.py`** - Standalone loader (NO Megatron needed)
   - Just PyTorch required
   - Inspect checkpoint contents
   - View statistics

2. **`load_checkpoint_with_megatron.py`** - Full Megatron loader
   - Requires Megatron-LM installation
   - Builds full model and loads weights
   - For inference with Megatron framework

3. **`convert_megatron_to_pytorch.py`** - Export script (recommended)
   - Converts Megatron checkpoint to PyTorch format
   - Requires Megatron-LM (for reading source checkpoint)

4. **`convert_megatron_checkpoint.py`** - Multi-format export script
   - Supports PyTorch and HuggingFace formats
   - Requires Megatron-LM

## Key Points

✅ **The exported checkpoint is independent** - no Megatron needed to load it  
✅ **Standard PyTorch format** - can be used with any PyTorch model  
✅ **bfloat16 precision** - matches training precision, saves space  
✅ **12.88 GB for 6.44B parameters** - efficient storage  

## Example: Quick Inspection

```bash
python -c "
import torch
ckpt = torch.load('exported_model.pt', map_location='cpu', weights_only=False)
print('Config:', ckpt['model_config'])
print('Keys:', len(ckpt['model_state_dict']))
"
```







