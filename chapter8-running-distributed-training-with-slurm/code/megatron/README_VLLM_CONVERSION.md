# Convert Megatron Checkpoint to vLLM Format

## vLLM Requirements

**vLLM cannot use a standalone `.pt` file directly**. It needs:

1. **Model architecture definition** (code)
2. **Configuration file** (`config.json`)
3. **Weight file** (`.pt`, `.safetensors`, or `.bin`)
4. **HuggingFace-compatible directory layout**

## Why Is This Needed?

vLLM mainly supports:
- ✅ **HuggingFace Transformers format** (most common, recommended)
- ✅ **Custom model classes** (requires implementing model architecture code)

## A standalone `.pt` file is not enough because:
- vLLM needs the **model architecture** to initialize the model
- It needs the **layer name mapping** (Megatron format → HuggingFace format)
- It needs the **configuration parameters** (number of layers, hidden size, etc.)

## Conversion Options

### Option 1: Convert to HuggingFace Format (Recommended)

This is the most compatible option, and vLLM supports HuggingFace format natively.

#### Step 1: Convert to HuggingFace Format

**Option A: Use Megatron-Bridge (if available)**

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

**Option B: Manual conversion**

You need to:
1. Map Megatron layer names to HuggingFace format
2. Create a `config.json` file
3. Save the weights in HuggingFace format

#### Step 2: Create the HuggingFace Directory Structure

```
hf_model/
├── config.json          # model config
├── pytorch_model.bin    # or model.safetensors
├── tokenizer_config.json
└── tokenizer.json       # if a tokenizer is needed
```

#### Step 3: Create `config.json`

Create `config.json` based on your model configuration:

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
  "model_type": "gpt2",
  "architectures": ["GPT2LMHeadModel"]
}
```

#### Step 4: Convert Weight Names

You need to convert Megatron layer names to HuggingFace format:

```python
# Example Megatron format → HuggingFace format mapping
mapping = {
    'embedding.word_embeddings.weight': 'transformer.wte.weight',
    'decoder.layers.0.self_attention.linear_proj.weight': 'transformer.h.0.attn.c_proj.weight',
    # ... more mappings
}
```

### Option 2: Use a Custom Model Class (Advanced)

If you do not want to convert to HuggingFace format, you can implement a custom model class:

```python
from vllm import LLM, SamplingParams
from vllm.model_executor.models import ModelRegistry

# Register the custom model
@ModelRegistry.register("custom_gpt")
class CustomGPTModel:
    # Implement the model architecture
    # Load your .pt file
    pass

# Use it
llm = LLM(
    model="custom_gpt",
    load_format="pt",
    # ... other parameters
)
```

This requires:
1. Implementing the full model architecture code
2. Handling the weight-loading logic
3. Ensuring compatibility with the vLLM interface

## Example Conversion Script

Create a conversion script to turn the exported `.pt` file into HuggingFace format:

```python
# convert_to_hf_for_vllm.py
import torch
import json
from pathlib import Path

def convert_megatron_to_hf(checkpoint_path, output_dir, config):
    """Convert a Megatron checkpoint to HuggingFace format"""
    
    # Load the checkpoint
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state_dict = ckpt['model_state_dict']
    
    # Create the output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # 1. Save config.json
    config_path = output_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # 2. Convert layer names and save weights
    hf_state_dict = {}
    for key, value in state_dict.items():
        # Convert the layer name (adjust for your actual model)
        hf_key = convert_key_megatron_to_hf(key)
        hf_state_dict[hf_key] = value
    
    # 3. Save the weights
    model_path = output_dir / 'pytorch_model.bin'
    torch.save(hf_state_dict, model_path)
    
    print(f"✓ Converted to HuggingFace format: {output_dir}")
    print(f"  Config: {config_path}")
    print(f"  Model: {model_path}")

def convert_key_megatron_to_hf(key):
    """Convert Megatron layer names to HuggingFace format"""
    # Implement the actual mapping based on your model architecture
    # Example mapping (adjust as needed)
    if key.startswith('embedding.word_embeddings'):
        return key.replace('embedding.word_embeddings', 'transformer.wte')
    elif 'decoder.layers' in key:
        # Convert decoder layers
        return key.replace('decoder.layers', 'transformer.h')
    # ... more mapping rules
    return key

if __name__ == '__main__':
    # Read the config from the exported checkpoint
    ckpt = torch.load('exported_model.pt', map_location='cpu')
    model_config = ckpt['model_config']
    
    # Create the HuggingFace config
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

## Load with vLLM

After conversion, load it with vLLM:

```bash
python -m vllm.entrypoints.openai.api_server \
    --model hf_model \
    --load-format pt \
    --tensor-parallel-size 1
```

Or use the Python API:

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="hf_model",  # HuggingFace-format directory
    load_format="pt",
    dtype="bfloat16",
    tensor_parallel_size=1
)

# Use
prompts = ["Hello, my name is"]
sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
outputs = llm.generate(prompts, sampling_params)
```

## Summary

| Option | Difficulty | Recommendation | Notes |
|------|------|--------|------|
| HuggingFace format | Medium | ⭐⭐⭐⭐⭐ | Most compatible, natively supported by vLLM |
| Custom model class | High | ⭐⭐ | Requires implementing the full model architecture |

**Recommended flow**:
1. Export to a `.pt` file (already done)
2. Convert to HuggingFace format (requires layer-name mapping)
3. Load the HuggingFace format with vLLM

**Note**: Layer-name mapping is the key step and must be implemented for your specific model architecture.
