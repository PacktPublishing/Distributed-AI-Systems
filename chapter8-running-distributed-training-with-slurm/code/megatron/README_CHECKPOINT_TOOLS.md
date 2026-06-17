# Megatron-LM tools/checkpoint/ Directory Overview

## Overview

The `tools/checkpoint/` directory contains Megatron-LM checkpoint conversion tools that use a **plugin system** (loader/saver) to convert between formats.

## Core Files

### 1. `convert.py` - Main Conversion Tool
**Purpose**: General checkpoint conversion framework

**How it works**:
- Uses a **loader** plugin to load the source checkpoint
- Uses a **saver** plugin to write the target checkpoint
- Passes data between loader and saver through `multiprocessing.Queue`

**Usage**:
```bash
python convert.py \
    --model-type GPT \
    --loader <loader_name> \
    --saver <saver_name> \
    --load-dir <source_dir> \
    --save-dir <target_dir>
```

**Supported conversion directions**:
- HF/Meta → Megatron (loader already available)
- Megatron → Megatron (change parallelism)
- Megatron → HF (requires a custom saver; currently only LLaVA has one)

## Loader Plugins

### 2. `loader_base.py` - Loader Base Class
**Purpose**: Defines the common interface and shared behavior for all loaders

**Key methods**:
- `load_checkpoint()`: Main function that receives data from the queue
- `send_metadata_over_queue()`: Sends model metadata
- `send_llm_over_queue()`: Sends LLM model weights

### 3. `loader_core.py` - Megatron Core Loader
**Purpose**: Loads Megatron-LM core-format checkpoints (`torch_dist` format)

**Use cases**:
- Load a model from a Megatron checkpoint
- Change tensor/pipeline parallel sizes

**Example**:
```bash
python convert.py \
    --loader core \
    --load-dir checkpoints/gpt_8b/iter_0000010
```

### 4. `loader_legacy.py` - Legacy Format Loader
**Purpose**: Loads old Megatron checkpoint formats

**Use cases**:
- Migrate older checkpoints to the new format

### 5. `loader_llama_mistral.py` - Llama/Mistral Loader
**Purpose**: Loads Llama/Mistral models from HuggingFace or Meta formats

**Supported formats**:
- HuggingFace format (`--checkpoint-type hf`)
- Meta format (`--checkpoint-type meta`)

**Example**:
```bash
python convert.py \
    --loader llama_mistral \
    --checkpoint-type hf \
    --load-dir /path/to/hf_checkpoint \
    --model-size llama2-7B
```

### 6. `loader_mixtral_hf.py` - Mixtral HF Loader
**Purpose**: Loads Mixtral models from HuggingFace format

### 7. `loader_llava.py` - LLaVA Loader
**Purpose**: Loads checkpoints for LLaVA multimodal models

## Saver Plugins

### 8. `saver_base.py` - Saver Base Class
**Purpose**: Defines the common interface and shared behavior for all savers

**Key methods**:
- `save_checkpoint()`: Main function that receives data from the queue and saves it
- `receive_checkpoint_metadata()`: Receives model metadata
- `receive_lm()`: Receives LLM model weights

### 9. `saver_core.py` - Megatron Core Saver
**Purpose**: Saves in Megatron-LM core format (`torch_dist` format)

**Use cases**:
- Convert other formats into Megatron format
- Change the checkpoint parallelism configuration

**Example**:
```bash
python convert.py \
    --saver core \
    --save-dir checkpoints/gpt_8b_converted \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1
```

### 10. `saver_legacy.py` - Legacy Format Saver
**Purpose**: Saves the old Megatron format

**Use cases**:
- Compatibility with older systems

### 11. `saver_hf_llava.py` - HuggingFace LLaVA Saver
**Purpose**: Saves LLaVA models in HuggingFace format

**Note**: This is the **only** HF saver, and it only supports LLaVA, not general GPT models

### 12. `saver_llava.py` - LLaVA Saver
**Purpose**: Saves LLaVA models in Megatron format

## Schema Files

### 13. `schema_base.py` - Schema Base Class
**Purpose**: Defines how model parameters are organized and accessed

**Functions**:
- Defines how to extract parameters from a model
- Defines how to set parameters on a model

### 14. `schema_core.py` - Core Schema
**Purpose**: Defines the parameter structure for Megatron Core models

**Supported model types**:
- GPT
- BERT
- MoE (Mixture of Experts)

### 15. `schema_hf.py` - HuggingFace Schema
**Purpose**: Defines the parameter structure for HuggingFace format

**Functions**:
- Provides the Megatron → HF layer-name mapping
- Currently used mainly for LLaVA

## Utility Files

### 16. `checkpoint_inspector.py` - Checkpoint Inspector
**Purpose**: Inspects and validates checkpoint contents

**Functions**:
- View checkpoint metadata
- Validate checkpoint integrity
- Convert checkpoint formats (`torch_dist` ↔ `fsdp_dtensor`)

### 17. `hybrid_conversion.py` - Hybrid Conversion
**Purpose**: Handles mixed-format conversion, likely for special cases

### 18. `utils.py` - Utility Functions
**Purpose**: Provides helper functions used during conversion

**Main functions**:
- `chunk_weight()`: Splits weights for tensor parallelism
- `chunk_bias()`: Splits bias for tensor parallelism
- `_ConverterFakeProcessGroup`: Simulates a process group for conversion

## Data Flow

```
Source checkpoint (HF/Meta/Megatron)
    ↓
[Loader plugin]
    ↓ (via Queue)
[convert.py]
    ↓ (via Queue)
[Saver plugin]
    ↓
Target checkpoint (Megatron/HF)
```

## Usage Examples

### Example 1: HF → Megatron
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

### Example 2: Megatron → Megatron (change parallelism)
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

### Example 3: Megatron → PyTorch (requires a custom saver)
```bash
# Copy megatron_saver_pytorch.py into tools/checkpoint/
cp megatron_saver_pytorch.py /path/to/Megatron-LM/tools/checkpoint/

python convert.py \
    --model-type GPT \
    --loader core \
    --saver pytorch \
    --load-dir checkpoints/gpt_8b/iter_0000010 \
    --save-dir exported \
    --target-tensor-parallel-size 1
```

## Limitations

1. **No generic HF saver**: only `saver_hf_llava.py`, which supports LLaVA only
2. **No PyTorch saver**: you must implement one yourself, such as `megatron_saver_pytorch.py`
3. **Conversion direction**: mainly HF/Meta → Megatron; reverse conversion is limited

## Summary

- **convert.py**: core conversion framework
- **loader_***: loaders for different formats
- **saver_***: savers for different formats
- **schema_***: parameter-structure definitions
- **utility files**: helper functionality

This plugin system is well designed and makes it easy to extend with new loaders/savers for additional formats.
