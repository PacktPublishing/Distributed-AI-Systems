## FSDP T5 Training Examples

This directory contains examples for training T5 models for text summarization using both FSDP1 and FSDP2 (Fully Sharded Data Parallel) for distributed training across multiple GPUs.

**Available Scripts**:
- `T5_training_Single.py`: Single GPU training
- `T5_training_FSDP1.py`: FSDP1 (deprecated, but included for comparison)
- `T5_training_FSDP2.py`: FSDP2 (recommended, PyTorch 2.1+)

**Note**: FSDP1 is deprecated. FSDP2 is the recommended approach with improved performance and simpler API. See [FSDP2 tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html) and [code examples](https://github.com/pytorch/examples/tree/main/distributed/FSDP2).

### Dataset

**WikiHow Dataset**: A text summarization dataset containing how-to articles with corresponding headlines/summaries. The dataset is downloaded as CSV files:
- `wikihowAll.csv`: Complete dataset
- `wikihowSep.csv`: Separated version
- Paper: https://arxiv.org/abs/1810.09305

**Data Format**:
- Each sample contains two fields:
  - `text`: The full article text (input)
  - `headline`: The summary/headline (target)
- The dataset is loaded using the `nlp` library's WikiHow dataset loader
- Text preprocessing includes cleaning special characters, newlines, and quotes

**Dataset Splits**:
- **Training**: 1,500 samples (configurable via `num_samples` parameter)
- **Validation**: 300 samples (configurable via `num_samples` parameter)

**Features**:
- **Input sequence length**: 512 tokens (max length, with padding/truncation)
- **Output sequence length**: 150 tokens (max length, with padding/truncation)
- Both input and output are tokenized using T5 tokenizer with padding to max length
- Attention masks are generated for both source and target sequences

### Model

**FLAN-T5 Models**: Modern instruction-tuned variants of T5 from HuggingFace Transformers. All models use the same encoder-decoder architecture and can be used as drop-in replacements.

**Available FLAN-T5 Models**:

| Model | Parameters | HuggingFace ID | Memory (bfloat16) | Best For |
|-------|-----------|----------------|-------------------|----------|
| **FLAN-T5-Small** | 60M | `google/flan-t5-small` | ~120 MB | Quick testing, single GPU |
| **FLAN-T5-Base** | 250M | `google/flan-t5-base` | ~500 MB | Small-scale training |
| **FLAN-T5-Large** | 780M (0.8B) | `google/flan-t5-large` | ~1.6 GB | Medium-scale, minimal FSDP benefit |
| **FLAN-T5-XL** | 3B | `google/flan-t5-xl` | ~6 GB | Good FSDP demonstration |
| **FLAN-T5-XXL** | 11B | `google/flan-t5-xxl` | ~22 GB | Best FSDP demonstration |

**Default Model**: `google/flan-t5-small` (60M parameters)

**Model Selection**:
- **For FSDP demonstration**: Use `google/flan-t5-xxl` (11B) or `google/flan-t5-xl` (3B) to clearly see FSDP benefits
- **For quick testing**: Use `google/flan-t5-small` (default)
- **Note**: Smaller models (<1B) may not show FSDP speedup due to communication overhead dominating computation

**Architecture**: Text-to-Text Transfer Transformer (T5) - encoder-decoder
- **Task**: Conditional text generation for summarization with improved instruction following
- **Tokenizer**: T5Tokenizer (SentencePiece-based)
- **Advantages**: Better performance on instruction-following tasks compared to original T5

The model is wrapped with FSDP for efficient distributed training across multiple GPUs.

### VRAM Requirements by Model

**Memory Requirements Table - Single GPU** (approximate, with batch_size=4):

| Model | Params | Model Weights | Gradients | Optimizer | Activations* | Temp Buffers | **Total per GPU** |
|-------|--------|---------------|-----------|-----------|--------------|-------------|-------------------|
| **FLAN-T5-Small** | 60M | 0.12 GB | 0.12 GB | 0.24 GB | 0.1-0.2 GB | 0.1 GB | **~0.6-0.8 GB** |
| **FLAN-T5-Base** | 250M | 0.5 GB | 0.5 GB | 1.0 GB | 0.2-0.5 GB | 0.2 GB | **~2.4-2.7 GB** |
| **FLAN-T5-Large** | 780M | 1.6 GB | 1.6 GB | 3.1 GB | 0.5-1.5 GB | 0.5 GB | **~7.3-8.3 GB** |
| **FLAN-T5-XL** | 3B | 6 GB | 6 GB | 12 GB | 2-6 GB | 2 GB | **~28-32 GB** |
| **FLAN-T5-XXL** | 11B | 22 GB | 22 GB | 44 GB | 25-45 GB | 10 GB | **~123-143 GB** |

**Notes**:
- All values are for **Single GPU training with bfloat16** (default precision for both scripts)
- **bfloat16**: 16-bit precision (2 bytes per parameter) - used by default in both FSDP and single GPU scripts
- **Optimizer**: AdamW requires 2× model size (momentum + variance)
- **Activations***: Peak memory during forward+backward pass. For T5 encoder-decoder models, activations are HIGH:
  - Encoder activations: ~10-15 GB (for XXL with batch_size=4)
  - Decoder activations: ~10-20 GB (decoder is memory-intensive)
  - Cross-attention: ~5-10 GB
  - **Total activations: 25-45 GB** (scales heavily with batch size)
- **Temp Buffers**: Temporary CUDA buffers, memory fragmentation overhead, and PyTorch allocator reserves (~10 GB for XXL)
- **Memory Fragmentation**: PyTorch's memory allocator can waste 5-15% due to fragmentation, especially for large models

**FSDP1 Memory Usage (with 2 GPUs, FULL_SHARD) - Actual Observed Values**:

**FLAN-T5-XXL (11B) - FSDP1 Observed**:
- **Allocated**: **84.05 GB** (steady-state memory usage)
- **Reserved**: **133.84 GB** (PyTorch reserved memory)
- **Peak Allocated**: **104.79 GB** (maximum during training)
- **Training Speed**: ~1.95-1.97 it/s
- **Training Time**: ~101.10 seconds per epoch

**FLAN-T5-XL (3B) - FSDP1 Observed ~45 GB per GPU**:
- **Sharded components** (steady-state): 3 GB (weights) + 3 GB (gradients) + 6 GB (optimizer) = **12 GB**
- **Peak during all-gather** (temporary): Full model weights (~6 GB) + full gradients (~6 GB) = **+12 GB peak overhead**
- **Activations** (NOT sharded): **10-15 GB** (actual peak, higher than estimated 2-6 GB)
- **Communication buffers**: **5-8 GB**
- **Memory fragmentation**: **3-5 GB**
- **Total observed**: **~45 GB per GPU**

**FSDP2 Memory Usage (with 2 GPUs) - Actual Observed Values**:

**FLAN-T5-XXL (11B) - FSDP2 Observed**:
- **Allocated**: **84.05 GB** (steady-state memory usage)
- **Reserved**: **123.11 GB** (PyTorch reserved memory)
- **Peak Allocated**: **104.80 GB** (maximum during training)
- **Training Speed**: ~1.85-1.87 it/s
- **Training Time**: ~106.44 seconds per epoch

**FSDP1 vs FSDP2 Comparison (FLAN-T5-XXL, 2 GPUs)**:
- **Memory Usage**: Both use similar allocated memory (~84 GB), but FSDP2 has lower reserved memory (123 GB vs 134 GB)
- **Training Speed**: FSDP1 is slightly faster (~1.96 it/s vs ~1.86 it/s, ~5% faster)
- **Training Time**: FSDP1 ~101s/epoch vs FSDP2 ~106s/epoch
- **Recommendation**: FSDP2 offers better memory efficiency (lower reserved memory) and simpler API, with minimal performance difference

**Key Insight**: The theoretical breakdown (sharded components + activations) underestimates actual usage because:
1. **Peak memory spikes** during all-gather operations temporarily materialize full model parameters
2. **Actual activations** are 2-3× higher than conservative estimates
3. **Fragmentation** wastes significant memory in dynamic allocation patterns

**Memory Requirements Table - FSDP1** (approximate, with batch_size=4 per GPU, bfloat16):

| Model | Params | GPUs | Sharded Weights | Sharded Gradients | Sharded Optimizer | Activations* | Comm Buffers | Temp/Frag | **Total per GPU** |
|-------|--------|------|-----------------|-------------------|-------------------|--------------|--------------|-----------|-------------------|
| **FLAN-T5-Small** | 60M | 2 | 0.06 GB | 0.06 GB | 0.12 GB | 0.1-0.2 GB | 0.5-1 GB | 0.2 GB | **~1.0-1.7 GB** |
| **FLAN-T5-Base** | 250M | 2 | 0.25 GB | 0.25 GB | 0.5 GB | 0.2-0.5 GB | 1-2 GB | 0.5 GB | **~2.7-4.0 GB** |
| **FLAN-T5-Large** | 780M | 2 | 0.8 GB | 0.8 GB | 1.6 GB | 0.5-1.5 GB | 2-3 GB | 1 GB | **~6.7-8.7 GB** |
| **FLAN-T5-XL** | 3B | 2 | 3 GB | 3 GB | 6 GB | 2-6 GB | 5-8 GB | 3 GB | **~19-26 GB** (FSDP1 observed: ~45 GB) |
| **FLAN-T5-XXL** | 11B | 2 | 11 GB | 11 GB | 22 GB | 25-45 GB | 10-15 GB | 10 GB | **~93-118 GB** (FSDP1 observed: ~84 GB allocated, ~105 GB peak) |

**Memory Requirements Table - FSDP2** (approximate, with batch_size=4 per GPU, bfloat16):

| Model | Params | GPUs | Allocated | Peak Allocated | **Observed per GPU** |
|-------|--------|------|-----------|----------------|----------------------|
| **FLAN-T5-XXL** | 11B | 2 | ~84 GB | ~105 GB | **84.05 GB allocated, 104.80 GB peak** |

**FSDP2 vs FSDP1 Comparison (FLAN-T5-XXL, 2 GPUs)**:
- **Allocated Memory**: Both use similar allocated memory (~84 GB)
- **Reserved Memory**: FSDP2 uses less reserved memory (123 GB vs 134 GB, ~8% reduction)
- **Training Speed**: FSDP1 is slightly faster (~1.96 it/s vs ~1.86 it/s, ~5% faster)
- **Training Time**: FSDP1 ~101s/epoch vs FSDP2 ~106s/epoch (~5% slower)
- **API**: FSDP2 has simpler API (`fully_shard()` function vs wrapper class)
- **Recommendation**: FSDP2 offers better memory efficiency and simpler API with minimal performance overhead


**FSDP1 Notes**:
- **Sharded components** (divided by number of GPUs): Model weights, gradients, and optimizer states
- **Unsharded components** (same per GPU): Activations are NOT sharded - each GPU stores full activation memory
- **Activations***: Peak memory during forward+backward. For T5 encoder-decoder, expect 25-45 GB for XXL (scales with batch size)
- **Comm Buffers**: Memory for all-gather operations during forward/backward passes. Larger for bigger models and more GPUs
- **Temp/Frag**: Temporary buffers, fragmentation overhead, and PyTorch allocator reserves

**FSDP2 Notes**:
- **Improved Memory Management**: FSDP2 uses more efficient memory allocation and communication patterns
- **Better Reserved Memory**: Lower reserved memory compared to FSDP1 (~8% reduction for XXL)
- **Simpler API**: Uses `fully_shard()` function instead of wrapper class
- **Mixed Precision**: Uses `MixedPrecisionPolicy` (simpler API than FSDP1)
- **Performance**: Slightly slower (~5%) than FSDP1, but offers better memory efficiency
- **Recommended**: Use FSDP2 for new projects (requires PyTorch 2.1+)

**Why Observed Values Are Much Higher Than Estimates**:
The large discrepancy (e.g., XL: 19-26 GB estimated vs 45 GB observed, XXL: 93-118 GB estimated vs 84-105 GB observed) is due to:

1. **Peak Memory During All-Gather**: FSDP temporarily gathers full model parameters during forward/backward passes. This means:
   - During forward: Each GPU temporarily holds **full model weights** (not just sharded portion) + activations
   - During backward: Full gradients + full optimizer states are temporarily materialized
   - For XXL: This adds ~22 GB (full weights) + ~22 GB (full gradients) = **~44 GB peak overhead**

2. **Underestimated Activations**: The table shows conservative estimates, but actual peak activations are higher:
   - XL: Estimated 2-6 GB, but actual peak can be 10-15 GB with batch_size=4
   - XXL: Estimated 25-45 GB, but actual peak can be 40-50 GB during decoder generation

3. **Memory Fragmentation**: PyTorch's memory allocator can waste 10-20% due to fragmentation, especially during dynamic allocation/deallocation cycles in FSDP

4. **Communication Overhead**: All-gather operations require temporary buffers that scale with model size and number of GPUs

**Practical Recommendations**:
- **FSDP1**:
  - **XL with 2 GPUs**: Plan for ~45 GB per GPU
  - **XXL with 2 GPUs**: Plan for ~84-105 GB per GPU (allocated), ~134 GB reserved
- **FSDP2** (Recommended):
  - **XXL with 2 GPUs**: Plan for ~84-105 GB per GPU (allocated), ~123 GB reserved
  - **Advantage**: Lower reserved memory, simpler API, minimal performance difference
- **Batch size**: Reduce batch size per GPU to lower activation memory (e.g., `--batch-size 2` or `--batch-size 1` for XXL)

### Hardware Requirements

**GPU VRAM (Video RAM)** - varies by model:

**FLAN-T5-Small (60M) - Default**:
- **Minimum per GPU**: 3-4 GB
- **Recommended per GPU**: 6-8 GB or more
- **For single GPU training**: 8-12 GB recommended
- **Memory breakdown per GPU (with 4 GPUs, batch_size=4)**:
  - Model parameters (sharded): ~0.15 GB (with bfloat16)
  - Gradients (sharded): ~0.15 GB
  - Optimizer states (AdamW, sharded): ~0.3 GB
  - Activations: ~0.1-0.2 GB
  - Communication buffers: ~1-2 GB
  - **Total**: ~2-3 GB per GPU (with safety margin: 3-4 GB)

**FLAN-T5-XXL (11B)**:
- **Minimum per GPU**: 30-35 GB (with FSDP, 2 GPUs)
- **Recommended per GPU**: 40-50 GB or more
- **Memory breakdown per GPU (with 2 GPUs, batch_size=2)**:
  - Model parameters (sharded): ~11 GB (with bfloat16)
  - Gradients (sharded): ~11 GB
  - Optimizer states (AdamW, sharded): ~22 GB
  - Activations: ~2-4 GB
  - Communication buffers: ~2-3 GB
  - **Total**: ~48-52 GB per GPU (with safety margin: 50-60 GB)
- **Note**: H200 GPUs (141GB) can handle this comfortably

**FLAN-T5-XL (3B)**:
- **Minimum per GPU**: 10-12 GB (with FSDP, 2 GPUs)
- **Recommended per GPU**: 16-20 GB or more
- **Memory breakdown per GPU (with 2 GPUs, batch_size=4)**:
  - Model parameters (sharded): ~3 GB (with bfloat16)
  - Gradients (sharded): ~3 GB
  - Optimizer states (AdamW, sharded): ~6 GB
  - Activations: ~1-2 GB
  - Communication buffers: ~1-2 GB
  - **Total**: ~14-16 GB per GPU (with safety margin: 16-20 GB)

**Note**: With FSDP FULL_SHARD, memory is distributed across GPUs. More GPUs = less memory per GPU, but total memory requirement remains similar.

**CPU RAM (System Memory)**:
- **Minimum**: 8 GB
- **Recommended**: 16 GB or more
- **Memory usage breakdown**:
  - Dataset files (CSV): ~1.9 GB (wikihowAll.csv: 591MB, wikihowSep.csv: 1.3GB)
  - Tokenized dataset in memory: ~5-10 MB (1,800 samples)
  - Model loading (before FSDP sharding): ~240 MB
  - DataLoader workers (2 workers): ~200-400 MB
  - System overhead and Python runtime: ~2-4 GB
  - **Total**: ~5-7 GB minimum, 8-16 GB recommended

**Hard Disk Storage**:
- **Minimum**: 10 GB free space
- **Recommended**: 20 GB or more
- **Storage breakdown**:
  - Dataset files: ~1.9 GB (wikihowAll.csv + wikihowSep.csv)
  - HuggingFace model cache: Varies by model
    - FLAN-T5-Small: ~240 MB
    - FLAN-T5-Base: ~500 MB
    - FLAN-T5-Large: ~1.6 GB
    - FLAN-T5-XL: ~6 GB
    - FLAN-T5-XXL: ~22 GB
  - Tokenizer cache: ~50-100 MB
  - Python packages and dependencies: ~2-3 GB
  - Model checkpoints (if saved): Varies by model size
    - FLAN-T5-Small: ~240 MB per checkpoint
    - FLAN-T5-XXL: ~22 GB per checkpoint
  - Training logs and outputs: ~100-500 MB
  - **Total**: ~6-8 GB minimum, 15-20 GB recommended for comfortable operation

**GPU Recommendations**:
- **Minimum**: 1x GPU with 8GB VRAM (e.g., NVIDIA RTX 3060, RTX 3070)
- **Recommended**: 2-4x GPUs with 8GB+ VRAM each (e.g., NVIDIA RTX 3080, A100, V100)
- **Optimal**: 4-8x GPUs with 16GB+ VRAM each for faster training

### Training Details

**Optimization**:
- **Optimizer**: AdamW
  - Learning rate: 0.002 (default)
  - Weight decay: 0.0
- **Scheduler**: StepLR
  - Step size: 1 epoch
  - Gamma: 0.85 (learning rate decay factor)

**Training Configuration**:
- **Batch size**: 4 (default, adjustable via `--batch-size`)
- **Validation batch size**: 4 (default, adjustable via `--test-batch-size`)
- **Epochs**: 2 (default, adjustable via `--epochs`)
- **DataLoader workers**: 2
- **Pin memory**: Enabled for faster GPU transfer

**FSDP1 Configuration**:
- **Sharding Strategy**: FULL_SHARD (shards model parameters, gradients, and optimizer states)
- **Mixed Precision**: Enabled by default (`mixed_precision=True`)
  - **Default policy**: `bfSixteen` (if bfloat16 is supported)
  - **Precision breakdown**:
    - **Model Parameters**: BF16 (`param_dtype=torch.bfloat16`)
    - **Gradients**: BF16 (`reduce_dtype=torch.bfloat16`)
    - **Buffers** (BatchNorm, LayerNorm stats): BF16 (`buffer_dtype=torch.bfloat16`)
    - **Activations**: BF16 (during forward/backward pass)
    - **Optimizer States** (AdamW momentum/variance): **FP32** (PyTorch default, not controlled by MixedPrecision policy)
  - Falls back to FP32 if bfloat16 is not supported
  - FP16 can be enabled via `use_fp16=True` configuration
- **Activation Checkpointing**: Optional (disabled by default to avoid known issues)
- **State Dict Type**: FULL_STATE_DICT (for checkpointing)
- **Limit All Gathers**: Enabled for performance optimization

**FSDP2 Configuration**:
- **API**: Uses `fully_shard()` function (FSDP2) instead of `FullyShardedDataParallel` wrapper (FSDP1)
- **Sharding**: Automatically shards model parameters, gradients, and optimizer states
- **Mixed Precision**: Enabled by default (`mixed_precision=True`)
  - **Default policy**: `MixedPrecisionPolicy` with bfloat16 (if supported)
  - **Precision breakdown**:
    - **Model Parameters**: BF16 (`param_dtype=torch.bfloat16`)
    - **Gradients**: BF16 (`reduce_dtype=torch.bfloat16`)
    - **Activations**: BF16 (during forward/backward pass)
    - **Optimizer States** (AdamW momentum/variance): **FP32** (PyTorch default)
  - Falls back to FP32 if bfloat16 is not supported
  - FP16 can be enabled via `use_fp16=True` configuration
- **Activation Checkpointing**: Optional (disabled by default)
- **Device Management**: Uses `torch.accelerator` API for device detection
- **Memory Efficiency**: More efficient memory management than FSDP1
- **Checkpointing**: Uses **Distributed Checkpointing (DCP)** via `torch.distributed.checkpoint`
  - **Why DCP?**: FSDP2 cannot use `torch.save(model.state_dict())` because it would gather all sharded parameters to CPU, causing OOM for large models
  - **How it works**: Each GPU saves its sharded parameters directly to disk in parallel
  - **Benefits**:
    - **Parallel saving**: Each GPU independently saves its shard, much faster than gathering
    - **No OOM**: Avoids gathering all parameters to a single node
    - **Asynchronous**: Supports async saving without blocking training
    - **Universal**: Checkpoints can be loaded by FSDP2 or converted to HuggingFace format for inference
  - **Implementation**: Uses `save_fsdp2_checkpoint()` and `load_fsdp2_checkpoint()` functions

**Distributed Training**:
- Uses `torchrun` for multi-GPU training
- DistributedSampler for data sharding across processes
- NCCL backend for GPU communication
- Validation loss tracking with best model checkpointing

**Additional Features**:
- Memory tracking (optional, enabled by default)
- Model checkpointing on best validation loss
  - **FSDP1**: Uses `FullStateDictConfig` or `ShardedStateDictConfig` context managers
  - **FSDP2**: Uses Distributed Checkpointing (DCP) - each GPU saves its shard in parallel
- Progress bars for training and validation
- Distributed loss aggregation across all processes

### Usage

To run the T5 example for text summarization:

## Get the wikihow dataset

```bash
sh download_dataset.sh
```

This downloads the WikiHow dataset CSV files into the `data/` directory.

## Install the requirements:
~~~
pip install -r requirements.txt
~~~

Required packages:
- `transformers`: For T5 model and tokenizer
- `datasets`: For dataset utilities
- `nlp`: For WikiHow dataset loading
- `tqdm`: For progress bars
- `SentencePiece`: For tokenization

## Ensure you are running a recent version of PyTorch:
- **FSDP1**: Requires PyTorch 1.12+
- **FSDP2**: Requires PyTorch 2.1+ (recommended)
- See https://pytorch.org/get-started/locally/ to install the appropriate version

Start the training with Torchrun (adjust nproc_per_node to your GPU count):

**FSDP2 (Recommended)**:
```bash
# Basic usage (default: FLAN-T5-Small)
torchrun --nnodes 1 --nproc_per_node 2 T5_training_FSDP2.py

# Use FLAN-T5-XXL (11B) - best for demonstrating FSDP benefits
torchrun --nnodes 1 --nproc_per_node 2 T5_training_FSDP2.py --model-name google/flan-t5-xxl

# Use FLAN-T5-XL (3B) - good middle ground
torchrun --nnodes 1 --nproc_per_node 2 T5_training_FSDP2.py --model-name google/flan-t5-xl

# With custom batch size and epochs
torchrun --nnodes 1 --nproc_per_node 2 T5_training_FSDP2.py \
    --model-name google/flan-t5-xxl \
    --batch-size 2 \
    --epochs 4
```

**FSDP1 (Deprecated, for comparison only)**:
```bash
# Basic usage (default: FLAN-T5-Small)
torchrun --nnodes 1 --nproc_per_node 2 T5_training_FSDP1.py

# Use FLAN-T5-XXL (11B)
torchrun --nnodes 1 --nproc_per_node 2 T5_training_FSDP1.py --model-name google/flan-t5-xxl

# Use FLAN-T5-XL (3B)
torchrun --nnodes 1 --nproc_per_node 2 T5_training_FSDP1.py --model-name google/flan-t5-xl
```

**Note**: For large models (XL, XXL), you may need to reduce batch size to avoid OOM:
- FLAN-T5-XXL: Try `--batch-size 1` or `--batch-size 2`
- FLAN-T5-XL: Try `--batch-size 2` or `--batch-size 4`

**Command-line Arguments**:
- `--model-name`: FLAN-T5 model to use (default: `google/flan-t5-small`)
  - Must be a FLAN-T5 model (contains "flan-t5" in name)
  - Valid options: `google/flan-t5-small`, `google/flan-t5-base`, `google/flan-t5-large`, `google/flan-t5-xl`, `google/flan-t5-xxl`
- `--batch-size`: Training batch size per GPU (default: 4)
- `--test-batch-size`: Validation batch size per GPU (default: 4)
- `--epochs`: Number of training epochs (default: 4)
- `--seed`: Random seed (default: 1)
- `--track_memory`: Track GPU memory usage (default: True)
- `--run_validation`: Run validation after each epoch (default: True)

**Single GPU Training**:
```bash
# Default model (FLAN-T5-Small)
python T5_training_Single.py

# With custom model
python T5_training_Single.py --model-name google/flan-t5-large

# With mixed precision
python T5_training_Single.py --model-name google/flan-t5-large --use-amp
```

### Checkpointing

**FSDP2 Checkpointing (Recommended)**:

FSDP2 uses **Distributed Checkpointing (DCP)** for saving and loading models. This is the recommended approach for FSDP2 because:

1. **Avoids OOM**: Traditional `torch.save(model.state_dict())` would gather all sharded parameters to CPU, causing OOM for large models
2. **Parallel Saving**: Each GPU saves its shard directly to disk in parallel
3. **Fast**: No need to gather parameters before saving

**How it works**:
- Checkpoints are saved to `checkpoints/{model_name}/checkpoint_epoch_{epoch}/`
- Each GPU saves its sharded parameters independently
- Loading is also parallel - each GPU loads its corresponding shard

**Enable checkpointing**:
- Set `save_model=True` in `configs/training.py` or modify the training script
- Checkpoints are automatically saved when validation loss improves

**Checkpoint format**:
- Model state dict: `model/`
- Optimizer state dict: `optimizer/` (if `save_optimizer=True`)
- Each rank saves its shard to separate files

**Loading checkpoints**:
```python
from model_checkpointing.checkpoint_handler import load_fsdp2_checkpoint

# Load checkpoint (resume training)
epoch = load_fsdp2_checkpoint(
    model=model,
    optimizer=optimizer,
    rank=rank,
    checkpoint_dir=checkpoint_dir,
    load_optimizer=True,
)
```

**FSDP1 Checkpointing (Legacy)**:

FSDP1 uses context managers (`FullStateDictConfig` or `ShardedStateDictConfig`) to gather parameters before saving. This can cause OOM for very large models.
