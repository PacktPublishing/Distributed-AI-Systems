# Megatron-LM Training with SLURM

This directory contains an example for running Megatron-LM GPT pretraining with SLURM on a multi-node single-GPU setup.

## Overview

Megatron-LM is NVIDIA's framework for training large language models with tensor, pipeline, and data parallelism. This example demonstrates how to run Megatron-LM training jobs using SLURM as the job scheduler.

## Files

1. **`run.slurm`** - SLURM batch script for launching Megatron-LM training
2. **`README.md`** - This file

## Prerequisites

1. **Megatron-LM Installation**: 
   
   **Important**: The PyPI package `megatron-core` only includes `megatron.core`, **NOT** `megatron.training`.
   Since `pretrain_gpt.py` requires `megatron.training`, you **must install from source**.
   
   **Install from source (Required)**
   ```bash
   conda activate research
   git clone https://github.com/NVIDIA/Megatron-LM.git
   cd Megatron-LM
   pip install --no-build-isolation .[mlm,dev]
   ```
   
   This installs:
   - `megatron.core` - Core library (parallelism, models, etc.)
   - `megatron.training` - Training utilities (required by `pretrain_gpt.py`)
   - All dependencies
   
   **Note**: This example includes `pretrain_gpt.py`, `gpt_builders.py`, and `model_provider.py` 
   from the Megatron-LM repository. However, these scripts still require `megatron.training` 
   to be installed, which is only available when installing from source.
   
   **Alternative**: If you want to build a package that includes `megatron.training`, see 
   `BUILD_PACKAGE.md` for instructions on modifying `pyproject.toml`.

2. **SLURM Cluster**: 
   - Multi-node setup with at least 2 nodes
   - One GPU per node (as configured in this example)
   - Conda environment with PyTorch and Megatron dependencies

3. **Conda Environment**:
   - Environment named `research` (or modify the script)
   - PyTorch with CUDA support
   - Megatron-Core installed (see above)

## Configuration

### Multi-Node Single-GPU Setup

The `run.slurm` script is configured for:
- **2 nodes** (`--nodes=2`)
- **1 GPU per node** (`--gres=gpu:1`)
- **1 task per node** (`--ntasks-per-node=1`)

To use more nodes, modify:
```bash
#SBATCH --nodes=4           # 4 nodes = 4 GPUs total
#SBATCH --gres=gpu:1        # 1 GPU per node
#SBATCH --ntasks-per-node=1 # 1 task per node
```

### Model Configuration

The script uses a smaller model configuration suitable for demonstration:
- **32 layers**
- **Hidden size: 4096**
- **FFN hidden size: 14336**
- **Attention heads: 32**
- **Sequence length: 2048** (reduced from 8192 for memory)

To train larger models, adjust:
- `NUM_LAYERS`
- `HIDDEN_SIZE`
- `FFN_HIDDEN_SIZE`
- `SEQ_LENGTH`

### Parallelism Configuration

- **Tensor Parallelism (TP)**: `TP_SIZE=1` (no tensor parallelism)
- **Context Parallelism (CP)**: `CP_SIZE=1` (no context parallelism)
- **Pipeline Parallelism (PP)**: `PP_SIZE=1` (no pipeline parallelism)

For larger models, you can enable:
- Tensor parallelism: `TP_SIZE=2` (requires 2 GPUs per node)
- Pipeline parallelism: `PP_SIZE=2` (splits model across 2 stages)

## Files

This directory contains:
- **`pretrain_gpt.py`** - Main training script (copied from Megatron-LM repository)
- **`gpt_builders.py`** - Model builder helper (required by pretrain_gpt.py)
- **`model_provider.py`** - Model provider helper (required by pretrain_gpt.py)
- **`run.slurm`** - SLURM batch script
- **`README.md`** - This file

## Usage

### 1. Install Megatron-Core

Install Megatron-Core in your conda environment:

```bash
conda activate research
pip install --no-build-isolation megatron-core[mlm,dev]
```

### 2. Create Logs Directory

```bash
mkdir -p logs
```

### 3. Submit Job

```bash
sbatch run.slurm
```

### 4. Monitor Job

```bash
# Check job status
squeue -u $USER

# View logs
tail -f logs/train_*.out
tail -f logs/train_*.err
```

## Customization

### Using Real Data

By default, the script uses mock data. To use real data:

1. Set environment variables before submitting:
```bash
export USE_MOCK_DATA=0
export DATA_ARG="/path/to/your/data"
export TOKENIZER_ARG="/path/to/tokenizer.model"
```

2. Or modify the script:
```bash
USE_MOCK_DATA=0
DATA_ARG="/path/to/your/data"
TOKENIZER_ARG="/path/to/tokenizer.model"
```

### Adjusting Model Size

Edit the configuration variables in `run.slurm`:

```bash
NUM_LAYERS=40              # More layers
HIDDEN_SIZE=5120           # Larger hidden size
FFN_HIDDEN_SIZE=20480      # Larger FFN
SEQ_LENGTH=4096           # Longer sequences
```

### Enabling Tensor Parallelism

For tensor parallelism (requires multiple GPUs per node):

1. Modify SLURM configuration:
```bash
#SBATCH --gres=gpu:2      # 2 GPUs per node
```

2. Set TP_SIZE:
```bash
TP_SIZE=2
```

3. Update torchrun arguments in the script to use `--nproc_per_node=2`

## Key Features

### SLURM Integration

- **Automatic master address**: Uses SLURM to determine master node
- **GPU mapping**: Maps virtual node names to GPU devices
- **Environment variables**: Automatically sets `MASTER_ADDR`, `MASTER_PORT`, `WORLD_SIZE`
- **Conda activation**: Ensures conda environment is activated on each compute node

### Megatron-LM Features

- **Multi-node training**: Supports distributed training across multiple nodes
- **Mixed precision**: Uses bfloat16 for training
- **Checkpointing**: Automatic checkpoint saving and loading
- **TensorBoard logging**: Logs training metrics to TensorBoard
- **Mock data support**: Can train with mock data for testing

## Troubleshooting

### Error: pretrain_gpt.py not found

Ensure `pretrain_gpt.py` is in the same directory as `run.slurm`. The script looks for it in the current directory.

### Error: ModuleNotFoundError: No module named 'megatron'

Install Megatron-Core:
```bash
pip install --no-build-isolation megatron-core[mlm,dev]
```

### Error: ModuleNotFoundError: No module named 'gpt_builders'

Ensure `gpt_builders.py` and `model_provider.py` are in the same directory as `pretrain_gpt.py`.

### Error: Conda environment not found

Modify the conda activation section in `run.slurm` to match your environment name and path.

### Out of Memory

- Reduce `SEQ_LENGTH` (e.g., from 2048 to 1024)
- Reduce `MICRO_BATCH_SIZE` (already set to 1)
- Reduce model size (`NUM_LAYERS`, `HIDDEN_SIZE`)
- Enable CPU offloading (not shown in this example)

### NCCL Communication Errors

- Check `NCCL_SOCKET_IFNAME` is set correctly
- Verify network connectivity between nodes
- Check firewall settings

## Comparison with Other Frameworks

### vs. PyTorch DDP

- **Megatron**: Supports tensor and pipeline parallelism, not just data parallelism
- **Megatron**: Optimized for very large models (billions of parameters)
- **DDP**: Simpler, better for smaller models

### vs. DeepSpeed

- **Megatron**: More control over parallelism strategies
- **DeepSpeed**: Automatic memory optimization with ZeRO
- **Megatron**: Better for models that fit in memory with parallelism
- **DeepSpeed**: Better for models that don't fit even with parallelism

### vs. FSDP

- **Megatron**: Tensor parallelism for intra-layer sharding
- **FSDP**: Parameter sharding across data parallel ranks
- **Megatron**: Pipeline parallelism support
- **FSDP**: Simpler API, PyTorch native

## References

- [Megatron-LM GitHub](https://github.com/NVIDIA/Megatron-LM)
- [Megatron-LM Documentation](https://github.com/NVIDIA/Megatron-LM/tree/main/docs)
- Original example: `resources/megatron-lm/examples/llama/train_llama3_8b_h100_fp8.sh`
