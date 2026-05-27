# Building a Package for megatron.training

## Quick Start: Build a Wheel

**Easiest way**: Use the provided build script:

```bash
cd /path/to/chapter8-running-distributed-training-with-slurm/code/megatron
./build_wheel.sh [MEGATRON_SOURCE_DIR]
```

If `MEGATRON_SOURCE_DIR` is not provided, it defaults to `../../resources/megatron-lm`.

The script will:
1. Copy the Megatron-LM source code
2. Modify `pyproject.toml` to include `megatron.training`
3. Build a wheel package
4. Save it to `wheels/` directory

Then install:
```bash
pip install wheels/megatron_core_with_training-*.whl[mlm,dev]
```

## Current Situation

The official `megatron-core` package from PyPI only includes `megatron.core` and `megatron.core.*`, but **NOT** `megatron.training`. 

Looking at `pyproject.toml`:
```toml
[tool.setuptools.packages.find]
include = ["megatron.core", "megatron.core.*"]
```

This means:
- ✅ `pip install megatron-core` → Only installs `megatron.core`
- ✅ `pip install --no-build-isolation .[mlm,dev]` (from source) → Installs everything including `megatron.training`

## Why megatron.training is Not in PyPI Package

`megatron.training` contains training scripts and utilities that are typically used with the full Megatron-LM repository. The PyPI package (`megatron-core`) focuses on the core library for building custom training frameworks, while `megatron.training` provides reference implementations.

## Solution: Build a Package with megatron.training

The official `pyproject.toml` only includes `megatron.core`:

```toml
[tool.setuptools.packages.find]
include = ["megatron.core", "megatron.core.*"]
```

To include `megatron.training`, you need to modify it:

### Option 1: Modify pyproject.toml to Include megatron.training

Edit `pyproject.toml` in the Megatron-LM repository:

```toml
[tool.setuptools.packages.find]
include = [
    "megatron.core",
    "megatron.core.*",
    "megatron.training",
    "megatron.training.*",
    "megatron.legacy",
    "megatron.legacy.*",
]
```

Then install:
```bash
cd /path/to/megatron-lm
pip install --no-build-isolation .[mlm,dev]
```

This will create a package that includes both `megatron.core` and `megatron.training`.

### Option 2: Install from Source (Easiest)

The simplest way to get `megatron.training` is to install from source:

```bash
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
pip install --no-build-isolation .[mlm,dev]
```

This installs everything including `megatron.training`.

### Option 3: Create a Custom Package

If you want to create a separate package for `megatron.training`, you would need to:

1. Create a new `pyproject.toml` that includes only `megatron.training`:
```toml
[project]
name = "megatron-training"
version = "0.12.0"
description = "Megatron Training - Training utilities and scripts"
dependencies = ["megatron-core", "torch", "numpy"]

[tool.setuptools.packages.find]
include = [
    "megatron.training",
    "megatron.training.*",
]
```

2. Build and install:
```bash
pip install --no-build-isolation .
```

However, this is **not recommended** because:
- `megatron.training` has dependencies on `megatron.core`
- It's designed to work with the full repository structure
- The official way is to install from source

## For This Example

Since we've copied `pretrain_gpt.py`, `gpt_builders.py`, and `model_provider.py` to this directory, you have two options:

### Option A: Install megatron-core from PyPI + Use Local Scripts

```bash
pip install --no-build-isolation megatron-core[mlm,dev]
# The scripts in this directory will work
```

**Note**: This won't work because `pretrain_gpt.py` imports from `megatron.training`, which is not in the PyPI package.

### Option B: Install from Source (Recommended)

```bash
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
pip install --no-build-isolation .[mlm,dev]
```

This installs `megatron.training` and everything else needed.

## Conclusion

**Answer**: Yes, you can build a wheel package for `megatron.training`!

### Recommended: Build a Wheel

Use the provided `build_wheel.sh` script to create a standalone wheel:

```bash
./build_wheel.sh
pip install wheels/megatron_core_with_training-*.whl[mlm,dev]
```

This creates a wheel that includes both `megatron.core` and `megatron.training`, which you can:
- Install on any machine without needing the source code
- Share with others
- Use in production environments

### Alternative: Install from Source

If you prefer to install from source:

```bash
conda activate research
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
pip install --no-build-isolation .[mlm,dev]
```

**Note**: The wheel approach is better because:
- ✅ No need to keep source code around
- ✅ Faster installation
- ✅ Can be shared/distributed
- ✅ Works in environments where git cloning is not possible
