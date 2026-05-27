# Building a Wheel for megatron.training

This directory contains a script to build a standalone wheel package that includes `megatron.training` (which is not included in the official PyPI `megatron-core` package).

## Quick Start

```bash
# Make sure you're in the megatron directory
cd chapter8-running-distributed-training-with-slurm/code/megatron

# Build the wheel (uses ../../resources/megatron-lm by default)
./build_wheel.sh

# Or specify a custom Megatron-LM source directory
./build_wheel.sh /path/to/Megatron-LM

# Install the wheel
pip install wheels/megatron_core_with_training-*.whl[mlm,dev]
```

## What the Script Does

1. **Copies source files**: Creates a temporary build directory and copies the necessary files from the Megatron-LM repository
2. **Modifies pyproject.toml**: Updates the package configuration to include `megatron.training` and `megatron.training.*`
3. **Builds the wheel**: Uses `python -m build` to create a wheel package
4. **Saves the wheel**: Places the built wheel in the `wheels/` directory

## Requirements

- Python 3.10+
- `pip`, `setuptools`, `wheel`, `build` (installed automatically by the script)
- Access to the Megatron-LM source code (default: `../../resources/megatron-lm`)

## Output

After building, you'll find the wheel in:
```
wheels/megatron_core_with_training-<version>-py3-none-any.whl
```

## Installation

Install the wheel on any machine:

```bash
# Basic installation
pip install wheels/megatron_core_with_training-*.whl

# With optional dependencies (recommended)
pip install wheels/megatron_core_with_training-*.whl[mlm,dev]
```

## Verification

After installation, verify that `megatron.training` is available:

```python
python3 -c "from megatron.training import get_args; print('SUCCESS: megatron.training is available')"
```

## Advantages of Using a Wheel

- ✅ **No source code needed**: Once built, you can install the wheel on any machine without the source repository
- ✅ **Faster installation**: Wheels are pre-built and install much faster than building from source
- ✅ **Portable**: Can be shared, distributed, or used in production environments
- ✅ **Reproducible**: Same wheel produces the same installation every time

## Troubleshooting

### Error: "Megatron-LM source directory not found"

Make sure the source directory exists. You can specify it explicitly:
```bash
./build_wheel.sh /path/to/Megatron-LM
```

### Error: "pyproject.toml not found"

The script expects a standard Megatron-LM repository structure. Make sure you're pointing to the root of the repository.

### Build fails with import errors

Make sure you have the required build dependencies:
```bash
pip install --upgrade pip setuptools wheel build
```

## Manual Build (Alternative)

If you prefer to build manually:

1. Copy the Megatron-LM source to a temporary directory
2. Modify `pyproject.toml` to include:
   ```toml
   [tool.setuptools.packages.find]
   include = [
       "megatron.core",
       "megatron.core.*",
       "megatron.training",
       "megatron.training.*",
   ]
   ```
3. Build:
   ```bash
   python -m build --wheel
   ```

The `build_wheel.sh` script automates all of this for you.
