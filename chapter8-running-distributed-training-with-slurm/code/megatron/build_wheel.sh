#!/bin/bash
# Build a wheel package for megatron-core that includes megatron.training
#
# Usage:
#   ./build_wheel.sh [MEGATRON_SOURCE_DIR]
#
# If MEGATRON_SOURCE_DIR is not provided, it defaults to:
#   ../../resources/megatron-lm

set -e

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEGATRON_SOURCE_DIR="${1:-${SCRIPT_DIR}/../../resources/megatron-lm}"

if [ ! -d "$MEGATRON_SOURCE_DIR" ]; then
    echo "ERROR: Megatron-LM source directory not found: $MEGATRON_SOURCE_DIR"
    echo "Please provide the path to the Megatron-LM repository"
    exit 1
fi

echo "=========================================="
echo "Building wheel for megatron-core with megatron.training"
echo "=========================================="
echo "Source directory: $MEGATRON_SOURCE_DIR"
echo ""

# Check if pyproject.toml exists
if [ ! -f "$MEGATRON_SOURCE_DIR/pyproject.toml" ]; then
    echo "ERROR: pyproject.toml not found in $MEGATRON_SOURCE_DIR"
    exit 1
fi

# Create a temporary directory for building
BUILD_DIR="${SCRIPT_DIR}/build_wheel_temp"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

echo "Step 1: Copying source files..."
cp -r "$MEGATRON_SOURCE_DIR/megatron" "$BUILD_DIR/"
cp "$MEGATRON_SOURCE_DIR/pyproject.toml" "$BUILD_DIR/"
cp "$MEGATRON_SOURCE_DIR/setup.py" "$BUILD_DIR/" 2>/dev/null || true
cp "$MEGATRON_SOURCE_DIR/README.md" "$BUILD_DIR/" 2>/dev/null || true
cp "$MEGATRON_SOURCE_DIR/MANIFEST.in" "$BUILD_DIR/" 2>/dev/null || true

# Copy any required files for building
if [ -d "$MEGATRON_SOURCE_DIR/megatron/core" ]; then
    echo "  ✓ Copied megatron/ directory"
fi

echo ""
echo "Step 2: Modifying pyproject.toml to include megatron.training..."

# Create modified pyproject.toml using Python
python3 - "$BUILD_DIR" << 'PYTHON_SCRIPT'
import sys
import os
import re

build_dir = sys.argv[1]
pyproject_path = os.path.join(build_dir, "pyproject.toml")

with open(pyproject_path, 'r') as f:
    content = f.read()

# Find the packages.find section and modify it
pattern = r'(\[tool\.setuptools\.packages\.find\]\s+include\s*=\s*\[)([^\]]+)(\])'

def replace_packages(match):
    includes = match.group(2)
    # Check if megatron.training is already included
    if 'megatron.training' in includes:
        return match.group(0)
    
    # Add megatron.training and megatron.training.*
    new_includes = includes.rstrip()
    if not new_includes.endswith(','):
        new_includes += ','
    new_includes += '\n    "megatron.training",\n    "megatron.training.*",'
    
    return match.group(1) + new_includes + match.group(3)

new_content = re.sub(pattern, replace_packages, content, flags=re.MULTILINE)

# Also update the project name to indicate it includes training
new_content = re.sub(
    r'(name\s*=\s*")megatron-core(")',
    r'\1megatron-core-with-training\2',
    new_content
)

# Update description
new_content = re.sub(
    r'(description\s*=\s*")Megatron Core[^"]*(")',
    r'\1Megatron Core with Training - includes megatron.core and megatron.training\2',
    new_content
)

with open(pyproject_path, 'w') as f:
    f.write(new_content)

print("  ✓ Modified pyproject.toml to include megatron.training")
PYTHON_SCRIPT

echo ""
echo "Step 3: Installing build dependencies..."
cd "$BUILD_DIR"
pip install --quiet --upgrade pip setuptools wheel build

echo ""
echo "Step 4: Building wheel..."
python3 -m build --wheel

# Find the built wheel
WHEEL_FILE=$(find "$BUILD_DIR/dist" -name "*.whl" | head -1)

if [ -z "$WHEEL_FILE" ]; then
    echo "ERROR: Wheel file not found after building"
    exit 1
fi

# Copy wheel to script directory
OUTPUT_DIR="${SCRIPT_DIR}/wheels"
mkdir -p "$OUTPUT_DIR"
cp "$WHEEL_FILE" "$OUTPUT_DIR/"
WHEEL_NAME=$(basename "$WHEEL_FILE")

echo ""
echo "=========================================="
echo "Build completed successfully!"
echo "=========================================="
echo "Wheel file: $OUTPUT_DIR/$WHEEL_NAME"
echo ""
echo "To install the wheel:"
echo "  pip install $OUTPUT_DIR/$WHEEL_NAME"
echo ""
echo "Or install with optional dependencies:"
echo "  pip install $OUTPUT_DIR/$WHEEL_NAME[mlm,dev]"
echo ""

# Cleanup (optional - comment out if you want to keep the build directory)
# echo "Cleaning up temporary build directory..."
# rm -rf "$BUILD_DIR"
