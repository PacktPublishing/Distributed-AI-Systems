"""
Check CUDA availability and GPU information.

This script displays basic GPU information including memory, PCIe info,
and helps verify CUDA setup.

Usage:
    python check_cuda.py
"""

import torch

def check_cuda():
    """Check and display CUDA/GPU information."""
    print("=" * 60)
    print("CUDA and GPU Information")
    print("=" * 60)
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("\nCUDA is not available. Make sure you have:")
        print("  1. NVIDIA GPU installed")
        print("  2. CUDA drivers installed")
        print("  3. PyTorch with CUDA support installed")
        return
    
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print()
    
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        vram_gb = props.total_memory / (1024**3)
        print(f"GPU {i}: {props.name}")
        print(f"  Total memory: {vram_gb:.1f} GB")
        print(f"  Compute capability: {props.major}.{props.minor}")
        print(f"  Multiprocessors: {props.multi_processor_count}")
        print()
    
    print("=" * 60)
    print("Note: For detailed PCIe and NVLink topology, run:")
    print("  nvidia-smi topo -m")
    print("  nvidia-smi --query-gpu=name,memory.total,pcie.link.gen.max,pcie.link.width.max --format=csv")
    print("=" * 60)

if __name__ == '__main__':
    check_cuda()
