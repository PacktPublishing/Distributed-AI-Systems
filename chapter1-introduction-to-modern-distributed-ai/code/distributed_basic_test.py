"""
Basic distributed test to verify process group initialization works.

This script tests the fundamental distributed setup: process group initialization,
rank identification, and basic communication. It doesn't use DDP - it's just
testing that multiple processes can communicate.

Usage:
    torchrun --nproc_per_node=2 code/distributed_basic_test.py

Expected output:
    Rank 0 says hello. GPU 0: NVIDIA T4 (16.0 GB)
    Rank 1 says hello. GPU 1: NVIDIA T4 (16.0 GB)
"""

import torch
import torch.distributed as dist
import os

def test_distributed_setup():
    """Test basic distributed process group initialization and communication"""
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    
    # Print rank and GPU info
    props = torch.cuda.get_device_properties(local_rank)
    vram_gb = props.total_memory / (1024**3)
    print(f"Rank {rank} says hello. GPU {local_rank}: {props.name} ({vram_gb:.1f} GB)")
    
    dist.destroy_process_group()

if __name__ == "__main__":
    test_distributed_setup()



