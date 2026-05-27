"""
Demo code for Broadcast collective operation using PyTorch distributed communication.

Broadcast: Copies an N-element buffer from the root rank to all the ranks.

Important note: The root argument is one of the ranks, not a device number, 
and is therefore impacted by a different rank to device mapping.

Usage:
    # Run with 2 processes
    OMP_NUM_THREADS=1 torchrun --nproc_per_node=2 demo_broadcast.py
    
    # Or with CPU (for testing without GPUs)
    OMP_NUM_THREADS=1 torchrun --nproc_per_node=2 demo_broadcast.py --use_cpu
"""

import torch
import torch.distributed as dist
import argparse
from mdaisy import init_distributed, sync_print

def demo_broadcast(rank, world_size, device):
    """
    Broadcast: Copies data from root rank to all other ranks.
    """
    print(f"\n{'='*60}")
    print(f"Demo: Broadcast (Rank {rank})")
    print(f"{'='*60}")
    
    root = 0
    if rank == root:
        # Root rank creates the data
        tensor = torch.tensor([10.0, 20.0, 30.0], dtype=torch.float32, device=device)
        print(f"Rank {rank} (root) broadcasting: {tensor}")
    else:
        # Other ranks start with zeros
        tensor = torch.zeros(3, dtype=torch.float32, device=device)
        print(f"Rank {rank} before broadcast: {tensor}")
    
    # Broadcast from root to all ranks
    dist.broadcast(tensor, src=root)
    print(f"Rank {rank} after broadcast: {tensor}")

def main():
    parser = argparse.ArgumentParser(description='PyTorch Broadcast Demo')
    parser.add_argument('--use_cpu', action='store_true', help='Use CPU instead of CUDA')
    args = parser.parse_args()
    
    rank, world_size, device, local_rank = init_distributed(args.use_cpu)
    
    print(f"\n{'#'*60}")
    print(f"PyTorch Broadcast Collective Operation Demo")
    print(f"Rank: {rank}, World Size: {world_size}, Device: {device}")
    print(f"{'#'*60}")
    
    # Synchronize before starting demo
    if world_size > 1:
        dist.barrier()
    
    try:
        demo_broadcast(rank, world_size, device)
        
        if world_size > 1:
            dist.barrier()
        
        print(f"\n{'#'*60}")
        print(f"Rank {rank}: Broadcast demo completed successfully!")
        print(f"{'#'*60}\n")
        
    except Exception as e:
        print(f"Error in rank {rank}: {e}")
        raise
    
    finally:
        if world_size > 1:
            dist.destroy_process_group()

if __name__ == '__main__':
    main()

