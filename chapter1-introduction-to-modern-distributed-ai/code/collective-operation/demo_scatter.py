"""
Demo code for Scatter collective operation using PyTorch distributed communication.

Scatter: Distributes a total of N*k values from the root rank to k ranks, 
each rank receiving N values.

Important note: The root argument is one of the ranks, not a device number, 
and is therefore impacted by a different rank to device mapping.

Usage:
    # Run with 2 processes
    OMP_NUM_THREADS=1 torchrun --nproc_per_node=2 demo_scatter.py
    
    # Or with CPU (for testing without GPUs)
    OMP_NUM_THREADS=1 torchrun --nproc_per_node=2 demo_scatter.py --use_cpu
"""

import torch
import torch.distributed as dist
import argparse
from mdaisy import init_distributed, sync_print

def demo_scatter(rank, world_size, device):
    """
    Scatter: Root rank distributes data to all ranks.
    Each rank receives a chunk from root.
    """
    print(f"\n{'='*60}")
    print(f"Demo: Scatter (Rank {rank})")
    print(f"{'='*60}")
    
    root = 0
    if rank == root:
        # Root rank prepares data for all ranks
        scatter_list = [torch.tensor([r * 10 + 1, r * 10 + 2], dtype=torch.float32, device=device) 
                       for r in range(world_size)]
        scatter_tensor = torch.cat(scatter_list)
        print(f"Rank {rank} (root) scattering: {scatter_tensor}")
    
    # Each rank receives a chunk
    output_tensor = torch.zeros(2, dtype=torch.float32, device=device)
    
    if rank == root:
        dist.scatter(output_tensor, scatter_list=scatter_list, src=root)
    else:
        dist.scatter(output_tensor, scatter_list=None, src=root)
    
    print(f"Rank {rank} after scatter: {output_tensor}")
    print(f"Expected: [rank*10 + 1, rank*10 + 2]")

def main():
    parser = argparse.ArgumentParser(description='PyTorch Scatter Demo')
    parser.add_argument('--use_cpu', action='store_true', help='Use CPU instead of CUDA')
    args = parser.parse_args()
    
    rank, world_size, device, local_rank = init_distributed(args.use_cpu)
    
    print(f"\n{'#'*60}")
    print(f"PyTorch Scatter Collective Operation Demo")
    print(f"Rank: {rank}, World Size: {world_size}, Device: {device}")
    print(f"{'#'*60}")
    
    # Synchronize before starting demo
    if world_size > 1:
        dist.barrier()
    
    try:
        demo_scatter(rank, world_size, device)
        
        if world_size > 1:
            dist.barrier()
        
        print(f"\n{'#'*60}")
        print(f"Rank {rank}: Scatter demo completed successfully!")
        print(f"{'#'*60}\n")
        
    except Exception as e:
        print(f"Error in rank {rank}: {e}")
        raise
    
    finally:
        if world_size > 1:
            dist.destroy_process_group()

if __name__ == '__main__':
    main()

