"""
Demo code for Gather collective operation using PyTorch distributed communication.

Gather: Gathers N values from k ranks into an output buffer on the root rank of size k*N.

Important note: The root argument is one of the ranks, not a device number, 
and is therefore impacted by a different rank to device mapping.

Usage:
    # Run with 2 processes
    OMP_NUM_THREADS=1 torchrun --nproc_per_node=2 demo_gather.py
    
    # Or with CPU (for testing without GPUs)
    OMP_NUM_THREADS=1 torchrun --nproc_per_node=2 demo_gather.py --use_cpu
"""

import torch
import torch.distributed as dist
import argparse
from mdaisy import init_distributed, sync_print

def demo_gather(rank, world_size, device):
    """
    Gather: Gathers data from all ranks to the root rank.
    Only root receives the gathered data.
    """
    print(f"\n{'='*60}")
    print(f"Demo: Gather (Rank {rank})")
    print(f"{'='*60}")
    
    root = 0
    # Each rank has different input
    input_tensor = torch.tensor([rank * 10 + 1, rank * 10 + 2], dtype=torch.float32, device=device)
    print(f"Rank {rank} input: {input_tensor}")
    
    if rank == root:
        # Root rank prepares output list
        output_list = [torch.zeros_like(input_tensor) for _ in range(world_size)]
        dist.gather(input_tensor, output_list, dst=root)
        output_tensor = torch.cat(output_list)
        print(f"Rank {rank} (root) after gather: {output_tensor}")
    else:
        dist.gather(input_tensor, None, dst=root)
        print(f"Rank {rank} after gather (no output): input unchanged")

def main():
    parser = argparse.ArgumentParser(description='PyTorch Gather Demo')
    parser.add_argument('--use_cpu', action='store_true', help='Use CPU instead of CUDA')
    args = parser.parse_args()
    
    rank, world_size, device, local_rank = init_distributed(args.use_cpu)
    
    print(f"\n{'#'*60}")
    print(f"PyTorch Gather Collective Operation Demo")
    print(f"Rank: {rank}, World Size: {world_size}, Device: {device}")
    print(f"{'#'*60}")
    
    # Synchronize before starting demo
    if world_size > 1:
        dist.barrier()
    
    try:
        demo_gather(rank, world_size, device)
        
        if world_size > 1:
            dist.barrier()
        
        print(f"\n{'#'*60}")
        print(f"Rank {rank}: Gather demo completed successfully!")
        print(f"{'#'*60}\n")
        
    except Exception as e:
        print(f"Error in rank {rank}: {e}")
        raise
    
    finally:
        if world_size > 1:
            dist.destroy_process_group()

if __name__ == '__main__':
    main()

