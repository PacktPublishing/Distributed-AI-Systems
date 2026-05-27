"""
Demo code for ReduceScatter collective operation using PyTorch distributed communication.

ReduceScatter: Performs the same operation as Reduce, except that the result is 
scattered in equal-sized blocks between ranks, each rank getting a chunk of data 
based on its rank index.

The ReduceScatter operation is impacted by a different rank to device mapping 
since the ranks determine the data layout.

Usage:
    # Run with 2 processes
    OMP_NUM_THREADS=1 torchrun --nproc_per_node=2 demo_reducescatter.py
    
    # Or with CPU (for testing without GPUs)
    OMP_NUM_THREADS=1 torchrun --nproc_per_node=2 demo_reducescatter.py --use_cpu
"""

import torch
import torch.distributed as dist
import argparse
from mdaisy import init_distributed, sync_print

def demo_reducescatter(rank, world_size, device):
    """
    ReduceScatter: Reduces data across ranks and scatters the result in chunks.
    Each rank receives a chunk based on its rank index.
    """
    print(f"\n{'='*60}")
    print(f"Demo: ReduceScatter (Rank {rank})")
    print(f"{'='*60}")
    
    # Each rank has input of size world_size * N
    # For simplicity, let N=2, so input size is world_size * 2
    input_list = [torch.tensor([rank * 10 + i, rank * 10 + i + 1], dtype=torch.float32, device=device) 
                  for i in range(world_size)]
    input_tensor = torch.cat(input_list)
    print(f"Rank {rank} input: {input_tensor}")
    
    # ReduceScatter: each rank gets a chunk of size N
    output_tensor = torch.zeros(2, dtype=torch.float32, device=device)
    dist.reduce_scatter(output_tensor, input_list, op=dist.ReduceOp.SUM)
    
    print(f"Rank {rank} after reduce_scatter: {output_tensor}")
    print(f"Expected: sum of i-th chunk from all ranks, where i=rank")

def main():
    parser = argparse.ArgumentParser(description='PyTorch ReduceScatter Demo')
    parser.add_argument('--use_cpu', action='store_true', help='Use CPU instead of CUDA')
    args = parser.parse_args()
    
    rank, world_size, device, local_rank = init_distributed(args.use_cpu)
    
    print(f"\n{'#'*60}")
    print(f"PyTorch ReduceScatter Collective Operation Demo")
    print(f"Rank: {rank}, World Size: {world_size}, Device: {device}")
    print(f"{'#'*60}")
    
    # Synchronize before starting demo
    if world_size > 1:
        dist.barrier()
    
    try:
        demo_reducescatter(rank, world_size, device)
        
        if world_size > 1:
            dist.barrier()
        
        print(f"\n{'#'*60}")
        print(f"Rank {rank}: ReduceScatter demo completed successfully!")
        print(f"{'#'*60}\n")
        
    except Exception as e:
        print(f"Error in rank {rank}: {e}")
        raise
    
    finally:
        if world_size > 1:
            dist.destroy_process_group()

if __name__ == '__main__':
    main()

