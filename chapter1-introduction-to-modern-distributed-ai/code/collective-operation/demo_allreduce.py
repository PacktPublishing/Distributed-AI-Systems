"""
Demo code for AllReduce collective operation using PyTorch distributed communication.

AllReduce: Performs reductions on data (for example, sum, min, max) across devices 
and stores the result in the receive buffer of every rank.

In a sum allreduce operation between k ranks, each rank will provide an array in of N values, 
and receive identical results in array out of N values, where out[i] = in0[i]+in1[i]+…+in(k-1)[i].

Usage:
    # Run with 2 processes
    OMP_NUM_THREADS=1 torchrun --nproc_per_node=2 demo_allreduce.py
    
    # Or with CPU (for testing without GPUs)
    OMP_NUM_THREADS=1 torchrun --nproc_per_node=2 demo_allreduce.py --use_cpu
"""

import torch
import torch.distributed as dist
import argparse
from mdaisy import init_distributed, sync_print

def demo_allreduce(rank, world_size, device):
    """
    AllReduce: Performs reduction (sum, min, max) across all ranks.
    Each rank receives the same result.
    """
    print(f"\n{'='*60}")
    print(f"Demo: AllReduce (Rank {rank})")
    print(f"{'='*60}")
    
    # Each rank has different input data
    tensor = torch.tensor([rank + 1, rank + 2, rank + 3], dtype=torch.float32, device=device)
    print(f"Rank {rank} input: {tensor}")
    
    # AllReduce with SUM operation
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    print(f"Rank {rank} after all_reduce(SUM): {tensor}")
    print(f"Expected sum: {sum(range(1, world_size + 1))} for each element")
    
    # AllReduce with MAX operation
    tensor = torch.tensor([rank + 1, rank + 2, rank + 3], dtype=torch.float32, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    print(f"Rank {rank} after all_reduce(MAX): {tensor}")
    
    # AllReduce with MIN operation
    tensor = torch.tensor([rank + 1, rank + 2, rank + 3], dtype=torch.float32, device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
    print(f"Rank {rank} after all_reduce(MIN): {tensor}")

def main():
    parser = argparse.ArgumentParser(description='PyTorch AllReduce Demo')
    parser.add_argument('--use_cpu', action='store_true', help='Use CPU instead of CUDA')
    args = parser.parse_args()
    
    rank, world_size, device, local_rank = init_distributed(args.use_cpu)
    
    print(f"\n{'#'*60}")
    print(f"PyTorch AllReduce Collective Operation Demo")
    print(f"Rank: {rank}, World Size: {world_size}, Device: {device}")
    print(f"{'#'*60}")
    
    # Synchronize before starting demo
    if world_size > 1:
        dist.barrier()
    
    try:
        demo_allreduce(rank, world_size, device)
        
        if world_size > 1:
            dist.barrier()
        
        print(f"\n{'#'*60}")
        print(f"Rank {rank}: AllReduce demo completed successfully!")
        print(f"{'#'*60}\n")
        
    except Exception as e:
        print(f"Error in rank {rank}: {e}")
        raise
    
    finally:
        if world_size > 1:
            dist.destroy_process_group()

if __name__ == '__main__':
    main()

