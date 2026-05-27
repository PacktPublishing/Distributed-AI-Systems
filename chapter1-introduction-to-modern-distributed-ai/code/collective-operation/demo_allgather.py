"""
Demo code for AllGather collective operation using PyTorch distributed communication.

AllGather: Gathers N values from k ranks into an output buffer of size k*N, 
and distributes that result to all ranks.

The output is ordered by the rank index. The AllGather operation is therefore 
impacted by a different rank to device mapping.

Note: Executing ReduceScatter, followed by AllGather, is equivalent to the AllReduce operation.

Usage:
    # Run with 2 processes
    OMP_NUM_THREADS=1 torchrun --nproc_per_node=2 demo_allgather.py
    
    # Or with CPU (for testing without GPUs)
    OMP_NUM_THREADS=1 torchrun --nproc_per_node=2 demo_allgather.py --use_cpu
"""

import torch
import torch.distributed as dist
import argparse
from mdaisy import init_distributed, sync_print

def demo_allgather(rank, world_size, device):
    """
    AllGather: Gathers data from all ranks and distributes to all ranks.
    Output is ordered by rank index.
    """
    print(f"\n{'='*60}")
    print(f"Demo: AllGather (Rank {rank})")
    print(f"{'='*60}")
    
    # Each rank has different input (N values)
    input_tensor = torch.tensor([rank * 10 + 1, rank * 10 + 2], dtype=torch.float32, device=device)
    print(f"Rank {rank} input: {input_tensor}")
    
    # AllGather: output size is world_size * N
    output_list = [torch.zeros_like(input_tensor) for _ in range(world_size)]
    dist.all_gather(output_list, input_tensor)
    
    # Concatenate the results
    output_tensor = torch.cat(output_list)
    print(f"Rank {rank} after all_gather: {output_tensor}")
    print(f"Expected: concatenation of [rank*10+1, rank*10+2] for all ranks")

def main():
    parser = argparse.ArgumentParser(description='PyTorch AllGather Demo')
    parser.add_argument('--use_cpu', action='store_true', help='Use CPU instead of CUDA')
    args = parser.parse_args()
    
    rank, world_size, device, local_rank = init_distributed(args.use_cpu)
    
    print(f"\n{'#'*60}")
    print(f"PyTorch AllGather Collective Operation Demo")
    print(f"Rank: {rank}, World Size: {world_size}, Device: {device}")
    print(f"{'#'*60}")
    
    # Synchronize before starting demo
    # Device is already set, so barrier will use the correct device context
    if world_size > 1:
        dist.barrier()
    
    try:
        demo_allgather(rank, world_size, device)
        
        if world_size > 1:
            dist.barrier()
        
        print(f"\n{'#'*60}")
        print(f"Rank {rank}: AllGather demo completed successfully!")
        print(f"{'#'*60}\n")
        
    except Exception as e:
        print(f"Error in rank {rank}: {e}")
        raise
    
    finally:
        if world_size > 1:
            dist.destroy_process_group()

if __name__ == '__main__':
    main()

