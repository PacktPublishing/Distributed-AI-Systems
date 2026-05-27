"""
Demo code for AlltoAll collective operation using PyTorch distributed communication.

AlltoAll: In an AlltoAll operation between k ranks, each rank provides an input buffer 
of size k*N values, where the j-th chunk of N values is sent to destination rank j. 
Each rank receives an output buffer of size k*N values, where the i-th chunk of N values 
comes from source rank i.

Usage:
    # Run with 2 processes
    OMP_NUM_THREADS=1 torchrun --nproc_per_node=2 demo_alltoall.py
    
    # Or with CPU (for testing without GPUs)
    OMP_NUM_THREADS=1 torchrun --nproc_per_node=2 demo_alltoall.py --use_cpu
"""

import torch
import torch.distributed as dist
import argparse
import sys
from mdaisy import init_distributed, sync_print

def demo_alltoall(rank, world_size, device):
    """
    AlltoAll: Each rank sends different data to every other rank.
    Input size: world_size * N, Output size: world_size * N
    """
    print(f"\n{'='*60}")
    print(f"Demo: AlltoAll (Rank {rank})")
    print(f"{'='*60}")
    
    # Each rank prepares data to send to each other rank
    # Input: world_size chunks of size N (N=2 for this example)
    input_list = []
    for dst_rank in range(world_size):
        # Rank sends [rank*100 + dst_rank*10 + 1, rank*100 + dst_rank*10 + 2] to dst_rank
        chunk = torch.tensor([rank * 100 + dst_rank * 10 + 1, rank * 100 + dst_rank * 10 + 2], 
                            dtype=torch.float32, device=device)
        input_list.append(chunk)
    
    input_tensor = torch.cat(input_list)
    print(f"Rank {rank} input: {input_tensor}")
    
    # AlltoAll: each rank receives data from all ranks
    output_list = [torch.zeros(2, dtype=torch.float32, device=device) for _ in range(world_size)]
    dist.all_to_all(output_list, input_list)
    
    output_tensor = torch.cat(output_list)
    print(f"Rank {rank} after all_to_all: {output_tensor}")
    print(f"Expected: receives [src_rank*100 + rank*10 + 1, src_rank*100 + rank*10 + 2] from each src_rank")

def main():
    parser = argparse.ArgumentParser(description='PyTorch AlltoAll Demo')
    parser.add_argument('--use_cpu', action='store_true', help='Use CPU instead of CUDA')
    args = parser.parse_args()
    
    rank, world_size, device, local_rank = init_distributed(args.use_cpu)
    
    # Check if backend supports alltoall BEFORE any other output
    if world_size > 1:
        backend = dist.get_backend()
        if backend == 'gloo':
            # Synchronize all ranks before printing error
            dist.barrier()
            error_msg = (
                f"\n{'='*60}\n"
                f"ERROR: Backend 'gloo' does not support alltoall operation!\n"
                f"{'='*60}\n"
                f"Solutions:\n"
                f"  1. Use GPUs (remove --use_cpu flag):\n"
                f"     OMP_NUM_THREADS=1 torchrun --nproc_per_node=4 demo_alltoall.py\n"
                f"  2. Install MPI and use MPI backend (if available)\n"
                f"  3. Use a different collective operation that gloo supports\n"
                f"{'='*60}\n"
            )
            if rank == 0:
                print(error_msg)
            # Clean exit: all processes synchronized, exit with 0 to avoid torchrun error reporting
            dist.destroy_process_group()
            sys.exit(0)
        dist.barrier()
    
    print(f"\n{'#'*60}")
    print(f"PyTorch AlltoAll Collective Operation Demo")
    print(f"Rank: {rank}, World Size: {world_size}, Device: {device}")
    print(f"{'#'*60}")
    
    try:
        demo_alltoall(rank, world_size, device)
        
        if world_size > 1:
            dist.barrier()
        
        print(f"\n{'#'*60}")
        print(f"Rank {rank}: AlltoAll demo completed successfully!")
        print(f"{'#'*60}\n")
        
    except Exception as e:
        print(f"Error in rank {rank}: {e}")
        raise
    
    finally:
        if world_size > 1:
            dist.destroy_process_group()

if __name__ == '__main__':
    main()

