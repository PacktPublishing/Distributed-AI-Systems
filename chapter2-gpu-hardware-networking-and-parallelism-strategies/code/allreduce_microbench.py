"""
AllReduce micro-benchmark: measures effective collective bandwidth across GPUs.

Reports two numbers, as NCCL-tests does:
  * Algorithm bandwidth: n * size / time   -- useful throughput seen by the caller
  * Bus bandwidth:       alg_bw * 2*(n-1)/n -- data actually crossing the interconnect
                                              for a ring AllReduce (NCCL may pick tree)

Usage:
    OMP_NUM_THREADS=1 torchrun --nproc_per_node=2 allreduce_microbench.py
    OMP_NUM_THREADS=1 torchrun --nproc_per_node=2 allreduce_microbench.py --size-mb 100
"""

import argparse
import os
import time

import torch
import torch.distributed as dist


def allreduce_microbench(size_mb=100, iterations=50, warmup=10):
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl", device_id=torch.device(f"cuda:{local_rank}")
    )
    rank, world_size = dist.get_rank(), dist.get_world_size()

    size_bytes = size_mb * 1024 * 1024
    tensor = torch.ones(size_bytes // 4, device=f"cuda:{local_rank}")

    for _ in range(warmup):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    dist.barrier()

    start = time.time()
    for _ in range(iterations):
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize()
    elapsed = time.time() - start

    n = world_size
    alg_bw = size_bytes * n * iterations / (1024**3) / elapsed
    bus_bw = alg_bw * 2 * (n - 1) / n

    if rank == 0:
        print("AllReduce Benchmark Results")
        print(f"World size: {n} GPUs")
        print(f"Tensor size: {size_mb} MB per GPU")
        print(f"Iterations: {iterations}")
        print(f"Total time: {elapsed:.3f} seconds")
        print(f"Algorithm bandwidth: {alg_bw:.2f} GB/s")
        print(f"Bus bandwidth (ring estimate): {bus_bw:.2f} GB/s")

    dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--size-mb", type=int, default=100)
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    a = p.parse_args()
    allreduce_microbench(a.size_mb, a.iterations, a.warmup)
