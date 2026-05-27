"""
Benchmark with proper warmup to avoid cold-start measurement errors.

CUDA operations are lazily compiled—the first execution triggers JIT compilation,
memory allocation, and cache population. Without warmup, measurements include
these one-time costs, making results unreliable.
"""
import time
import torch
import numpy as np


def benchmark_with_warmup(model, dataloader, num_warmup=10, num_iterations=100):
    """
    Benchmark model inference with proper warmup.
    
    Args:
        model: PyTorch model to benchmark
        dataloader: DataLoader providing input batches
        num_warmup: Number of warmup iterations (discarded)
        num_iterations: Number of measurement iterations
    
    Returns:
        List of per-iteration timings in seconds
    """
    # Warmup: discard initial iterations
    # This triggers CUDA compilation and cache warming
    for i in range(num_warmup):
        _ = model(next(iter(dataloader)))
    
    # Synchronize before measurement to ensure warmup is complete
    torch.cuda.synchronize()
    
    # Actual measurement
    timings = []
    for i in range(num_iterations):
        start = time.time()
        _ = model(next(iter(dataloader)))
        # Synchronize after each iteration to ensure GPU work is done
        torch.cuda.synchronize()
        timings.append(time.time() - start)
    
    return timings


def report_statistics(timings):
    """Report benchmark statistics from timing data."""
    print(f"Mean: {np.mean(timings):.4f}s")
    print(f"Std:  {np.std(timings):.4f}s")
    print(f"P50:  {np.percentile(timings, 50):.4f}s")
    print(f"P95:  {np.percentile(timings, 95):.4f}s")
    print(f"P99:  {np.percentile(timings, 99):.4f}s")
