"""
Custom Training Benchmark with phase-level breakdown.

This class measures each training phase separately—data loading, forward pass,
backward pass, and optimizer step—enabling you to pinpoint exactly where time
is spent.

Why separate phase timing matters:
- If backward pass takes 3x longer than forward, check gradient computation
- If data loading dominates, increase DataLoader workers or use faster storage
- Without phase-level breakdown, you're optimizing blind
"""
import time
import torch
import numpy as np
from collections import defaultdict


class TrainingBenchmark:
    """Benchmark training with per-phase timing breakdown."""
    
    def __init__(self, model, dataloader, optimizer, criterion):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.metrics = defaultdict(list)
    
    def _training_step(self, data, target):
        """Execute a single training step."""
        data, target = data.cuda(), target.cuda()
        output = self.model(data)
        loss = self.criterion(output, target)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return loss
    
    def benchmark_iteration(self, warmup=True):
        """
        Benchmark a single training iteration with phase breakdown.
        
        Returns dict with timing for each phase:
            - data_loading: Time to move data to GPU
            - forward: Forward pass time
            - backward: Backward pass time
            - optimizer: Optimizer step time
            - total: Sum of compute phases (excludes data loading)
        """
        if warmup:
            data, target = next(iter(self.dataloader))
            _ = self._training_step(data, target)
            torch.cuda.synchronize()
        
        data, target = next(iter(self.dataloader))
        
        # Data loading time
        data_start = time.time()
        data, target = data.cuda(), target.cuda()
        torch.cuda.synchronize()
        data_time = time.time() - data_start
        
        # Forward pass
        forward_start = time.time()
        output = self.model(data)
        loss = self.criterion(output, target)
        torch.cuda.synchronize()
        forward_time = time.time() - forward_start
        
        # Backward pass
        backward_start = time.time()
        loss.backward()
        torch.cuda.synchronize()
        backward_time = time.time() - backward_start
        
        # Optimizer step
        optimizer_start = time.time()
        self.optimizer.step()
        self.optimizer.zero_grad()
        torch.cuda.synchronize()
        optimizer_time = time.time() - optimizer_start
        
        return {
            'data_loading': data_time,
            'forward': forward_time,
            'backward': backward_time,
            'optimizer': optimizer_time,
            'total': forward_time + backward_time + optimizer_time
        }
    
    def benchmark(self, num_warmup=10, num_iterations=100):
        """
        Run full benchmark with statistics.
        
        Returns dict with statistics for each phase:
            - mean: Average time
            - std: Standard deviation
            - p50: Median (50th percentile)
            - p95: 95th percentile
            - p99: 99th percentile
        """
        # Warmup phase
        for _ in range(num_warmup):
            self.benchmark_iteration(warmup=False)
        
        # Measurement phase
        for _ in range(num_iterations):
            metrics = self.benchmark_iteration(warmup=False)
            for key, value in metrics.items():
                self.metrics[key].append(value)
        
        # Calculate statistics
        stats = {}
        for key, values in self.metrics.items():
            stats[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'p50': np.percentile(values, 50),
                'p95': np.percentile(values, 95),
                'p99': np.percentile(values, 99)
            }
        return stats
    
    def print_results(self, stats):
        """Print benchmark results in a readable format."""
        print("\nTraining Benchmark Results:")
        print("-" * 60)
        print(f"{'Phase':<15} {'Mean':>10} {'Std':>10} {'P95':>10} {'P99':>10}")
        print("-" * 60)
        for phase in ['data_loading', 'forward', 'backward', 'optimizer', 'total']:
            if phase in stats:
                s = stats[phase]
                print(f"{phase:<15} {s['mean']:>10.4f} {s['std']:>10.4f} "
                      f"{s['p95']:>10.4f} {s['p99']:>10.4f}")
        print("-" * 60)


# Example interpretation:
# Phase         Mean      P95       P99
# data_loading  0.005s    0.008s    0.012s
# forward       0.045s    0.048s    0.052s
# backward      0.065s    0.070s    0.078s
# optimizer     0.015s    0.018s    0.022s
#
# If data_loading exceeds 10% of total time, increase DataLoader workers.
# If backward is more than 2x forward, check for gradient checkpointing
# opportunities or memory fragmentation.
# High variance (large gap between mean and P99) indicates system instability.
