"""
Scaling Bottleneck Analysis and Optimization Strategies.

Once you know your scaling efficiency is poor, use these tools to identify
why and apply the appropriate fix.
"""
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader


def analyze_scaling_bottlenecks(metrics_1gpu, metrics_ngpu, n):
    """
    Analyze what's limiting scaling.
    
    Common patterns:
    - "Data loading not scaling well": DataLoader can't keep up with multiple
      GPUs. Increase num_workers or use faster storage.
    - "High communication overhead": Network is the bottleneck. Consider
      gradient compression, larger batch sizes, or better interconnect.
    - "Low GPU utilization": GPUs are waiting for something—usually data
      or synchronization.
    """
    bottlenecks = []
    
    # Check data loading
    if metrics_ngpu['data_loading'] > metrics_1gpu['data_loading'] * 1.5:
        bottlenecks.append("Data loading not scaling well")
    
    # Check communication
    comm_overhead = metrics_ngpu['communication'] / metrics_ngpu['total']
    if comm_overhead > 0.3:
        bottlenecks.append(f"High communication overhead: {comm_overhead*100:.1f}%")
    
    # Check compute utilization
    gpu_util = metrics_ngpu['gpu_utilization']
    if gpu_util < 0.8:
        bottlenecks.append(f"Low GPU utilization: {gpu_util*100:.1f}%")
    
    return bottlenecks


def create_optimized_ddp_model(model, rank):
    """
    Create DDP model with optimized gradient bucketing.
    
    DDP uses gradient bucketing to overlap backward computation with gradient
    synchronization. Tuning bucket size can improve overlap:
    - Smaller buckets start communication earlier but have more overhead
    - Larger buckets have less overhead but delay communication
    - Start with 25MB and tune based on profiling
    """
    return DDP(
        model,
        device_ids=[rank],
        bucket_cap_mb=25,  # Tune bucket size
        find_unused_parameters=False
    )


def create_optimized_dataloader(dataset, batch_size):
    """
    Create DataLoader optimized for distributed training.
    
    Data loading often becomes the bottleneck when scaling to many GPUs.
    Each GPU needs data fast enough to keep compute saturated.
    
    - num_workers: Set to 2-4x your CPU cores per GPU
    - pin_memory: Enables faster CPU→GPU transfers
    - prefetch_factor: Controls how many batches each worker prefetches
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=4,        # Parallel data loading
        pin_memory=True,      # Faster H2D transfer
        prefetch_factor=2     # Prefetch batches
    )


def train_with_gradient_accumulation(model, dataloader, criterion, optimizer, 
                                     accumulation_steps=4):
    """
    Training loop with gradient accumulation.
    
    If communication overhead is high, reduce synchronization frequency by
    accumulating gradients over multiple micro-batches.
    
    This effectively increases batch size by accumulation_steps without
    increasing memory usage. Gradients are synchronized only every
    accumulation_steps iterations, reducing communication overhead by
    the same factor.
    """
    for i, (data, target) in enumerate(dataloader):
        output = model(data)
        loss = criterion(output, target) / accumulation_steps
        loss.backward()
        
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
