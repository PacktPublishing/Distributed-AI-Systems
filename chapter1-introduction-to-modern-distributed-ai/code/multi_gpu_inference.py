"""
Multi-GPU distributed inference for high throughput.

Distributes inference requests across multiple GPUs to achieve higher throughput.
Each GPU processes requests independently (data parallelism for inference).

Usage:
    OMP_NUM_THREADS=4 torchrun --nproc_per_node=2 code/multi_gpu_inference.py --requests 1000
"""
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
import os
import time
from mdaisy import get_resnet18_fashionmnist, setup_distributed, cleanup_distributed

def benchmark_distributed_inference(num_requests=1000, batch_size=1):
    """Benchmark distributed inference across multiple GPUs"""
    rank, world_size, local_rank = setup_distributed()
    
    # Each GPU loads its own model copy
    model = get_resnet18_fashionmnist(num_classes=10).cuda()
    model.eval()
    
    # Create test data with distributed sampler
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    test_dataset = datasets.FashionMNIST(
        root='./data', train=False, download=(rank == 0), transform=transform
    )
    sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, sampler=sampler, num_workers=2
    )
    
    # Warmup
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            if i >= 10:
                break
            data = data.cuda()
            _ = model(data)
    
    torch.cuda.synchronize()
    dist.barrier(device_ids=[local_rank])  # Sync all GPUs before benchmarking
    
    # Benchmark - each GPU processes its share of requests
    requests_per_gpu = num_requests // world_size
    start_time = time.time()
    requests_processed = 0
    
    with torch.no_grad():
        for data, _ in test_loader:
            if requests_processed >= requests_per_gpu:
                break
            data = data.cuda()
            _ = model(data)
            requests_processed += batch_size
    
    torch.cuda.synchronize()
    dist.barrier(device_ids=[local_rank])  # Wait for all GPUs to finish
    
    total_time = time.time() - start_time
    
    # Gather results from all GPUs
    requests_tensor = torch.tensor([requests_processed], device=f'cuda:{local_rank}')
    gathered_requests = [torch.zeros_like(requests_tensor) for _ in range(world_size)]
    dist.all_gather(gathered_requests, requests_tensor)
    
    total_requests = sum(r.item() for r in gathered_requests)
    throughput = total_requests / total_time
    
    if rank == 0:
        print(f"Processed {total_requests} requests across {world_size} GPUs in {total_time:.2f}s")
        print(f"Time to finish {total_requests} requests: {total_time:.2f}s")
        print(f"Throughput: {throughput:.2f} requests/second")
    
    cleanup_distributed()
    return throughput, total_time

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Multi-GPU distributed inference benchmark')
    parser.add_argument('--requests', type=int, default=1000, help='Total number of requests to process')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size per request')
    args = parser.parse_args()
    benchmark_distributed_inference(num_requests=args.requests, batch_size=args.batch_size)
