"""
Multi-GPU distributed inference with request-split pattern.

This implements a request-split pattern where requests are distributed
via round-robin assignment to multiple GPU workers. This simulates a real
production serving scenario where requests arrive dynamically and are
assigned to available GPUs.

Key differences from data-split approach:
1. Requests are assigned dynamically (round-robin) rather than pre-split
2. Each GPU processes requests as they arrive, simulating a queue
3. Better load balancing when requests have varying processing times

Usage:
    OMP_NUM_THREADS=4 torchrun --nproc_per_node=2 code/multi_gpu_inference_queue.py --requests 1000
"""
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import time
from mdaisy import get_resnet18_fashionmnist, setup_distributed, cleanup_distributed

def benchmark_distributed_inference_queue(num_requests=1000, batch_size=1):
    """
    Benchmark distributed inference using request-split pattern.
    
    In this pattern:
    - Rank 0 acts as the request dispatcher (simulates incoming requests)
    - Requests are assigned to GPUs in round-robin fashion
    - Each GPU processes requests as they are assigned (queue-like behavior)
    - This is more realistic for production where requests arrive dynamically
    """
    rank, world_size, local_rank = setup_distributed()
    
    # Load model on each GPU
    model = get_resnet18_fashionmnist(num_classes=10).cuda()
    model.eval()
    
    # Create test data (only rank 0 needs to load the full dataset)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    test_dataset = datasets.FashionMNIST(
        root='./data', train=False, download=(rank == 0), transform=transform
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    # Warmup: Run a few iterations to warm up GPU and CUDA kernels
    # This avoids cold-start overhead that would skew benchmark results
    # 10 iterations is typically enough to reach stable performance
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            if i >= 10:
                break
            data = data.cuda()
            _ = model(data)
    torch.cuda.synchronize()
    dist.barrier(device_ids=[local_rank])
    
    # Each GPU tracks its own processed requests
    requests_processed = 0
    start_time = None
    
    # Simulate round-robin request assignment
    # In real queue system: request arrives -> assign to next available GPU -> process
    request_id = 0
    
    torch.cuda.synchronize()
    dist.barrier(device_ids=[local_rank])  # Sync before starting
    
    start_time = time.time()  # Start timing for all GPUs together
    
    with torch.no_grad():
        for data, _ in test_loader:
            if request_id >= num_requests:
                break
            
            # Round-robin assignment: request_id % world_size determines which GPU handles it
            assigned_gpu = request_id % world_size
            
            if rank == assigned_gpu:
                # This GPU should process this request
                data = data.cuda()
                _ = model(data)
                requests_processed += data.size(0)
            
            request_id += 1
    
    torch.cuda.synchronize()
    dist.barrier(device_ids=[local_rank])  # Wait for all GPUs to finish
    
    # Calculate total time (all GPUs finish together)
    total_time = time.time() - start_time
    
    # Gather results from all GPUs
    requests_tensor = torch.tensor([requests_processed], device=f'cuda:{local_rank}')
    gathered_requests = [torch.zeros_like(requests_tensor) for _ in range(world_size)]
    dist.all_gather(gathered_requests, requests_tensor)
    
    total_requests = sum(r.item() for r in gathered_requests)
    throughput = total_requests / total_time
    
    if rank == 0:
        print(f"[Request-Split Pattern] Processed {total_requests} requests across {world_size} GPUs in {total_time:.2f}s")
        print(f"Time to finish {total_requests} requests: {total_time:.2f}s")
        print(f"Throughput: {throughput:.2f} requests/second")
        print(f"Average latency per request: {total_time / total_requests * 1000:.2f} ms")
        print(f"Requests per GPU: {[int(r.item()) for r in gathered_requests]}")
    
    cleanup_distributed()
    return throughput, total_time

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Multi-GPU distributed inference with request-split pattern')
    parser.add_argument('--requests', type=int, default=1000, help='Total number of requests to process')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size per request')
    args = parser.parse_args()
    benchmark_distributed_inference_queue(num_requests=args.requests, batch_size=args.batch_size)
