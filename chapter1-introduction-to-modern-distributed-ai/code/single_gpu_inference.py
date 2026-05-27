"""
Single-GPU inference baseline for comparison with distributed inference.

Measures inference throughput (requests per second) on a single GPU.
"""
import torch
import torch.nn as nn
import time
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from mdaisy import get_resnet18_fashionmnist

def benchmark_inference(num_requests=1000, batch_size=1):
    """Benchmark inference throughput on single GPU"""
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load model (use a pretrained-like model for inference)
    model = get_resnet18_fashionmnist(num_classes=10).to(device)
    model.eval()
    
    # Create test data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    test_dataset = datasets.FashionMNIST(
        root='./data', train=False, download=True, transform=transform
    )
    # Take a subset for benchmarking
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    
    # Warmup
    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            if i >= 10:
                break
            data = data.to(device)
            _ = model(data)
    
    torch.cuda.synchronize()
    
    # Benchmark
    start_time = time.time()
    requests_processed = 0
    
    with torch.no_grad():
        for data, _ in test_loader:
            if requests_processed >= num_requests:
                break
            data = data.to(device)
            _ = model(data)
            requests_processed += batch_size
    
    torch.cuda.synchronize()
    total_time = time.time() - start_time
    
    throughput = requests_processed / total_time
    
    print(f"Processed {requests_processed} requests in {total_time:.2f}s")
    print(f"Time to finish {requests_processed} requests: {total_time:.2f}s")
    print(f"Throughput: {throughput:.2f} requests/second")
    
    return throughput, total_time

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Single-GPU inference benchmark')
    parser.add_argument('--requests', type=int, default=1000, help='Number of requests to process')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size per request')
    args = parser.parse_args()
    benchmark_inference(num_requests=args.requests, batch_size=args.batch_size)
