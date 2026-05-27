"""
Network diagnostic tools for distributed training.
"""
import torch
import torch.distributed as dist
import time
import subprocess

def test_bandwidth(rank, world_size):
    """Test network bandwidth between nodes"""
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
    # Test different message sizes
    sizes_mb = [1, 10, 100, 1000]
    
    for size_mb in sizes_mb:
        size = size_mb * 1024 * 1024 // 4  # float32 elements
        tensor = torch.randn(size, device='cuda')
        
        # Warmup
        for _ in range(5):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        
        # Benchmark
        torch.cuda.synchronize()
        start = time.time()
        iterations = 10
        for _ in range(iterations):
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        # Calculate bandwidth
        data_mb = size_mb * 2 * iterations  # send + receive
        bandwidth = data_mb / elapsed
        
        if rank == 0:
            print(f"Size: {size_mb}MB, Bandwidth: {bandwidth:.2f} MB/s")
    
    dist.destroy_process_group()

def check_network_health():
    """Check network health using system tools"""
    # Check latency
    result = subprocess.run(
        ['ping', '-c', '5', 'target_host'],
        capture_output=True,
        text=True
    )
    print("Network Latency:")
    print(result.stdout)
    
    # Check bandwidth (requires iperf3)
    # result = subprocess.run(
    #     ['iperf3', '-c', 'target_host', '-t', '10'],
    #     capture_output=True,
    #     text=True
    # )
    # print("Network Bandwidth:")
    # print(result.stdout)

if __name__ == "__main__":
    # This would be called with torchrun
    # torchrun --nproc_per_node=2 ch10_network_diagnostics.py
    print("Run with: torchrun --nproc_per_node=2 ch10_network_diagnostics.py")

