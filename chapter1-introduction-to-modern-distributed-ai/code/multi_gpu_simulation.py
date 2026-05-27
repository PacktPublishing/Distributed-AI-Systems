"""
Single-GPU simulation of multi-GPU distributed training.

This script allows you to test distributed training code on a single GPU by simulating
multiple processes. It's useful for:
- Testing distributed code logic without multiple GPUs
- Debugging distributed training issues on a single-GPU machine
- Learning how distributed training works

The script can optionally use CUDA Multi-Process Service (MPS) to efficiently share
a single GPU among multiple processes, reducing context switching overhead. MPS is
optional - the script works without it, but performance may be better with MPS enabled.

Usage:
    # Option 1: With MPS (recommended for better performance)
    # Step 1: Start MPS daemon (only needed once, requires sudo)
    sudo nvidia-cuda-mps-control -d
    
    # Step 2: Run the simulation (2 processes on GPU 0)
    CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=2 code/multi_gpu_simulation.py
    
    # Option 2: Without MPS (works but may have higher context switching overhead)
    CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=2 code/multi_gpu_simulation.py

Expected output:
    Rank 0 says hello.
    Rank 1 says hello.

Note: MPS (Multi-Process Service) is optional but recommended. It allows multiple CUDA
      processes to share a single GPU more efficiently by reducing context switching overhead.
      Without MPS, the script will still work correctly, but you may experience:
      - Higher GPU context switching overhead
      - Slightly slower performance
      - More GPU memory fragmentation
"""
import torch
import torch.distributed as dist
import os

def simulate_multi_gpu():
    """Simulate multi-GPU distributed training on a single GPU"""
    # torchrun 会自动初始化进程组，但为了兼容性我们也可以手动初始化
    if not dist.is_initialized():
        dist.init_process_group("nccl")
    
    # 获取 rank 和 local_rank
    rank = dist.get_rank()
    local_rank = int(os.environ.get('LOCAL_RANK', rank))
    world_size = dist.get_world_size()
    
    # 设置设备（在单GPU模拟时，所有进程都使用GPU 0）
    available_gpus = torch.cuda.device_count()
    if available_gpus == 1:
        device_id = 0
        mode = "单GPU模拟模式"
    else:
        device_id = local_rank
        mode = "多GPU模式"
    
    torch.cuda.set_device(device_id)
    device = torch.device(f'cuda:{device_id}')
    
    # 打印信息
    print(f"Rank {rank} (local_rank={local_rank}, world_size={world_size}) says hello. "
          f"使用设备: {device} [{mode}]")
    
    dist.destroy_process_group()

if __name__ == "__main__":
    simulate_multi_gpu()
