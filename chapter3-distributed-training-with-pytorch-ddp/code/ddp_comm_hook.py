"""
Minimal example: register a communication hook on DDP to customize gradient
synchronization (here, a custom AllReduce per bucket). See "Communication Hooks"
in Chapter 3.

Launch from the chapter directory:
    torchrun --nproc_per_node=2 code/ddp_comm_hook.py
"""
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def setup():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


class TinyNet(nn.Module):
    """Small model so a few buckets are used during backward."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, 4)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def main():
    rank, local_rank, world_size = setup()
    device = torch.device(f"cuda:{local_rank}")

    model = TinyNet().to(device)
    model = DDP(model, device_ids=[local_rank])

    def allreduce_hook(state, bucket):
        """Custom hook that does AllReduce on the gradient bucket."""
        tensor = bucket.buffer()
        if rank == 0:
            print(f"  [rank {rank}] comm hook: AllReduce on bucket (numel={tensor.numel()})")
        dist.all_reduce(tensor, async_op=False)
        fut = torch.futures.Future()
        fut.set_result(tensor)
        return fut

    model.register_comm_hook(state=None, hook=allreduce_hook)

    # Forward + backward so the communication hook runs for each bucket
    x = torch.randn(4, 32, device=device)
    y = model(x).sum()
    y.backward()

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
