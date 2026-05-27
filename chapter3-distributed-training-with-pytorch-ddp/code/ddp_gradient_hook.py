"""
Minimal example: register a gradient hook on a DDP-wrapped model to inspect or
modify gradients during backward (e.g. logging norm, gradient clipping).

Launch from the chapter directory:
    torchrun --nproc_per_node=2 code/ddp_gradient_hook.py
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
    """Small model with a single layer named 'fc' for hook demo."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(8, 4)

    def forward(self, x):
        return self.fc(x)


def main():
    rank, local_rank, world_size = setup()
    device = torch.device(f"cuda:{local_rank}")

    model = TinyNet().to(device)
    model = DDP(model, device_ids=[local_rank])

    def gradient_hook(grad):
        # Inspect or modify gradient (e.g. log norm, clip)
        norm = grad.norm().item()
        if rank == 0:
            print(f"  [rank {rank}] gradient norm: {norm:.4f}")
        # Optional: custom modification, e.g. gradient clipping per-parameter
        # grad = grad.clamp(-1.0, 1.0)
        return grad

    # Register hook on a parameter (access underlying module via .module)
    model.module.fc.weight.register_hook(gradient_hook)

    # One forward + backward so the hook runs
    x = torch.randn(2, 8, device=device)
    y = model(x).sum()
    y.backward()

    if rank == 0:
        print("Gradient hook ran during backward.")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
