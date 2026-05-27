"""
Demo of DDP join() for uneven inputs: ranks have different numbers of batches.
Without join(), the rank that finishes first would hang waiting in AllReduce.
With join(), early-finished ranks participate in dummy AllReduces until everyone exits.

Launch from the chapter directory:
    torchrun --nproc_per_node=2 code/ddp_join_demo.py
"""
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset


def setup():
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def main():
    rank, local_rank, world_size = setup()
    device = torch.device(f"cuda:{local_rank}")

    # Uneven: rank 0 has 4 samples (2 batches), rank 1 has 6 samples (3 batches)
    num_samples = 4 if rank == 0 else 6
    batch_size = 2
    dataset = TensorDataset(
        torch.randn(num_samples, 8, device=device),
        torch.randint(0, 2, (num_samples,), device=device),
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model = nn.Sequential(
        nn.Linear(8, 16),
        nn.ReLU(),
        nn.Linear(16, 2),
    ).to(device)
    model = DDP(model, device_ids=[local_rank])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    if rank == 0:
        print("Running with join(): rank 0 has 2 batches, rank 1 has 3 batches.")

    with model.join():
        for batch_idx, (data, target) in enumerate(dataloader):
            output = model(data)
            loss = nn.functional.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if rank == 0:
                print(f"  [rank {rank}] step {batch_idx + 1} done.")

    if rank == 0:
        print("join() demo finished (no hang).")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
