"""
Elastic / fault-tolerant training with checkpointing: load at startup, train,
save periodically (rank 0 only, atomic write). After a worker restart, the
next run resumes from the latest checkpoint. See "Implementing Checkpointing
for Elastic Training" in Chapter 3.

Launch from the chapter directory (standard DDP, 2 processes):
    torchrun --nproc_per_node=2 code/train_elastic_checkpoint.py

Fault-tolerant (restart up to 2 times on failure; requires MASTER_ADDR/MASTER_PORT):
    torchrun --nproc_per_node=2 --max_restarts=2 --rdzv_backend=c10d \
      --rdzv_endpoint=127.0.0.1:29500 --rdzv_id=elastic1 code/train_elastic_checkpoint.py
"""
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data.distributed import DistributedSampler


CHECKPOINT_PATH = "checkpoint_elastic.pt"
NUM_EPOCHS = 4


def setup():
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return rank, local_rank, world_size


def load_checkpoint(checkpoint_path):
    if os.path.exists(checkpoint_path):
        return torch.load(checkpoint_path, map_location="cpu")
    return None


def save_checkpoint(model, optimizer, epoch, path):
    if dist.get_rank() != 0:
        return
    tmp = path + ".tmp"
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.module.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        tmp,
    )
    os.replace(tmp, path)


def create_model():
    return nn.Sequential(
        nn.Linear(8, 32),
        nn.ReLU(),
        nn.Linear(32, 2),
    )


def main():
    rank, local_rank, world_size = setup()
    device = torch.device(f"cuda:{local_rank}")

    model = create_model().to(device)
    model = DDP(model, device_ids=[local_rank])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Dummy data with DistributedSampler
    dataset = TensorDataset(
        torch.randn(64, 8),
        torch.randint(0, 2, (64,)),
    )
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    dataloader = DataLoader(dataset, batch_size=8, sampler=sampler)

    checkpoint = load_checkpoint(CHECKPOINT_PATH)
    start_epoch = (checkpoint["epoch"] + 1) if checkpoint else 0
    if checkpoint:
        model.module.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if rank == 0:
            print(f"Resumed from epoch {checkpoint['epoch']}")

    for epoch in range(start_epoch, NUM_EPOCHS):
        sampler.set_epoch(epoch)
        model.train()
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(data), target)
            loss.backward()
            optimizer.step()
        save_checkpoint(model, optimizer, epoch, CHECKPOINT_PATH)
        if rank == 0:
            print(f"Epoch {epoch} done, checkpoint saved.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
