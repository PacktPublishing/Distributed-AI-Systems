"""
DDP training with DistributedSampler (matches the "Using DistributedSampler" snippet in Chapter 3).
Each rank sees a different shard of the dataset; sampler.set_epoch(epoch) gives different shuffle per epoch.

Launch from the chapter directory:
    torchrun --nproc_per_node=4 code/train_ddp_sampler.py

Or from the code directory:
    torchrun --nproc_per_node=4 train_ddp_sampler.py
"""
import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


def setup():
    """Initialize process group and set device."""
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(local_rank)
    device = torch.device(f'cuda:{local_rank}')
    dist.init_process_group(backend='nccl')
    return rank, local_rank, world_size, device


def cleanup():
    dist.destroy_process_group()


class MyDataset(Dataset):
    def __init__(self, size=1000):
        self.data = torch.randn(size, 10)
        self.labels = torch.randn(size, 1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def get_dataloader(rank, world_size, batch_size=32):
    dataset = MyDataset(size=1000)
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,  # avoids DDP sync issues
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,  # use 0 for portability; set to 4 for faster loading when supported
        pin_memory=True,
    )
    return dataloader, sampler


def create_model():
    return nn.Linear(10, 1)


def train():
    rank, local_rank, world_size, device = setup()
    dataloader, sampler = get_dataloader(rank, world_size)
    model = create_model().to(device)
    model = DDP(model, device_ids=[local_rank])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()

    for epoch in range(10):
        sampler.set_epoch(epoch)
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        for batch_idx, (data, target) in enumerate(dataloader):
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1
            #if rank == 0 and batch_idx % 10 == 0:
            #    print(f'Epoch {epoch}, batch {batch_idx}, loss {loss.item():.4f}')
        if rank == 0 and num_batches > 0:
            print(f'Epoch {epoch} avg loss: {epoch_loss / num_batches:.4f}')

    if rank == 0:
        print('Done.')
    cleanup()


if __name__ == '__main__':
    train()
