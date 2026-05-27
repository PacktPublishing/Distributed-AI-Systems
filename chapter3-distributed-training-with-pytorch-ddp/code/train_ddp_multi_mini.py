"""
Minimal DDP training script for single-node and multi-node (Chapter 3).

Single-node (from chapter directory):
    torchrun --nproc_per_node=4 code/train_ddp_multi_mini.py

Multi-node: on the master node (node 0):
    torchrun --nnodes=2 --nproc_per_node=2 --node_rank=0 \
      --master_addr=<master_ip> --master_port=29500 code/train_ddp_multi_mini.py
On each worker node, use the same command with --node_rank=1, 2, ...
"""
import os
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


def setup():
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
        drop_last=True,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,
        pin_memory=True,
    )
    return dataloader, sampler


def main():
    rank, local_rank, world_size, device = setup()
    dataloader, sampler = get_dataloader(rank, world_size)
    model = nn.Linear(10, 1).to(device)
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
        if rank == 0 and num_batches > 0:
            print(f'Epoch {epoch} avg loss: {epoch_loss / num_batches:.4f}')

    if rank == 0:
        print('Done.')
    cleanup()


if __name__ == '__main__':
    main()
