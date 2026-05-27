"""
Profile DDP with focus on NCCL communication (matches the "Profiling Multi-Node DDP"
snippet in Chapter 3). Runs one training step under the profiler, prints top NCCL ops,
and exports ddp_multinode_rank0.json. Works for single-node or multi-node.

From the chapter directory:
    torchrun --nproc_per_node=2 code/profile_ddp_multinode.py

From the code directory:
    torchrun --nproc_per_node=2 profile_ddp_multinode.py
"""
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import profile, ProfilerActivity
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler


def setup():
    rank = int(os.environ['RANK'])
    local_rank = int(os.environ['LOCAL_RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    return rank, local_rank, world_size


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
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0, pin_memory=True), sampler


def _device_time(e):
    """Get device (CUDA) time in microseconds; attribute name varies by PyTorch version."""
    try:
        for name in ('device_time', 'device_time_total', 'self_device_time_total',
                     'cuda_time_total', 'self_cuda_time_total'):
            t = getattr(e, name, None)
            if t is not None:
                return int(t)
    except (TypeError, ValueError):
        pass
    return 0


def profile_multi_node_ddp(model, dataloader, optimizer, criterion, rank, local_rank):
    """Profile DDP with focus on inter-node vs intra-node communication."""
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
    ) as prof:
        for data, target in dataloader:
            data = data.cuda(local_rank, non_blocking=True)
            target = target.cuda(local_rank, non_blocking=True)
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            break  # Profile just one iteration
    if rank == 0:
        events = prof.key_averages()
        nccl_ops = [e for e in events if 'nccl' in e.key.lower()]
        print("=" * 80)
        print("Multi-Node DDP Communication Analysis")
        print("=" * 80)
        for op in nccl_ops[:10]:
            ms = _device_time(op) / 1000.0
            print(f"{op.key}: {ms:.2f} ms")
        prof.export_chrome_trace(f"ddp_multinode_rank{rank}.json")
        print(f"\nTrace exported to ddp_multinode_rank{rank}.json")
        print("Open chrome://tracing in Chrome browser to visualize")
    return prof


def main():
    rank, local_rank, world_size = setup()
    device = torch.device(f'cuda:{local_rank}')
    dataloader, _ = get_dataloader(rank, world_size)
    model = nn.Linear(10, 1).to(device)
    model = DDP(model, device_ids=[local_rank])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    profile_multi_node_ddp(model, dataloader, optimizer, criterion, rank, local_rank)
    cleanup()


if __name__ == '__main__':
    main()
