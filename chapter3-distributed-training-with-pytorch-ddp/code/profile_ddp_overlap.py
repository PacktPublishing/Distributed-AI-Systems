"""
Profile DDP backward pass to analyze computation-communication overlap (matches the
"Analyzing Computation-Communication Overlap" snippet in Chapter 3).
Prints AllReduce vs backward times and exports ddp_overlap_trace.json.

From the chapter directory:
    torchrun --nproc_per_node=2 code/profile_ddp_overlap.py

From the code directory:
    torchrun --nproc_per_node=2 profile_ddp_overlap.py
"""
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import profile, record_function, ProfilerActivity
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


def analyze_ddp_overlap(model, loss, rank):
    """Analyze computation-communication overlap in DDP backward pass."""
    with profile(
        activities=[ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
    ) as prof:
        with record_function("backward_with_ddp"):
            loss.backward()
    if rank == 0:
        events = prof.key_averages()
        allreduce_ops = [e for e in events if 'nccl' in e.key.lower() and 'allreduce' in e.key.lower()]
        backward_ops = [e for e in events if 'backward' in e.key.lower() or 'gradient' in e.key.lower()]
        print("=" * 80)
        print("DDP Overlap Analysis")
        print("=" * 80)
        print(f"AllReduce operations found: {len(allreduce_ops)}")
        print(f"Backward operations found: {len(backward_ops)}")
        # Profiler event attribute for device (CUDA) time varies by PyTorch version
        def _device_time(e):
            try:
                for name in ('device_time', 'device_time_total', 'self_device_time_total',
                             'cuda_time_total', 'self_cuda_time_total'):
                    t = getattr(e, name, None)
                    if t is not None:
                        return int(t)
            except (TypeError, ValueError):
                pass
            return 0
        total_allreduce_time = sum(_device_time(e) for e in allreduce_ops)
        total_backward_time = sum(_device_time(e) for e in backward_ops)
        print(f"\nTotal AllReduce time: {total_allreduce_time / 1000:.2f} ms")
        print(f"Total backward compute time: {total_backward_time / 1000:.2f} ms")
        if total_backward_time > total_allreduce_time * 1.5:
            print("✓ Good overlap: Computation time exceeds communication time")
            print("  This indicates AllReduce is happening concurrently with gradient computation")
        else:
            print("⚠ Limited overlap: Communication time is significant")
            print("  Consider: larger bucket size, faster interconnects, or larger models")
        prof.export_chrome_trace("ddp_overlap_trace.json")
        print("\nOverlap trace exported to ddp_overlap_trace.json")
        print("Open chrome://tracing in Chrome browser to visualize")
    return prof


def main():
    rank, local_rank, world_size = setup()
    device = torch.device(f'cuda:{local_rank}')
    dataloader, _ = get_dataloader(rank, world_size)
    model = nn.Linear(10, 1).to(device)
    model = DDP(model, device_ids=[local_rank])
    criterion = nn.MSELoss()
    # One forward pass, then profile backward only
    model.train()
    data, target = next(iter(dataloader))
    data, target = data.to(device), target.to(device)
    output = model(data)
    loss = criterion(output, target)
    analyze_ddp_overlap(model, loss, rank)
    cleanup()


if __name__ == '__main__':
    main()
