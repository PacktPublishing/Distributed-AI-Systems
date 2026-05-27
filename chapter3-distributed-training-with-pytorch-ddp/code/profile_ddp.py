"""
Profile DDP training (matches the "complete example of profiling DDP training" in Chapter 3).
Prints key averages and exports a Chrome trace (ddp_trace.json). View at chrome://tracing.

From the chapter directory:
    torchrun --nproc_per_node=2 code/profile_ddp.py

From the code directory:
    torchrun --nproc_per_node=2 profile_ddp.py
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


def train_with_profiling(model, dataloader, optimizer, criterion, rank, num_iterations=10):
    """Train with profiling to analyze DDP performance."""
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        with record_function("training_loop"):
            for i, (data, target) in enumerate(dataloader):
                if i >= num_iterations:
                    break
                data = data.cuda(rank, non_blocking=True)
                target = target.cuda(rank, non_blocking=True)
                with record_function("forward_pass"):
                    output = model(data)
                    loss = criterion(output, target)
                with record_function("backward_pass"):
                    loss.backward()
                with record_function("optimizer_step"):
                    optimizer.step()
                    optimizer.zero_grad()
    if rank == 0:
        print("=" * 80)
        print("DDP Performance Profile - Key Averages")
        print("=" * 80)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
        print("\n" + "=" * 80)
        print("Top Operations by Self CUDA Time")
        print("=" * 80)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
        prof.export_chrome_trace("ddp_trace.json")
        print("\nChrome trace exported to ddp_trace.json")
        print("Open chrome://tracing in Chrome browser to visualize")
    return prof


def main():
    rank, local_rank, world_size = setup()
    device = torch.device(f'cuda:{local_rank}')
    dataloader, sampler = get_dataloader(rank, world_size)
    model = nn.Linear(10, 1).to(device)
    model = DDP(model, device_ids=[local_rank])
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    train_with_profiling(model, dataloader, optimizer, criterion, local_rank, num_iterations=10)
    cleanup()


if __name__ == '__main__':
    main()
