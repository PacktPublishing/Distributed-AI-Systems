"""
Profile ResNet50 training with DDP on CIFAR-10 (practice script for Chapter 3).
Profiles 5 iterations, prints key averages on rank 0, and exports resnet50_ddp_trace.json.
CIFAR-10 downloads to ./data on first run.

From the chapter directory:
    torchrun --nproc_per_node=2 code/profile_ddp_resnet50.py
"""
import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.profiler import profile, ProfilerActivity
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


def main():
    rank, local_rank, world_size = setup()
    device = torch.device(f'cuda:{local_rank}')
    # Model
    model = torchvision.models.resnet50(num_classes=10)
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank])
    # Data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    sampler = DistributedSampler(trainset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
    dataloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, sampler=sampler, num_workers=4, pin_memory=True
    )
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    # Profile 5 iterations
    model.train()
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        for i, (data, target) in enumerate(dataloader):
            if i >= 5:
                break
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    if rank == 0:
        print("=" * 20)
        print("ResNet50 DDP Performance Profile")
        print("=" * 20)
        try:
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
        except (TypeError, AttributeError):
            print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=30))
        prof.export_chrome_trace("resnet50_ddp_trace.json")
        print("\nTrace exported to resnet50_ddp_trace.json")
        print("Open chrome://tracing in Chrome browser to visualize")
    cleanup()


if __name__ == '__main__':
    main()
