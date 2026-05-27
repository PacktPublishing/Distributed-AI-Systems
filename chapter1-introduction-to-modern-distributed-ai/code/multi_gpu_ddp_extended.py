"""
Extended multi-GPU distributed training with larger dataset and more epochs.

Trains ResNet18 on CIFAR-10 dataset using distributed training.
Uses the same model as single_gpu_extended.py for fair comparison.

Usage:
    OMP_NUM_THREADS=4 torchrun --nproc_per_node=2 code/multi_gpu_ddp_extended.py --epochs 20

Or use the launch script:
    bash code/launch_torchrun_extended.sh
"""
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
from mdaisy import get_resnet18_cifar10, setup_distributed, cleanup_distributed
import os
import time

def train_ddp(num_epochs=20):
    """Run distributed training using PyTorch DDP"""
    rank, world_size, local_rank = setup_distributed()
    
    # Data loading with distributed sampler
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Only rank 0 downloads the dataset
    if rank == 0:
        train_dataset = datasets.CIFAR10(
            root='./data', train=True, download=True, transform=transform_train
        )
    # Wait for rank 0 to finish downloading
    dist.barrier(device_ids=[local_rank])
    # Now all ranks can load the dataset
    if rank != 0:
        train_dataset = datasets.CIFAR10(
            root='./data', train=True, download=False, transform=transform_train
        )
    
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(
        train_dataset, batch_size=128, sampler=train_sampler, num_workers=4, pin_memory=True
    )
    
    # Only rank 0 downloads the test dataset
    if rank == 0:
        test_dataset = datasets.CIFAR10(
            root='./data', train=False, download=True, transform=transform_test
        )
    # Wait for rank 0 to finish downloading
    dist.barrier(device_ids=[local_rank])
    # Now all ranks can load the test dataset
    if rank != 0:
        test_dataset = datasets.CIFAR10(
            root='./data', train=False, download=False, transform=transform_test
        )
    
    test_sampler = DistributedSampler(test_dataset, num_replicas=world_size, rank=rank)
    test_loader = DataLoader(
        test_dataset, batch_size=128, sampler=test_sampler, num_workers=4, pin_memory=True
    )
    
    # Create model (same as single_gpu_extended.py)
    model = get_resnet18_cifar10(num_classes=10).cuda()
    
    # Wrap with DDP
    model = DDP(model, device_ids=[local_rank])
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.7)
    
    # Training loop
    model.train()
    start_time = time.time()
    
    if rank == 0:
        print(f"Training ResNet18 on CIFAR-10 for {num_epochs} epochs with {world_size} GPUs...")
        print(f"Training samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
        print("-" * 60)
    
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)  # Important for shuffling
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.cuda(), target.cuda()
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        # Evaluate on test set
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.cuda(), target.cuda()
                output = model(data)
                _, predicted = output.max(1)
                test_total += target.size(0)
                test_correct += predicted.eq(target).sum().item()
        test_acc = 100. * test_correct / test_total
        model.train()
        
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        if rank == 0:
            print(f"Epoch {epoch+1:2d}/{num_epochs} | Loss: {epoch_loss:.4f} | "
                  f"Train Acc: {epoch_acc:.2f}% | Test Acc: {test_acc:.2f}% | LR: {current_lr:.6f}", flush=True)
    
    total_time = time.time() - start_time
    if rank == 0:
        print("-" * 60)
        print(f"Total training time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
        print(f"Final test accuracy: {test_acc:.2f}%")
    
    cleanup_distributed()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Extended multi-GPU distributed training on CIFAR-10')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs (default: 20)')
    args = parser.parse_args()
    train_ddp(num_epochs=args.epochs)
