"""
Extended single-GPU training with larger dataset and more epochs.

Trains ResNet18 on CIFAR-10 dataset (larger than FashionMNIST) for more epochs.
This demonstrates training on a more realistic workload.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import time
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from mdaisy import get_resnet18_cifar10

def train_single_gpu(num_epochs=20):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Data loading with data augmentation for CIFAR-10
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
    
    train_dataset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    
    test_dataset = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    
    # Model setup
    model = get_resnet18_cifar10(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.7)
    
    # Training loop
    model.train()
    start_time = time.time()
    
    print(f"Training ResNet18 on CIFAR-10 for {num_epochs} epochs...")
    print(f"Training samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    print("-" * 60)
    
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
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
                data, target = data.to(device), target.to(device)
                output = model(data)
                _, predicted = output.max(1)
                test_total += target.size(0)
                test_correct += predicted.eq(target).sum().item()
        test_acc = 100. * test_correct / test_total
        model.train()
        
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"Epoch {epoch+1:2d}/{num_epochs} | Loss: {epoch_loss:.4f} | "
              f"Train Acc: {epoch_acc:.2f}% | Test Acc: {test_acc:.2f}% | LR: {current_lr:.6f}")
    
    total_time = time.time() - start_time
    print("-" * 60)
    print(f"Total training time: {total_time:.2f}s ({total_time/60:.2f} minutes)")
    print(f"Final test accuracy: {test_acc:.2f}%")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Extended single-GPU training on CIFAR-10')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs (default: 20)')
    args = parser.parse_args()
    train_single_gpu(num_epochs=args.epochs)
