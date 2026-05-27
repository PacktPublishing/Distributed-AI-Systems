"""
Single-GPU training baseline for comparison with distributed training.

Trains ResNet18 from torchvision on FashionMNIST dataset.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import time
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from mdaisy import get_resnet18_fashionmnist

def train_single_gpu():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    train_dataset = datasets.FashionMNIST(
        root='./data', train=True, download=True, transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    
    # Model setup
    model = get_resnet18_fashionmnist(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop
    model.train()
    start_time = time.time()
    
    num_epochs = 3  # Enough to see training progress, completes in < 30s
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        for _, (data, target) in enumerate(train_loader):
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
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")
    
    total_time = time.time() - start_time
    print(f"\nTotal training time: {total_time:.2f}s")

if __name__ == "__main__":
    train_single_gpu()
