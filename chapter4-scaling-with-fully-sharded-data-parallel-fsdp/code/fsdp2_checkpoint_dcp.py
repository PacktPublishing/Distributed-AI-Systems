"""
FSDP2 checkpointing example using the DCP (Distributed Checkpoint) API.
Demonstrates saving and loading sharded state dicts.

Usage:
    torchrun --nproc_per_node=2 code/fsdp2_checkpoint_dcp.py

Output:
    Epoch 0 completed, loss: 0.xxxx
    Checkpoint saved to checkpoints/epoch_0
    Epoch 1 completed, loss: 0.xxxx
    Checkpoint saved to checkpoints/epoch_1
    Loading checkpoint from epoch 1...
    Checkpoint loaded, resuming from epoch 1
    Training completed!
"""
import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
import torch.distributed.checkpoint as dcp


class SimpleModel(nn.Module):
    """A simple model for demonstration."""
    def __init__(self, hidden_dim=1024, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(num_layers)
        ])
        self.output = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.output(x)


def save_checkpoint_dcp(model, optimizer, epoch, checkpoint_dir):
    """Save checkpoint using DCP API."""
    rank = dist.get_rank()
    
    # Get state dicts with full_state_dict=False to get sharded dicts
    model_state_dict = get_model_state_dict(
        model=model,
        options=StateDictOptions(
            full_state_dict=False,  # Keep sharded
            cpu_offload=True,  # Offload to CPU for saving
        ),
    )
    
    optim_state_dict = get_optimizer_state_dict(
        model=model,
        optimizers=optimizer,
        options=StateDictOptions(
            full_state_dict=False,
            cpu_offload=True,
        ),
    )
    
    checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch}")
    
    # Save using DCP
    state_dict = {
        "model": model_state_dict,
        "optimizer": optim_state_dict,
        "epoch": epoch,
    }
    dcp.save(state_dict, checkpoint_id=checkpoint_path)
    
    if rank == 0:
        print(f"Checkpoint saved to {checkpoint_path}")


def load_checkpoint_dcp(model, optimizer, checkpoint_dir, epoch):
    """Load checkpoint using DCP API."""
    rank = dist.get_rank()
    checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch}")
    
    # Prepare state dict structure for loading
    model_state_dict = get_model_state_dict(
        model=model,
        options=StateDictOptions(full_state_dict=False),
    )
    optim_state_dict = get_optimizer_state_dict(
        model=model,
        optimizers=optimizer,
        options=StateDictOptions(full_state_dict=False),
    )
    
    state_dict = {
        "model": model_state_dict,
        "optimizer": optim_state_dict,
        "epoch": 0,
    }
    
    # Load using DCP
    dcp.load(state_dict, checkpoint_id=checkpoint_path)
    
    # Set state dicts
    set_model_state_dict(
        model=model,
        model_state_dict=state_dict["model"],
        options=StateDictOptions(full_state_dict=False),
    )
    
    set_optimizer_state_dict(
        model=model,
        optimizers=optimizer,
        optim_state_dict=state_dict["optimizer"],
        options=StateDictOptions(full_state_dict=False),
    )
    
    loaded_epoch = state_dict["epoch"]
    if rank == 0:
        print(f"Checkpoint loaded from {checkpoint_path}, epoch {loaded_epoch}")
    
    return loaded_epoch


def main():
    # Initialize distributed
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    # Create device mesh
    mesh = init_device_mesh("cuda", (world_size,))
    
    if rank == 0:
        print(f"FSDP2 Checkpointing Demo with {world_size} GPUs")
    
    # Create model
    model = SimpleModel(hidden_dim=1024, num_layers=4).to(device)
    
    # Apply FSDP2 to each layer, then to the whole model
    for layer in model.layers:
        fully_shard(layer, mesh=mesh)
    fully_shard(model, mesh=mesh)
    
    # Setup optimizer
    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    
    # Checkpoint directory
    checkpoint_dir = "checkpoints"
    if rank == 0 and os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
    dist.barrier()
    
    # Training loop with checkpointing
    model.train()
    for epoch in range(2):
        # Dummy training step
        inputs = torch.randn(32, 1024, device=device)
        targets = torch.randn(32, 1, device=device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = nn.functional.mse_loss(outputs, targets)
        loss.backward()
        optimizer.step()
        
        if rank == 0:
            print(f"Epoch {epoch} completed, loss: {loss.item():.4f}")
        
        # Save checkpoint
        save_checkpoint_dcp(model, optimizer, epoch, checkpoint_dir)
    
    # Demonstrate loading checkpoint
    if rank == 0:
        print("\nLoading checkpoint from epoch 1...")
    
    # Create fresh model and optimizer to demonstrate loading
    model2 = SimpleModel(hidden_dim=1024, num_layers=4).to(device)
    for layer in model2.layers:
        fully_shard(layer, mesh=mesh)
    fully_shard(model2, mesh=mesh)
    optimizer2 = optim.AdamW(model2.parameters(), lr=0.001)
    
    # Load checkpoint
    loaded_epoch = load_checkpoint_dcp(model2, optimizer2, checkpoint_dir, epoch=1)
    
    if rank == 0:
        print(f"Checkpoint loaded, resuming from epoch {loaded_epoch}")
        print("Training completed!")
    
    # Cleanup
    dist.barrier()
    if rank == 0 and os.path.exists(checkpoint_dir):
        shutil.rmtree(checkpoint_dir)
    
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
