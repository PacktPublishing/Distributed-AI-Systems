"""
FSDP via SPMD for TPU/XLA devices.

This script requires TPU hardware and PyTorch/XLA. It will not run on GPU.

Usage (on TPU VM):
    python code/fsdp_spmd_tpu.py

For Google Cloud TPU:
    1. Create a TPU VM: gcloud compute tpus tpu-vm create ...
    2. SSH into the VM: gcloud compute tpus tpu-vm ssh ...
    3. Install PyTorch/XLA: pip install torch torch_xla
    4. Run this script: python fsdp_spmd_tpu.py
"""
import numpy as np
import torch
import torch.nn as nn

try:
    import torch_xla.core.xla_model as xm
    import torch_xla.runtime as xr
    import torch_xla.distributed.spmd as xs
    from torch_xla.experimental.spmd_fully_sharded_data_parallel import (
        SpmdFullyShardedDataParallel as FSDPv2
    )
    HAS_XLA = True
except ImportError:
    HAS_XLA = False
    print("PyTorch/XLA not installed. This script requires TPU hardware.")
    print("Install with: pip install torch_xla")


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
        self.output = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.output(x)


def main():
    if not HAS_XLA:
        print("Exiting: PyTorch/XLA required for TPU training.")
        return
    
    # Enable XLA SPMD execution mode
    xr.use_spmd()
    
    # Define the mesh (must have an axis named 'fsdp')
    num_devices = xr.global_runtime_device_count()
    print(f"Running on {num_devices} TPU devices")
    
    mesh_shape = (num_devices, 1)
    device_ids = np.array(range(num_devices))
    mesh = xs.Mesh(device_ids, mesh_shape, ('fsdp', 'model'))
    
    # Create model
    model = SimpleModel(hidden_dim=1024, num_layers=4)
    
    # Apply FSDP via SPMD
    model = FSDPv2(model, mesh)
    
    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    # Create dummy input and shard it
    batch_size = 32
    hidden_dim = 1024
    x = torch.randn(batch_size, hidden_dim)
    x = xs.mark_sharding(x, mesh, ('fsdp', None))
    
    # Training step
    optimizer.zero_grad()
    output = model(x)
    loss = output.sum()
    loss.backward()
    optimizer.step()
    
    # Sync and print
    xm.mark_step()
    print(f"Training step completed. Loss: {loss.item():.4f}")


if __name__ == "__main__":
    main()
