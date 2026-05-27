"""
Tensor Parallelism Example: Column-Parallel and Row-Parallel Linear.

This script demonstrates the core concepts of Megatron-style tensor parallelism
using pure PyTorch. It shows how to split matrix operations across GPUs.

Usage:
    torchrun --nproc_per_node=2 tensor_parallel_mlp.py

The script implements:
1. ColumnParallelLinear: Splits weight columns across GPUs (no communication)
2. RowParallelLinear: Splits weight rows across GPUs (requires all-reduce)
3. TensorParallelMLP: Combines both for a complete MLP block

Requirements:
    pip install torch
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP


def setup_distributed():
    """Initialize distributed training."""
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
    return local_rank, dist.get_world_size(), dist.get_rank()


class ColumnParallelLinear(nn.Module):
    """
    Linear layer with column-wise weight partitioning.
    
    Weight matrix W is split along columns:
    - Full W: [in_features, out_features]
    - Each GPU holds: [in_features, out_features // world_size]
    
    Input is replicated, output is partitioned.
    No communication needed in forward pass.
    """
    def __init__(self, in_features: int, out_features: int, world_size: int, rank: int):
        super().__init__()
        assert out_features % world_size == 0, "out_features must be divisible by world_size"
        
        self.in_features = in_features
        self.out_features = out_features
        self.world_size = world_size
        self.rank = rank
        
        # Each GPU holds 1/world_size of the columns
        self.out_features_per_partition = out_features // world_size
        
        self.weight = nn.Parameter(
            torch.empty(self.out_features_per_partition, in_features)
        )
        self.bias = nn.Parameter(torch.empty(self.out_features_per_partition))
        
        # Initialize
        nn.init.kaiming_uniform_(self.weight)
        nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq, in_features] (replicated on all GPUs)
        # output: [batch, seq, out_features_per_partition] (partitioned)
        return nn.functional.linear(x, self.weight, self.bias)


class RowParallelLinear(nn.Module):
    """
    Linear layer with row-wise weight partitioning.
    
    Weight matrix W is split along rows:
    - Full W: [in_features, out_features]
    - Each GPU holds: [in_features // world_size, out_features]
    
    Input is partitioned (from ColumnParallelLinear), output needs all-reduce.
    """
    def __init__(self, in_features: int, out_features: int, world_size: int, rank: int):
        super().__init__()
        assert in_features % world_size == 0, "in_features must be divisible by world_size"
        
        self.in_features = in_features
        self.out_features = out_features
        self.world_size = world_size
        self.rank = rank
        
        # Each GPU holds 1/world_size of the rows
        self.in_features_per_partition = in_features // world_size
        
        self.weight = nn.Parameter(
            torch.empty(out_features, self.in_features_per_partition)
        )
        self.bias = nn.Parameter(torch.empty(out_features))
        
        # Initialize
        nn.init.kaiming_uniform_(self.weight)
        nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq, in_features_per_partition] (partitioned)
        # Local matmul
        output = nn.functional.linear(x, self.weight)
        
        # All-reduce to get full output
        dist.all_reduce(output, op=dist.ReduceOp.SUM)
        
        # Add bias after all-reduce (only on rank 0 to avoid double-counting)
        if self.rank == 0:
            output = output + self.bias
        else:
            # Broadcast bias contribution
            output = output + self.bias / self.world_size
        
        return output


class TensorParallelMLP(nn.Module):
    """
    Tensor-parallel MLP block (Megatron-style).
    
    Structure:
        Input -> ColumnParallelLinear -> GELU -> RowParallelLinear -> Output
    
    Communication happens only once (all-reduce in RowParallelLinear).
    """
    def __init__(self, hidden_size: int, intermediate_size: int, world_size: int, rank: int):
        super().__init__()
        
        # First linear: column-parallel (no communication)
        self.fc1 = ColumnParallelLinear(
            hidden_size, intermediate_size, world_size, rank
        )
        
        # Second linear: row-parallel (all-reduce at end)
        self.fc2 = RowParallelLinear(
            intermediate_size, hidden_size, world_size, rank
        )
        
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, seq, hidden_size] (replicated)
        x = self.fc1(x)           # -> [batch, seq, intermediate_size // world_size]
        x = self.activation(x)
        x = self.fc2(x)           # -> [batch, seq, hidden_size] (all-reduced)
        return x


class RegularMLP(nn.Module):
    """Regular MLP for comparison (no tensor parallelism)."""
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, intermediate_size)
        self.fc2 = nn.Linear(intermediate_size, hidden_size)
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.activation(self.fc1(x)))


def main():
    local_rank, world_size, rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    
    # Model dimensions
    batch_size = 4
    seq_length = 128
    hidden_size = 256
    intermediate_size = 1024  # 4x hidden_size, typical for transformers
    
    # Create tensor-parallel MLP
    tp_mlp = TensorParallelMLP(
        hidden_size, intermediate_size, world_size, rank
    ).to(device)
    
    # Create regular MLP for comparison (only on rank 0)
    if rank == 0:
        regular_mlp = RegularMLP(hidden_size, intermediate_size).to(device)
    
    # Create input (same on all ranks for verification)
    torch.manual_seed(42)
    x = torch.randn(batch_size, seq_length, hidden_size, device=device)
    
    # Forward pass with tensor parallelism
    with torch.no_grad():
        tp_output = tp_mlp(x)
    
    # Verify output shape
    if rank == 0:
        print(f"Input shape: {x.shape}")
        print(f"TP MLP output shape: {tp_output.shape}")
        print(f"World size: {world_size}")
        print(f"\nWeight shapes per GPU:")
        print(f"  fc1.weight: {tp_mlp.fc1.weight.shape} (columns split)")
        print(f"  fc2.weight: {tp_mlp.fc2.weight.shape} (rows split)")
        
        # Memory comparison
        tp_params = sum(p.numel() for p in tp_mlp.parameters())
        regular_params = sum(p.numel() for p in regular_mlp.parameters())
        
        print(f"\nParameter count:")
        print(f"  Regular MLP (per GPU): {regular_params:,}")
        print(f"  TP MLP (per GPU): {tp_params:,}")
        print(f"  Memory reduction: {1 - tp_params/regular_params:.1%}")
    
    # Synchronize before cleanup
    dist.barrier()
    
    if rank == 0:
        print("\nTensor parallelism demo complete!")
    
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
