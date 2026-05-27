"""
Simple Pipeline Parallelism Example.

Demonstrates the core concepts of pipeline parallelism using pure PyTorch.
This is a simplified version of Megatron's pipeline parallelism.

Usage:
    torchrun --nproc_per_node=2 pipeline_parallel_simple.py

The script implements:
1. Model partitioning across pipeline stages
2. Micro-batch scheduling
3. Forward and backward pass coordination

Requirements:
    pip install torch
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from typing import List, Optional


def setup_distributed():
    """Initialize distributed training."""
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
    return local_rank, dist.get_world_size(), dist.get_rank()


class TransformerBlock(nn.Module):
    """Simple transformer block for demonstration."""
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        # FFN with residual
        x = self.norm2(x + self.ffn(x))
        return x


class PipelineStage(nn.Module):
    """
    A pipeline stage containing multiple transformer blocks.
    
    In pipeline parallelism, the model is split into stages,
    each stage runs on a different GPU.
    """
    def __init__(self, hidden_size: int, num_heads: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads)
            for _ in range(num_layers)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class PipelineParallelModel(nn.Module):
    """
    Pipeline-parallel model wrapper.
    
    Handles:
    1. Receiving activations from previous stage
    2. Running local computation
    3. Sending activations to next stage
    """
    def __init__(
        self,
        stage: PipelineStage,
        rank: int,
        world_size: int,
        is_first_stage: bool,
        is_last_stage: bool,
    ):
        super().__init__()
        self.stage = stage
        self.rank = rank
        self.world_size = world_size
        self.is_first_stage = is_first_stage
        self.is_last_stage = is_last_stage
    
    def recv_from_prev(self, shape: tuple, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Receive activations from previous stage."""
        tensor = torch.empty(shape, dtype=dtype, device=device)
        dist.recv(tensor, src=self.rank - 1)
        return tensor
    
    def send_to_next(self, tensor: torch.Tensor):
        """Send activations to next stage."""
        dist.send(tensor, dst=self.rank + 1)
    
    def recv_grad_from_next(self, shape: tuple, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        """Receive gradients from next stage."""
        tensor = torch.empty(shape, dtype=dtype, device=device)
        dist.recv(tensor, src=self.rank + 1)
        return tensor
    
    def send_grad_to_prev(self, tensor: torch.Tensor):
        """Send gradients to previous stage."""
        dist.send(tensor, dst=self.rank - 1)


def run_pipeline_forward(
    model: PipelineParallelModel,
    micro_batches: List[torch.Tensor],
    device: torch.device,
) -> List[torch.Tensor]:
    """
    Run forward pass for all micro-batches.
    
    In pipeline parallelism, we process multiple micro-batches
    to keep all stages busy (pipeline fill).
    """
    outputs = []
    
    for mb_idx, micro_batch in enumerate(micro_batches):
        if model.is_first_stage:
            # First stage: use input directly
            x = micro_batch.to(device)
        else:
            # Other stages: receive from previous stage
            x = model.recv_from_prev(
                micro_batch.shape, micro_batch.dtype, device
            )
        
        # Run local computation
        x.requires_grad_(True)
        output = model.stage(x)
        outputs.append((x, output))
        
        if not model.is_last_stage:
            # Send to next stage
            model.send_to_next(output.detach())
    
    return outputs


def run_pipeline_backward(
    model: PipelineParallelModel,
    forward_outputs: List[tuple],
    targets: Optional[List[torch.Tensor]],
    device: torch.device,
):
    """
    Run backward pass for all micro-batches.
    
    Backward pass runs in reverse order of forward pass.
    """
    loss_fn = nn.MSELoss()
    total_loss = 0.0
    
    # Process micro-batches in reverse order
    for mb_idx in range(len(forward_outputs) - 1, -1, -1):
        input_activation, output = forward_outputs[mb_idx]
        
        if model.is_last_stage:
            # Last stage: compute loss
            target = targets[mb_idx].to(device)
            loss = loss_fn(output, target)
            total_loss += loss.item()
            loss.backward()
        else:
            # Other stages: receive gradient from next stage
            grad = model.recv_grad_from_next(output.shape, output.dtype, device)
            output.backward(grad)
        
        if not model.is_first_stage:
            # Send gradient to previous stage
            model.send_grad_to_prev(input_activation.grad)
    
    return total_loss


def main():
    local_rank, world_size, rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    
    # Model configuration
    hidden_size = 256
    num_heads = 4
    total_layers = 8
    layers_per_stage = total_layers // world_size
    
    # Micro-batch configuration
    num_micro_batches = 4
    micro_batch_size = 2
    seq_length = 64
    
    # Create pipeline stage for this rank
    stage = PipelineStage(hidden_size, num_heads, layers_per_stage).to(device)
    
    # Wrap in pipeline model
    model = PipelineParallelModel(
        stage=stage,
        rank=rank,
        world_size=world_size,
        is_first_stage=(rank == 0),
        is_last_stage=(rank == world_size - 1),
    )
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Create micro-batches (only needed on first stage)
    torch.manual_seed(42)
    if rank == 0:
        micro_batches = [
            torch.randn(micro_batch_size, seq_length, hidden_size)
            for _ in range(num_micro_batches)
        ]
    else:
        # Placeholder shapes for receiving
        micro_batches = [
            torch.empty(micro_batch_size, seq_length, hidden_size)
            for _ in range(num_micro_batches)
        ]
    
    # Create targets (only needed on last stage)
    if rank == world_size - 1:
        targets = [
            torch.randn(micro_batch_size, seq_length, hidden_size)
            for _ in range(num_micro_batches)
        ]
    else:
        targets = None
    
    # Training step
    optimizer.zero_grad()
    
    # Forward pass
    forward_outputs = run_pipeline_forward(model, micro_batches, device)
    
    # Backward pass
    loss = run_pipeline_backward(model, forward_outputs, targets, device)
    
    # Update weights
    optimizer.step()
    
    # Print results
    if rank == world_size - 1:
        print(f"Pipeline parallelism demo complete!")
        print(f"World size: {world_size}")
        print(f"Layers per stage: {layers_per_stage}")
        print(f"Micro-batches: {num_micro_batches}")
        print(f"Average loss: {loss / num_micro_batches:.4f}")
    
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
