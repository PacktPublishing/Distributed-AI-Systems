"""
DeepSpeed-Ulysses Sequence Parallelism Demo

This script demonstrates the core idea behind DeepSpeed-Ulysses: using all-to-all
communication to transpose between sequence-parallel and head-parallel layouts,
enabling standard attention computation without partial softmax handling.

The key transformation:
  Before all-to-all: (batch, local_seq, num_heads, head_dim) - each GPU has seq chunk
  After all-to-all:  (batch, full_seq, local_heads, head_dim) - each GPU has head chunk

Usage:
    # Run with 2 GPUs
    torchrun --nproc_per_node=2 ulysses_demo.py

    # Run with 4 GPUs
    torchrun --nproc_per_node=4 ulysses_demo.py

Requirements:
    - PyTorch with NCCL backend
    - Multiple GPUs (2 or 4 recommended)
"""

import torch
import torch.distributed as dist
import torch.nn.functional as F


def all_to_all_seq_to_head(x: torch.Tensor, world_size: int) -> torch.Tensor:
    """
    Transpose from sequence-parallel to head-parallel layout.
    
    Input:  (batch, local_seq, num_heads, head_dim) - each GPU has a sequence chunk
    Output: (batch, full_seq, local_heads, head_dim) - each GPU has a head chunk
    
    This is the "gather sequence, scatter heads" operation.
    """
    batch, local_seq, num_heads, head_dim = x.shape
    
    # Reshape for all-to-all: split heads into world_size chunks
    # (batch, local_seq, world_size, local_heads, head_dim)
    x = x.view(batch, local_seq, world_size, num_heads // world_size, head_dim)
    
    # Permute to prepare for all-to-all
    # (world_size, batch, local_seq, local_heads, head_dim)
    x = x.permute(2, 0, 1, 3, 4).contiguous()
    
    # All-to-all: exchange sequence chunks for head chunks
    output = torch.empty_like(x)
    dist.all_to_all_single(output, x)
    
    # Reshape back: now we have full sequence but only local heads
    # (batch, full_seq, local_heads, head_dim)
    output = output.permute(1, 0, 2, 3, 4).contiguous()
    output = output.view(batch, local_seq * world_size, num_heads // world_size, head_dim)
    
    return output


def all_to_all_head_to_seq(x: torch.Tensor, world_size: int) -> torch.Tensor:
    """
    Transpose from head-parallel to sequence-parallel layout.
    
    Input:  (batch, full_seq, local_heads, head_dim) - each GPU has a head chunk
    Output: (batch, local_seq, num_heads, head_dim) - each GPU has a sequence chunk
    
    This is the "gather heads, scatter sequence" operation (reverse of seq_to_head).
    """
    batch, full_seq, local_heads, head_dim = x.shape
    local_seq = full_seq // world_size
    
    # Reshape for all-to-all
    # (world_size, batch, local_seq, local_heads, head_dim)
    x = x.view(batch, world_size, local_seq, local_heads, head_dim)
    x = x.permute(1, 0, 2, 3, 4).contiguous()
    
    # All-to-all: exchange head chunks for sequence chunks
    output = torch.empty_like(x)
    dist.all_to_all_single(output, x)
    
    # Reshape back: now we have local sequence but all heads
    # (batch, local_seq, num_heads, head_dim)
    output = output.permute(1, 2, 0, 3, 4).contiguous()
    output = output.view(batch, local_seq, local_heads * world_size, head_dim)
    
    return output


def ulysses_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, 
                      world_size: int) -> torch.Tensor:
    """
    Ulysses-style attention with sequence parallelism.
    
    Input Q, K, V: (batch, local_seq, num_heads, head_dim) - sequence-parallel layout
    Output: (batch, local_seq, num_heads, head_dim) - sequence-parallel layout
    
    Steps:
    1. All-to-all to get full sequence, local heads
    2. Standard attention (each GPU handles subset of heads)
    3. All-to-all to return to sequence-parallel layout
    """
    # Step 1: Transpose to head-parallel layout
    # Now each GPU has full sequence but only local_heads
    Q_full = all_to_all_seq_to_head(Q, world_size)
    K_full = all_to_all_seq_to_head(K, world_size)
    V_full = all_to_all_seq_to_head(V, world_size)
    
    # Step 2: Standard attention on local heads
    # Q_full, K_full, V_full: (batch, full_seq, local_heads, head_dim)
    batch, full_seq, local_heads, head_dim = Q_full.shape
    
    # Transpose for attention: (batch, local_heads, full_seq, head_dim)
    Q_t = Q_full.transpose(1, 2)
    K_t = K_full.transpose(1, 2)
    V_t = V_full.transpose(1, 2)
    
    # Scaled dot-product attention
    scale = head_dim ** -0.5
    scores = torch.matmul(Q_t, K_t.transpose(-2, -1)) * scale
    attn_weights = F.softmax(scores, dim=-1)
    attn_output = torch.matmul(attn_weights, V_t)
    
    # Back to (batch, full_seq, local_heads, head_dim)
    attn_output = attn_output.transpose(1, 2)
    
    # Step 3: Transpose back to sequence-parallel layout
    output = all_to_all_head_to_seq(attn_output, world_size)
    
    return output


def main():
    import os
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    if rank == 0:
        print("=" * 65)
        print("DeepSpeed-Ulysses Sequence Parallelism Demo")
        print("=" * 65)
        print(f"\nWorld size: {world_size} GPUs")
    
    # Configuration
    batch_size = 2
    full_seq_len = 8192
    num_heads = 8  # Must be divisible by world_size
    head_dim = 64
    
    local_seq_len = full_seq_len // world_size
    local_heads = num_heads // world_size
    
    if rank == 0:
        print(f"\nConfiguration:")
        print(f"  Full sequence length: {full_seq_len}")
        print(f"  Local sequence per GPU: {local_seq_len}")
        print(f"  Total attention heads: {num_heads}")
        print(f"  Local heads per GPU (after transpose): {local_heads}")
        print(f"  Head dimension: {head_dim}")
    
    # Each GPU starts with a chunk of the sequence (all heads)
    Q = torch.randn(batch_size, local_seq_len, num_heads, head_dim,
                    device=f"cuda:{rank}", dtype=torch.float16)
    K = torch.randn(batch_size, local_seq_len, num_heads, head_dim,
                    device=f"cuda:{rank}", dtype=torch.float16)
    V = torch.randn(batch_size, local_seq_len, num_heads, head_dim,
                    device=f"cuda:{rank}", dtype=torch.float16)
    
    dist.barrier()
    if rank == 0:
        print(f"\n" + "-" * 65)
        print("Initial layout (sequence-parallel):")
        print(f"  Q, K, V shape per GPU: ({batch_size}, {local_seq_len}, {num_heads}, {head_dim})")
        print(f"  Each GPU holds: {local_seq_len} tokens, all {num_heads} heads")
    
    # Demonstrate the all-to-all transpose
    Q_transposed = all_to_all_seq_to_head(Q, world_size)
    
    dist.barrier()
    if rank == 0:
        print(f"\n" + "-" * 65)
        print("After all-to-all (head-parallel):")
        print(f"  Q shape per GPU: {tuple(Q_transposed.shape)}")
        print(f"  Each GPU holds: {full_seq_len} tokens, only {local_heads} heads")
        print(f"  -> Now standard attention can be computed!")
    
    # Run full Ulysses attention
    output = ulysses_attention(Q, K, V, world_size)
    
    dist.barrier()
    if rank == 0:
        print(f"\n" + "-" * 65)
        print("After Ulysses attention:")
        print(f"  Output shape per GPU: {tuple(output.shape)}")
        print(f"  Back to sequence-parallel layout")
        
        print(f"\n" + "-" * 65)
        print("Summary:")
        print(f"  1. Start: each GPU has {local_seq_len} tokens × {num_heads} heads")
        print(f"  2. All-to-all #1: transpose to {full_seq_len} tokens × {local_heads} heads")
        print(f"  3. Attention: standard self-attention on local heads (no partial softmax!)")
        print(f"  4. All-to-all #2: transpose back to {local_seq_len} tokens × {num_heads} heads")
        print(f"\nKey advantage: No complex partial softmax accumulation like ring attention")
        print("=" * 65)
    
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
