"""
Sequence Parallelism and Context Parallelism Demo

This script demonstrates the core concepts behind sequence parallelism and
context parallelism (ring attention) for handling long sequences in distributed
training.

Three modes:
  - memory: Shows how activation memory scales with sequence length (single GPU)
  - sequence_parallel: Demonstrates splitting activations along sequence dim (2 GPUs)
  - ring_attention: Demonstrates the ring attention pattern for context parallelism (2 GPUs)

Usage:
    # Show memory scaling (single GPU)
    python sequence_parallel_demo.py --mode memory

    # Demonstrate sequence parallelism (2 GPUs)
    torchrun --nproc_per_node=2 sequence_parallel_demo.py --mode sequence_parallel

    # Demonstrate ring attention concept (2 GPUs)
    torchrun --nproc_per_node=2 sequence_parallel_demo.py --mode ring_attention
"""

import argparse
import torch
import torch.nn as nn


def demo_memory_scaling():
    """Show how activation memory grows with sequence length."""
    print("=" * 60)
    print("Activation Memory Scaling with Sequence Length")
    print("=" * 60)
    
    hidden_dim = 4096
    num_layers = 32
    batch_size = 1
    bytes_per_element = 2  # FP16
    
    print(f"\nModel config: hidden_dim={hidden_dim}, layers={num_layers}, batch=1, FP16")
    print("-" * 60)
    print(f"{'Seq Length':>12} | {'Per Layer':>12} | {'Total (32L)':>12} | {'Note'}")
    print("-" * 60)
    
    for seq_len in [2048, 4096, 8192, 16384, 32768, 65536, 131072]:
        per_layer_bytes = batch_size * seq_len * hidden_dim * bytes_per_element
        total_bytes = per_layer_bytes * num_layers
        per_layer_gb = per_layer_bytes / (1024**3)
        total_gb = total_bytes / (1024**3)
        
        note = ""
        if total_gb > 80:
            note = "Exceeds H100 80GB"
        elif total_gb > 40:
            note = "Exceeds A100 40GB"
        
        print(f"{seq_len:>12,} | {per_layer_gb:>10.2f}GB | {total_gb:>10.1f}GB | {note}")
    
    print("-" * 60)
    print("\nConclusion: Long sequences require sequence/context parallelism")
    print("to distribute activation memory across multiple GPUs.")


def demo_sequence_parallel():
    """Demonstrate sequence parallelism: split activations along sequence dim."""
    import torch.distributed as dist
    
    rank = int(__import__("os").environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{rank}"))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    if rank == 0:
        print("=" * 60)
        print("Sequence Parallelism Demo")
        print("=" * 60)
        print(f"\nWorld size: {world_size}")
    
    # Simulate a full sequence
    batch_size = 2
    full_seq_len = 8192
    hidden_dim = 1024
    local_seq_len = full_seq_len // world_size
    
    # Each GPU holds only its portion of the sequence
    local_activations = torch.randn(
        batch_size, local_seq_len, hidden_dim,
        device=f"cuda:{rank}", dtype=torch.float16
    )
    
    if rank == 0:
        print(f"\nFull sequence: ({batch_size}, {full_seq_len}, {hidden_dim})")
        print(f"Local chunk per GPU: ({batch_size}, {local_seq_len}, {hidden_dim})")
        print(f"Memory per GPU: {local_activations.numel() * 2 / 1024**2:.1f} MB")
        print(f"Memory savings: {world_size}x reduction vs full sequence")
    
    # LayerNorm operates on local chunk (no communication needed)
    layer_norm = nn.LayerNorm(hidden_dim).cuda(rank).half()
    local_output = layer_norm(local_activations)
    
    dist.barrier()
    if rank == 0:
        print(f"\nLayerNorm applied locally - no cross-GPU communication!")
        print("Each GPU processes its sequence chunk independently.")
    
    dist.destroy_process_group()


def demo_ring_attention():
    """Demonstrate ring attention pattern for context parallelism."""
    import torch.distributed as dist
    
    rank = int(__import__("os").environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(rank)
    dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{rank}"))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    
    if rank == 0:
        print("=" * 60)
        print("Ring Attention Demo (Context Parallelism)")
        print("=" * 60)
        print(f"\nWorld size: {world_size}")
    
    # Each GPU holds local Q, K, V chunks
    batch_size = 1
    local_seq_len = 4096  # Each GPU has 4K tokens
    num_heads = 8
    head_dim = 64
    
    # Local Q, K, V
    Q_local = torch.randn(batch_size, num_heads, local_seq_len, head_dim,
                          device=f"cuda:{rank}", dtype=torch.float16)
    K_local = torch.randn(batch_size, num_heads, local_seq_len, head_dim,
                          device=f"cuda:{rank}", dtype=torch.float16)
    V_local = torch.randn(batch_size, num_heads, local_seq_len, head_dim,
                          device=f"cuda:{rank}", dtype=torch.float16)
    
    if rank == 0:
        print(f"\nEach GPU holds:")
        print(f"  Q: ({batch_size}, {num_heads}, {local_seq_len}, {head_dim})")
        print(f"  K: ({batch_size}, {num_heads}, {local_seq_len}, {head_dim})")
        print(f"  V: ({batch_size}, {num_heads}, {local_seq_len}, {head_dim})")
        print(f"\nTotal sequence length: {local_seq_len * world_size}")
    
    # Ring attention: rotate KV around the ring
    # After world_size steps, every Q has seen every KV
    
    K_recv = torch.empty_like(K_local)
    V_recv = torch.empty_like(V_local)
    
    # Current KV to process
    K_curr = K_local.clone()
    V_curr = V_local.clone()
    
    # Accumulator for attention output
    output_accum = torch.zeros_like(Q_local)
    
    dist.barrier()
    if rank == 0:
        print("\n" + "-" * 60)
        print("Ring Attention Steps:")
        print("-" * 60)
    
    for step in range(world_size):
        # Compute attention with current KV chunk
        # (simplified - real impl uses flash attention)
        scores = torch.matmul(Q_local, K_curr.transpose(-2, -1)) / (head_dim ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        local_output = torch.matmul(attn_weights, V_curr)
        
        # Accumulate (simplified - real impl tracks softmax normalization)
        output_accum += local_output
        
        dist.barrier()
        if rank == 0:
            src_rank = (rank - step) % world_size
            print(f"  Step {step}: GPU {rank} attends to KV from GPU {src_rank}")
        
        # Ring pass: send KV to next, receive from previous
        # Use batch_isend_irecv to avoid P2P deadlock (isend/irecv order can hang)
        if step < world_size - 1:
            next_rank = (rank + 1) % world_size
            prev_rank = (rank - 1) % world_size

            reqs = dist.batch_isend_irecv([
                dist.P2POp(dist.isend, K_curr, next_rank),
                dist.P2POp(dist.irecv, K_recv, prev_rank),
            ])
            for req in reqs:
                req.wait()

            reqs = dist.batch_isend_irecv([
                dist.P2POp(dist.isend, V_curr, next_rank),
                dist.P2POp(dist.irecv, V_recv, prev_rank),
            ])
            for req in reqs:
                req.wait()

            K_curr = K_recv.clone()
            V_curr = V_recv.clone()
    
    dist.barrier()
    if rank == 0:
        print("-" * 60)
        print(f"\nAfter {world_size} steps:")
        print(f"  - Every Q has attended to all {local_seq_len * world_size} tokens")
        print(f"  - No GPU ever held the full {local_seq_len * world_size}-token KV")
        print(f"  - Memory per GPU: {world_size}x reduction vs full attention")
    
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description="Sequence/Context Parallelism Demo")
    parser.add_argument("--mode", type=str, default="memory",
                        choices=["memory", "sequence_parallel", "ring_attention"],
                        help="Demo mode to run")
    args = parser.parse_args()
    
    if args.mode == "memory":
        demo_memory_scaling()
    elif args.mode == "sequence_parallel":
        demo_sequence_parallel()
    elif args.mode == "ring_attention":
        demo_ring_attention()


if __name__ == "__main__":
    main()
