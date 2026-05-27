"""
Expert Parallelism Demo

This script demonstrates the core concepts of Expert Parallelism (EP) for
Mixture-of-Experts (MoE) models: token routing, all-to-all communication,
and expert computation.

The demo shows:
1. How tokens are routed to different experts based on router scores
2. The all-to-all communication pattern for dispatching tokens
3. Expert computation on local tokens
4. All-to-all to return results to original GPUs

Usage:
    # Run with 2 GPUs (2 experts)
    torchrun --nproc_per_node=2 expert_parallel_demo.py

    # Run with 4 GPUs (4 experts)
    torchrun --nproc_per_node=4 expert_parallel_demo.py

Requirements:
    - PyTorch with NCCL backend
    - Multiple GPUs
"""

import torch
import torch.nn as nn
import torch.distributed as dist


class SimpleExpert(nn.Module):
    """A simple expert: just a 2-layer MLP."""
    def __init__(self, hidden_dim: int, expert_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, expert_dim)
        self.fc2 = nn.Linear(expert_dim, hidden_dim)
        self.act = nn.GELU()
    
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class SimpleRouter(nn.Module):
    """A simple top-k router."""
    def __init__(self, hidden_dim: int, num_experts: int):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
    
    def forward(self, x, top_k: int = 2):
        # x: (batch * seq, hidden_dim)
        logits = self.gate(x)  # (batch * seq, num_experts)
        scores = torch.softmax(logits, dim=-1)
        top_scores, top_indices = torch.topk(scores, top_k, dim=-1)
        # Normalize top-k scores
        top_scores = top_scores / top_scores.sum(dim=-1, keepdim=True)
        return top_scores, top_indices


def expert_parallel_forward(
    tokens: torch.Tensor,
    router: SimpleRouter,
    local_expert: SimpleExpert,
    world_size: int,
    rank: int,
    top_k: int = 2
) -> torch.Tensor:
    """
    Perform expert-parallel MoE forward pass.
    
    Args:
        tokens: (num_tokens, hidden_dim) - local tokens on this GPU
        router: Router module to compute expert assignments
        local_expert: The expert hosted on this GPU
        world_size: Number of GPUs (= number of experts)
        rank: This GPU's rank (= this expert's index)
        top_k: Number of experts each token is routed to
    
    Returns:
        output: (num_tokens, hidden_dim) - processed tokens
    """
    num_tokens, hidden_dim = tokens.shape
    device = tokens.device
    
    # Step 1: Route tokens to experts
    scores, expert_indices = router(tokens, top_k)
    # scores: (num_tokens, top_k)
    # expert_indices: (num_tokens, top_k)
    
    if rank == 0:
        print(f"\n  Router assigned tokens to experts:")
        expert_counts = torch.zeros(world_size, device=device)
        for e in range(world_size):
            count = (expert_indices == e).sum().item()
            expert_counts[e] = count
            print(f"    Expert {e}: {int(count)} token-expert pairs")
    
    # Step 2: Prepare tokens for all-to-all dispatch
    # For simplicity, we'll send each token to its top-1 expert only
    # (Real implementations handle top-k more carefully)
    top1_expert = expert_indices[:, 0]  # (num_tokens,)
    top1_score = scores[:, 0]  # (num_tokens,)
    
    # Count how many tokens go to each expert
    send_counts = torch.zeros(world_size, dtype=torch.long, device=device)
    for e in range(world_size):
        send_counts[e] = (top1_expert == e).sum()
    
    # Gather receive counts from all GPUs
    recv_counts = torch.zeros(world_size, dtype=torch.long, device=device)
    dist.all_to_all_single(recv_counts, send_counts)
    
    if rank == 0:
        print(f"\n  Token dispatch (top-1 only for demo):")
        print(f"    GPU 0 sends: {send_counts.tolist()}")
        print(f"    GPU 0 receives: {recv_counts.tolist()}")
    
    # Step 3: Reorder tokens by destination expert
    sort_indices = torch.argsort(top1_expert)
    sorted_tokens = tokens[sort_indices]
    sorted_scores = top1_score[sort_indices]
    
    # Step 4: All-to-all to dispatch tokens to experts
    # Prepare send buffer (tokens grouped by destination)
    send_splits = send_counts.tolist()
    recv_splits = recv_counts.tolist()
    
    total_recv = sum(recv_splits)
    recv_buffer = torch.empty(total_recv, hidden_dim, device=device, dtype=tokens.dtype)
    
    # All-to-all for tokens
    dist.all_to_all(
        list(recv_buffer.split(recv_splits)),
        list(sorted_tokens.split(send_splits))
    )
    
    dist.barrier()
    if rank == 0:
        print(f"\n  After all-to-all dispatch:")
        print(f"    GPU 0 now has {total_recv} tokens to process with Expert 0")
    
    # Step 5: Process tokens with local expert
    if total_recv > 0:
        expert_output = local_expert(recv_buffer)
    else:
        expert_output = recv_buffer  # Empty tensor
    
    # Step 6: All-to-all to return results
    # Reverse the communication pattern
    result_buffer = torch.empty(num_tokens, hidden_dim, device=device, dtype=tokens.dtype)
    
    dist.all_to_all(
        list(result_buffer.split(send_splits)),
        list(expert_output.split(recv_splits))
    )
    
    # Step 7: Unsort to restore original token order
    unsort_indices = torch.argsort(sort_indices)
    output = result_buffer[unsort_indices]
    
    # Weight by router scores (for top-1, this is just scaling)
    output = output * top1_score.unsqueeze(-1)
    
    return output


def main():
    import os
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", device_id=torch.device(f"cuda:{local_rank}"))
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = f"cuda:{rank}"
    
    if rank == 0:
        print("=" * 65)
        print("Expert Parallelism Demo")
        print("=" * 65)
        print(f"\nWorld size: {world_size} GPUs = {world_size} experts")
    
    # Configuration
    num_tokens = 16  # Tokens per GPU
    hidden_dim = 256
    expert_dim = 512
    top_k = 2
    
    if rank == 0:
        print(f"\nConfiguration:")
        print(f"  Tokens per GPU: {num_tokens}")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Expert dim: {expert_dim}")
        print(f"  Top-k routing: {top_k}")
    
    # Each GPU hosts one expert
    local_expert = SimpleExpert(hidden_dim, expert_dim).to(device)
    
    # Shared router (in practice, router weights are replicated)
    router = SimpleRouter(hidden_dim, world_size).to(device)
    
    # Synchronize router weights across GPUs
    for param in router.parameters():
        dist.broadcast(param.data, src=0)
    
    # Create local tokens
    torch.manual_seed(42 + rank)  # Different tokens per GPU
    tokens = torch.randn(num_tokens, hidden_dim, device=device)
    
    dist.barrier()
    if rank == 0:
        print(f"\n" + "-" * 65)
        print("Running Expert-Parallel Forward Pass")
        print("-" * 65)
    
    # Run expert-parallel forward
    output = expert_parallel_forward(
        tokens, router, local_expert, world_size, rank, top_k=1
    )
    
    dist.barrier()
    if rank == 0:
        print(f"\n" + "-" * 65)
        print("Summary")
        print("-" * 65)
        print(f"  Input shape per GPU: ({num_tokens}, {hidden_dim})")
        print(f"  Output shape per GPU: ({output.shape[0]}, {output.shape[1]})")
        print(f"\n  Key steps:")
        print(f"    1. Router computes expert assignments for each token")
        print(f"    2. All-to-all dispatches tokens to their assigned experts")
        print(f"    3. Each GPU processes tokens with its local expert")
        print(f"    4. All-to-all returns results to original GPUs")
        print(f"    5. Results weighted by router scores")
        print("=" * 65)
    
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
