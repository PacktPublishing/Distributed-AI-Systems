"""
Advanced Parallelism Strategies

This module provides implementations for:
- Hierarchical parallelism (DP + TP + PP)
- Sequence parallelism for long contexts
- Ring attention for distributed attention computation
"""

import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Optional


class HierarchicalParallelism:
    """
    Combine data parallelism, tensor parallelism, and pipeline parallelism.
    
    For a cluster with world_size GPUs, we partition into:
    - dp_size data parallel groups (different data, same model)
    - tp_size tensor parallel groups (same data, sharded model)
    - pp_size pipeline stages (different layers)
    
    Constraint: dp_size * tp_size * pp_size == world_size
    """
    
    def __init__(self, world_size: int, dp_size: int, tp_size: int, pp_size: int):
        self.world_size = world_size
        self.dp_size = dp_size
        self.tp_size = tp_size
        self.pp_size = pp_size
        
        assert dp_size * tp_size * pp_size == world_size, \
            f"dp_size({dp_size}) * tp_size({tp_size}) * pp_size({pp_size}) != world_size({world_size})"
        
        self.dp_groups = self._create_dp_groups()
        self.tp_groups = self._create_tp_groups()
        self.pp_groups = self._create_pp_groups()
    
    def _create_dp_groups(self) -> list[list[int]]:
        """Create data parallel process groups."""
        groups = []
        for dp_id in range(self.dp_size):
            ranks = []
            for tp_id in range(self.tp_size):
                for pp_id in range(self.pp_size):
                    rank = dp_id * self.tp_size * self.pp_size + tp_id * self.pp_size + pp_id
                    ranks.append(rank)
            groups.append(ranks)
        return groups
    
    def _create_tp_groups(self) -> list[list[int]]:
        """Create tensor parallel process groups."""
        groups = []
        for dp_id in range(self.dp_size):
            for pp_id in range(self.pp_size):
                ranks = []
                for tp_id in range(self.tp_size):
                    rank = dp_id * self.tp_size * self.pp_size + tp_id * self.pp_size + pp_id
                    ranks.append(rank)
                groups.append(ranks)
        return groups
    
    def _create_pp_groups(self) -> list[list[int]]:
        """Create pipeline parallel process groups."""
        groups = []
        for dp_id in range(self.dp_size):
            for tp_id in range(self.tp_size):
                ranks = []
                for pp_id in range(self.pp_size):
                    rank = dp_id * self.tp_size * self.pp_size + tp_id * self.pp_size + pp_id
                    ranks.append(rank)
                groups.append(ranks)
        return groups
    
    def get_rank_info(self, global_rank: int) -> dict:
        """Get parallelism info for a given global rank."""
        dp_id = global_rank // (self.tp_size * self.pp_size)
        remainder = global_rank % (self.tp_size * self.pp_size)
        tp_id = remainder // self.pp_size
        pp_id = remainder % self.pp_size
        
        return {
            'global_rank': global_rank,
            'dp_rank': dp_id,
            'tp_rank': tp_id,
            'pp_rank': pp_id,
            'dp_group': self.dp_groups[dp_id],
            'tp_group': self.tp_groups[dp_id * self.pp_size + pp_id],
            'pp_group': self.pp_groups[dp_id * self.tp_size + tp_id],
        }


class SequenceParallelAttention(nn.Module):
    """
    Attention layer with sequence parallelism for long contexts.
    
    Splits the sequence dimension across GPUs. Each GPU computes
    attention for its local sequence chunk but needs full K, V
    from all GPUs via all-gather.
    """
    
    def __init__(self, d_model: int, num_heads: int, seq_parallel_size: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.seq_parallel_size = seq_parallel_size
        self.head_dim = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
    
    def forward(self, x: torch.Tensor, seq_rank: int, 
                seq_world_size: int, process_group: Optional[dist.ProcessGroup] = None
                ) -> torch.Tensor:
        batch, seq_len, d_model = x.shape
        
        # Each rank processes a chunk of the sequence
        local_seq_len = seq_len // seq_world_size
        start_idx = seq_rank * local_seq_len
        end_idx = (seq_rank + 1) * local_seq_len
        x_local = x[:, start_idx:end_idx, :]
        
        # Compute local Q, K, V
        q_local = self.q_proj(x_local)
        k_local = self.k_proj(x_local)
        v_local = self.v_proj(x_local)
        
        # Reshape for multi-head attention
        q_local = q_local.view(batch, local_seq_len, self.num_heads, self.head_dim)
        k_local = k_local.view(batch, local_seq_len, self.num_heads, self.head_dim)
        v_local = v_local.view(batch, local_seq_len, self.num_heads, self.head_dim)
        
        # All-gather K and V from all sequence parallel ranks
        k_full = self._all_gather_sequence(k_local, seq_world_size, process_group)
        v_full = self._all_gather_sequence(v_local, seq_world_size, process_group)
        
        # Compute attention: Q_local @ K_full^T
        # q_local: [batch, local_seq_len, num_heads, head_dim]
        # k_full: [batch, full_seq_len, num_heads, head_dim]
        scores = torch.einsum('blhd,bshd->bhls', q_local, k_full) / (self.head_dim ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Apply attention to values
        # attn_weights: [batch, num_heads, local_seq_len, full_seq_len]
        # v_full: [batch, full_seq_len, num_heads, head_dim]
        attn_output = torch.einsum('bhls,bshd->blhd', attn_weights, v_full)
        attn_output = attn_output.contiguous().view(batch, local_seq_len, d_model)
        
        output = self.o_proj(attn_output)
        return output
    
    def _all_gather_sequence(self, tensor: torch.Tensor, world_size: int,
                             process_group: Optional[dist.ProcessGroup] = None
                             ) -> torch.Tensor:
        """All-gather sequence chunks from all ranks."""
        if not dist.is_initialized() or world_size == 1:
            return tensor
        
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.all_gather(gathered, tensor, group=process_group)
        return torch.cat(gathered, dim=1)


class RingAttention(nn.Module):
    """
    Ring attention for memory-efficient long context processing.
    
    Instead of all-gathering full K, V, uses ring communication
    to pass K, V chunks around the ring while computing partial
    attention scores.
    """
    
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.o_proj = nn.Linear(d_model, d_model)
    
    def forward(self, x: torch.Tensor, ring_rank: int, ring_size: int,
                process_group: Optional[dist.ProcessGroup] = None) -> torch.Tensor:
        batch, local_seq_len, d_model = x.shape
        
        q = self.q_proj(x).view(batch, local_seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch, local_seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch, local_seq_len, self.num_heads, self.head_dim)
        
        # Initialize output accumulator and normalization factor
        output_acc = torch.zeros(batch, local_seq_len, self.num_heads, self.head_dim, 
                                 device=x.device, dtype=x.dtype)
        lse_acc = torch.full((batch, self.num_heads, local_seq_len), float('-inf'),
                            device=x.device, dtype=x.dtype)
        
        k_recv = k.clone()
        v_recv = v.clone()
        
        for step in range(ring_size):
            # Compute attention with current K, V chunk
            scores = torch.einsum('blhd,bshd->bhls', q, k_recv) / (self.head_dim ** 0.5)
            
            # Online softmax update
            max_scores = scores.max(dim=-1, keepdim=True).values.squeeze(-1)
            new_lse = torch.logsumexp(scores, dim=-1)
            
            # Update running max and normalization
            max_combined = torch.maximum(lse_acc, new_lse)
            exp_old = torch.exp(lse_acc - max_combined)
            exp_new = torch.exp(new_lse - max_combined)
            
            # Update output
            attn_weights = torch.softmax(scores, dim=-1)
            chunk_output = torch.einsum('bhls,bshd->blhd', attn_weights, v_recv)
            
            output_acc = output_acc * exp_old.unsqueeze(-1).transpose(1, 2) + \
                        chunk_output * exp_new.unsqueeze(-1).transpose(1, 2)
            lse_acc = max_combined + torch.log(exp_old + exp_new)
            
            # Ring send/recv for next iteration
            if step < ring_size - 1 and dist.is_initialized():
                k_recv, v_recv = self._ring_exchange(k_recv, v_recv, ring_rank, 
                                                     ring_size, process_group)
        
        # Normalize output
        output = output_acc / torch.exp(lse_acc).unsqueeze(-1).transpose(1, 2)
        output = output.contiguous().view(batch, local_seq_len, d_model)
        
        return self.o_proj(output)
    
    def _ring_exchange(self, k: torch.Tensor, v: torch.Tensor, 
                       rank: int, world_size: int,
                       process_group: Optional[dist.ProcessGroup] = None
                       ) -> tuple[torch.Tensor, torch.Tensor]:
        """Exchange K, V with neighbors in the ring."""
        send_rank = (rank + 1) % world_size
        recv_rank = (rank - 1 + world_size) % world_size
        
        k_recv = torch.empty_like(k)
        v_recv = torch.empty_like(v)
        
        # Async send/recv
        send_k = dist.isend(k, send_rank, group=process_group)
        send_v = dist.isend(v, send_rank, group=process_group)
        recv_k = dist.irecv(k_recv, recv_rank, group=process_group)
        recv_v = dist.irecv(v_recv, recv_rank, group=process_group)
        
        send_k.wait()
        send_v.wait()
        recv_k.wait()
        recv_v.wait()
        
        return k_recv, v_recv


if __name__ == "__main__":
    # Test hierarchical parallelism configuration
    hp = HierarchicalParallelism(world_size=16, dp_size=2, tp_size=4, pp_size=2)
    
    print("Hierarchical Parallelism Configuration:")
    print(f"  World size: {hp.world_size}")
    print(f"  DP size: {hp.dp_size}, TP size: {hp.tp_size}, PP size: {hp.pp_size}")
    
    for rank in [0, 1, 8, 15]:
        info = hp.get_rank_info(rank)
        print(f"\n  Rank {rank}: DP={info['dp_rank']}, TP={info['tp_rank']}, PP={info['pp_rank']}")
    
    # Test sequence parallel attention (single GPU simulation)
    sp_attn = SequenceParallelAttention(d_model=256, num_heads=8, seq_parallel_size=4)
    x = torch.randn(2, 1024, 256)
    output = sp_attn(x, seq_rank=0, seq_world_size=1)
    print(f"\nSequence Parallel Attention - Input: {x.shape}, Output: {output.shape}")
