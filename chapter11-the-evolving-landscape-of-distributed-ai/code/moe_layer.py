"""
Mixture of Experts (MoE) Layer Implementation

This module provides basic MoE layer implementations including:
- Basic MoE with top-k routing
- Expert parallelism across GPUs
- Load balancing with auxiliary loss
"""

import torch
import torch.nn as nn
import torch.distributed as dist


class MoELayer(nn.Module):
    """Basic Mixture of Experts layer with top-k routing."""
    
    def __init__(self, d_model: int, num_experts: int, expert_capacity: int, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.expert_capacity = expert_capacity
        
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Linear(d_model * 4, d_model)
            ) for _ in range(num_experts)
        ])
        
        self.router = nn.Linear(d_model, num_experts)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        
        router_logits = self.router(x)
        top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)
        top_k_probs = torch.softmax(top_k_logits, dim=-1)
        
        x_flat = x.view(-1, d_model)
        top_k_indices_flat = top_k_indices.view(-1, self.top_k)
        top_k_probs_flat = top_k_probs.view(-1, self.top_k, 1)
        
        expert_outputs = []
        for expert_id in range(self.num_experts):
            mask = (top_k_indices_flat == expert_id).any(dim=1)
            if mask.sum() == 0:
                continue
            
            expert_tokens = x_flat[mask]
            expert_weights = top_k_probs_flat[mask].squeeze(-1)
            expert_out = self.experts[expert_id](expert_tokens)
            expert_out = expert_out * expert_weights.sum(dim=1, keepdim=True)
            expert_outputs.append((expert_out, mask))
        
        output = torch.zeros_like(x_flat)
        for expert_out, mask in expert_outputs:
            output[mask] += expert_out
        
        return output.view(batch_size, seq_len, d_model)


class ExpertParallelMoE(nn.Module):
    """MoE layer with experts distributed across GPUs."""
    
    def __init__(self, d_model: int, num_experts: int, world_size: int, rank: int):
        super().__init__()
        self.num_experts = num_experts
        self.world_size = world_size
        self.rank = rank
        
        experts_per_rank = num_experts // world_size
        self.local_expert_start = rank * experts_per_rank
        self.local_expert_end = (rank + 1) * experts_per_rank
        self.local_num_experts = experts_per_rank
        
        self.local_experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Linear(d_model * 4, d_model)
            ) for _ in range(self.local_num_experts)
        ])
        
        self.router = nn.Linear(d_model, num_experts)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, d_model = x.shape
        
        router_logits = self.router(x)
        top_k_logits, top_k_indices = torch.topk(router_logits, k=2, dim=-1)
        top_k_probs = torch.softmax(top_k_logits, dim=-1)
        
        x_flat = x.view(-1, d_model)
        top_k_indices_flat = top_k_indices.view(-1, 2)
        top_k_probs_flat = top_k_probs.view(-1, 2, 1)
        
        local_outputs = []
        for local_expert_id in range(self.local_num_experts):
            global_expert_id = self.local_expert_start + local_expert_id
            mask = (top_k_indices_flat == global_expert_id).any(dim=1)
            if mask.sum() == 0:
                continue
            
            expert_tokens = x_flat[mask]
            expert_weights = top_k_probs_flat[mask].squeeze(-1)
            expert_out = self.local_experts[local_expert_id](expert_tokens)
            expert_out = expert_out * expert_weights.sum(dim=1, keepdim=True)
            local_outputs.append((expert_out, mask, global_expert_id))
        
        output = torch.zeros_like(x_flat)
        for expert_out, mask, expert_id in local_outputs:
            output[mask] += expert_out
        
        return output.view(batch_size, seq_len, d_model)


class LoadBalancedMoE(nn.Module):
    """MoE layer with load balancing auxiliary loss."""
    
    def __init__(self, d_model: int, num_experts: int, top_k: int = 2, 
                 load_balance_weight: float = 0.01):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.load_balance_weight = load_balance_weight
        
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.GELU(),
                nn.Linear(d_model * 4, d_model)
            ) for _ in range(num_experts)
        ])
        self.router = nn.Linear(d_model, num_experts)
    
    def compute_load_balance_loss(self, router_probs: torch.Tensor) -> torch.Tensor:
        avg_probs = router_probs.mean(dim=[0, 1])
        load_balance_loss = self.num_experts * torch.sum(avg_probs ** 2)
        return load_balance_loss
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, d_model = x.shape
        
        router_logits = self.router(x)
        router_probs = torch.softmax(router_logits, dim=-1)
        top_k_logits, top_k_indices = torch.topk(router_logits, k=self.top_k, dim=-1)
        top_k_probs = torch.softmax(top_k_logits, dim=-1)
        
        x_flat = x.view(-1, d_model)
        top_k_indices_flat = top_k_indices.view(-1, self.top_k)
        top_k_probs_flat = top_k_probs.view(-1, self.top_k, 1)
        
        output = torch.zeros_like(x_flat)
        for expert_id in range(self.num_experts):
            mask = (top_k_indices_flat == expert_id).any(dim=1)
            if mask.sum() == 0:
                continue
            
            expert_tokens = x_flat[mask]
            expert_weights = top_k_probs_flat[mask].squeeze(-1)
            expert_out = self.experts[expert_id](expert_tokens)
            expert_out = expert_out * expert_weights.sum(dim=1, keepdim=True)
            output[mask] += expert_out
        
        output = output.view(batch_size, seq_len, d_model)
        load_balance_loss = self.compute_load_balance_loss(router_probs)
        
        return output, load_balance_loss


if __name__ == "__main__":
    # Test basic MoE
    moe = MoELayer(d_model=256, num_experts=8, expert_capacity=32, top_k=2)
    x = torch.randn(4, 128, 256)
    output = moe(x)
    print(f"Basic MoE - Input: {x.shape}, Output: {output.shape}")
    
    # Test load balanced MoE
    lb_moe = LoadBalancedMoE(d_model=256, num_experts=8, top_k=2)
    output, lb_loss = lb_moe(x)
    print(f"Load Balanced MoE - Output: {output.shape}, LB Loss: {lb_loss.item():.4f}")
