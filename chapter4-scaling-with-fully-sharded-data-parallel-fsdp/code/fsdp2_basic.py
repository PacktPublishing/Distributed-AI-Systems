"""
FSDP2 minimal example from Mark Saroufim's tweet
Courtesy of Andrew Gu

This version uses a standalone Transformer implementation instead of 
internal PyTorch testing modules.

Usage:
    torchrun --standalone --nproc_per_node=2 code/chapter4/fsdp2_basic.py
"""
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy


class TransformerBlock(nn.Module):
    """A simple transformer block for demonstration"""
    def __init__(self, dim=2048, n_heads=32, dropout_p=0.0):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        head_dim = dim // n_heads
        
        # Self-attention
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_out = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout_p)
    
    def forward(self, x):
        # Self-attention
        residual = x
        x = self.norm1(x)
        qkv = self.qkv(x)
        B, T, _ = x.shape
        qkv = qkv.reshape(B, T, 3, self.n_heads, self.dim // self.n_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, n_heads, T, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Simple attention (no masking for this example)
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.dim // self.n_heads) ** 0.5
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        attn_out = torch.matmul(attn, v)
        attn_out = attn_out.transpose(1, 2).reshape(B, T, self.dim)
        x = self.attn_out(attn_out)
        x = self.dropout(x)
        x = x + residual
        
        # FFN
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.dropout(x)
        x = x + residual
        
        return x


class Transformer(nn.Module):
    """A simple transformer model for demonstration"""
    def __init__(self, n_layers=12, vocab_size=50304, n_heads=32, dim=2048, 
                 max_seq_len=2048, dropout_p=0.0):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.max_seq_len = max_seq_len
        
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_seq_len, dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(dim=dim, n_heads=n_heads, dropout_p=dropout_p)
            for _ in range(n_layers)
        ])
        self.ln_out = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)
    
    def forward(self, idx):
        B, T = idx.shape
        assert T <= self.max_seq_len, f"Sequence length {T} exceeds max_seq_len {self.max_seq_len}"
        
        # Token and position embeddings
        tok_emb = self.token_emb(idx)
        pos = torch.arange(0, T, device=idx.device)
        pos_emb = self.pos_emb(pos)
        x = tok_emb + pos_emb
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Output head
        x = self.ln_out(x)
        logits = self.head(x)
        return logits


def main():
    torch.manual_seed(42)
    
    # Model configuration matching the original tweet
    n_layers = 12
    vocab_size = 50304
    n_heads = 32
    dim = 2048
    max_seq_len = 2048
    dropout_p = 0.0
    
    model = Transformer(
        n_layers=n_layers,
        vocab_size=vocab_size,
        n_heads=n_heads,
        dim=dim,
        max_seq_len=max_seq_len,
        dropout_p=dropout_p,
    ).cuda()
    
    # Apply FSDP2 with mixed precision
    mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16)
    fsdp_cfg = {"mp_policy": mp_policy}
    
    # Apply fully_shard to each TransformerBlock and then the whole model
    for module in model.modules():
        if isinstance(module, TransformerBlock):
            fully_shard(module, **fsdp_cfg)
    fully_shard(model, **fsdp_cfg)
    
    # Setup optimizer and training step
    optim = torch.optim.AdamW(model.parameters(), lr=1e-2)
    inp = torch.randint(0, vocab_size, (8, 1024), device="cuda")
    
    # Forward and backward
    model(inp).sum().backward()
    optim.step()
    
    if dist.get_rank() == 0:
        print("FSDP2 training step completed successfully!")


if __name__ == "__main__":
    dist.init_process_group(backend="nccl")
    gpu_id = int(os.environ["LOCAL_RANK"])
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(device)
    rank = gpu_id
    main()
    dist.destroy_process_group()
