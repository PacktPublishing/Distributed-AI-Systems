"""
FSDP2 profiling example - demonstrates how to profile FSDP training.

Usage:
    torchrun --nproc_per_node=2 code/fsdp2_profile.py

Output:
    Profiling FSDP2 training with 2 GPUs...
    [profiler table output]
    Trace saved to fsdp_trace_rank0.json
    
Open the trace file in chrome://tracing or https://ui.perfetto.dev/ to visualize.
"""
import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.device_mesh import init_device_mesh
from torch.profiler import profile, record_function, ProfilerActivity


class TransformerBlock(nn.Module):
    """A transformer block for profiling demonstration."""
    def __init__(self, dim=1024, n_heads=8):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_out = nn.Linear(dim, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.norm2 = nn.LayerNorm(dim)
    
    def forward(self, x):
        B, T, _ = x.shape
        residual = x
        x = self.norm1(x)
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.dim // self.n_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.dim // self.n_heads) ** 0.5
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, T, self.dim)
        x = self.attn_out(out) + residual
        residual = x
        x = self.norm2(x)
        x = self.ffn(x) + residual
        return x


class SimpleTransformer(nn.Module):
    """A simple transformer for profiling."""
    def __init__(self, n_layers=6, dim=1024, n_heads=8, vocab_size=10000):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(dim=dim, n_heads=n_heads)
            for _ in range(n_layers)
        ])
        self.head = nn.Linear(dim, vocab_size)
    
    def forward(self, x):
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        return self.head(x)


def train_with_profiling(model, optimizer, device, num_iterations=10):
    """Profile training loop."""
    rank = dist.get_rank()
    criterion = nn.CrossEntropyLoss()
    
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        with record_function("training_loop"):
            for i in range(num_iterations):
                # Generate dummy data
                data = torch.randint(0, 10000, (8, 256), device=device)
                target = torch.randint(0, 10000, (8, 256), device=device)
                
                with record_function("forward"):
                    output = model(data)
                    loss = criterion(output.view(-1, output.size(-1)), target.view(-1))
                
                with record_function("backward"):
                    loss.backward()
                
                with record_function("optimizer"):
                    optimizer.step()
                    optimizer.zero_grad()
    
    if rank == 0:
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))
        trace_file = f"fsdp_trace_rank{rank}.json"
        prof.export_chrome_trace(trace_file)
        print(f"\nTrace saved to {trace_file}")
        print("Open in chrome://tracing or https://ui.perfetto.dev/ to visualize.")


def main():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    mesh = init_device_mesh("cuda", (world_size,))
    
    if rank == 0:
        print(f"Profiling FSDP2 training with {world_size} GPUs...")
    
    # Create model
    model = SimpleTransformer(n_layers=6, dim=1024, n_heads=8).to(device)
    
    # Apply FSDP2 to each block, then to the whole model
    for block in model.blocks:
        fully_shard(block, mesh=mesh)
    fully_shard(model, mesh=mesh)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    
    # Profile training
    train_with_profiling(model, optimizer, device, num_iterations=10)
    
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
