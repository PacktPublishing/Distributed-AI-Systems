"""
Minimal ZeRO Example - Run on a single GPU.

This script demonstrates ZeRO stages with a small model that can run
on a single GPU. It shows memory usage differences between stages.

Usage:
    # ZeRO Stage 1 (optimizer state sharding)
    deepspeed --num_gpus=1 zero_minimal.py --zero_stage 1

    # ZeRO Stage 2 (+ gradient sharding)
    deepspeed --num_gpus=1 zero_minimal.py --zero_stage 2

    # ZeRO Stage 3 (+ parameter sharding)
    deepspeed --num_gpus=1 zero_minimal.py --zero_stage 3

Requirements:
    pip install deepspeed torch
"""

import argparse
import torch
import torch.nn as nn
import deepspeed


class SimpleMLP(nn.Module):
    """A simple MLP for demonstration."""
    def __init__(self, hidden_size=1024, num_layers=4):
        super().__init__()
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x):
        return self.head(self.layers(x))


def get_ds_config(zero_stage, batch_size):
    """Build DeepSpeed config for specified ZeRO stage."""
    config = {
        "train_batch_size": batch_size,
        "optimizer": {
            "type": "Adam",
            "params": {"lr": 1e-4}
        },
        "fp16": {"enabled": True},
        "zero_optimization": {
            "stage": zero_stage,
        }
    }
    
    if zero_stage >= 2:
        config["zero_optimization"]["allgather_bucket_size"] = 2e8
        config["zero_optimization"]["reduce_bucket_size"] = 2e8
    
    if zero_stage == 3:
        config["zero_optimization"]["stage3_prefetch_bucket_size"] = 2e8
        config["zero_optimization"]["stage3_param_persistence_threshold"] = 1e6
    
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zero_stage", type=int, default=1, choices=[1, 2, 3])
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_steps", type=int, default=10)
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    # Create model
    model = SimpleMLP(args.hidden_size, args.num_layers)
    param_count = sum(p.numel() for p in model.parameters())
    
    # Initialize DeepSpeed
    ds_config = get_ds_config(args.zero_stage, args.batch_size)
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config
    )
    
    device = model_engine.device
    
    if model_engine.local_rank == 0:
        print(f"\n{'='*20}")
        print(f"ZeRO Stage {args.zero_stage} Demo")
        print(f"{'='*20}")
        print(f"Model: {param_count:,} parameters")
        print(f"Hidden size: {args.hidden_size}")
        print(f"Num layers: {args.num_layers}")
        print(f"Batch size: {args.batch_size}")
        print(f"{'='*20}\n")

    # Training loop
    model_engine.train()
    for step in range(args.num_steps):
        # Random input (use half precision to match model)
        x = torch.randn(args.batch_size, args.hidden_size, device=device, dtype=torch.half)
        target = torch.randn(args.batch_size, args.hidden_size, device=device, dtype=torch.half)
        
        # Forward
        output = model_engine(x)
        loss = nn.functional.mse_loss(output, target)
        
        # Backward
        model_engine.backward(loss)
        model_engine.step()
        
        if model_engine.local_rank == 0:
            mem_gb = torch.cuda.max_memory_allocated() / 1e9
            print(f"Step {step+1}/{args.num_steps}, Loss: {loss.item():.4f}, "
                  f"Peak Memory: {mem_gb:.2f} GB")

    if model_engine.local_rank == 0:
        print(f"\nZeRO Stage {args.zero_stage} training complete!")
        print(f"Final peak memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
