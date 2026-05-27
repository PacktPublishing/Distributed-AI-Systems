"""
ZeRO-Offload and ZeRO-Infinity Example.

Demonstrates CPU and NVMe offloading for training large models
on limited GPU memory.

Usage:
    # CPU offloading (ZeRO-Offload)
    deepspeed --num_gpus=1 zero_offload_example.py --offload_device cpu

    # NVMe offloading (ZeRO-Infinity) - requires NVMe path
    deepspeed --num_gpus=1 zero_offload_example.py --offload_device nvme --nvme_path /tmp/nvme_offload

Requirements:
    pip install deepspeed transformers
    
Note: NVMe offloading requires:
    - Fast NVMe SSD (PCIe 4.0+ recommended)
    - libaio library (apt install libaio-dev)
"""

import argparse
import os
import torch
from transformers import GPT2LMHeadModel, GPT2Config
from torch.utils.data import Dataset
import deepspeed


class RandomTextDataset(Dataset):
    """Simple random dataset for demonstration."""
    def __init__(self, vocab_size, seq_length, num_samples=500):
        self.vocab_size = vocab_size
        self.seq_length = seq_length
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        input_ids = torch.randint(0, self.vocab_size, (self.seq_length,))
        return {"input_ids": input_ids, "labels": input_ids}


def get_offload_config(args):
    """Build DeepSpeed config with offloading."""
    
    config = {
        "train_batch_size": args.batch_size,
        "gradient_accumulation_steps": 1,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 1e-4,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.01
            }
        },
        "fp16": {
            "enabled": True,
            "loss_scale": 0,
            "initial_scale_power": 16
        },
        "zero_optimization": {
            "stage": 3,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "stage3_prefetch_bucket_size": 5e7,
            "stage3_param_persistence_threshold": 1e5,
            "stage3_max_live_parameters": 1e8,
            "stage3_max_reuse_distance": 1e8,
        },
        "gradient_clipping": 1.0,
        "steps_per_print": 10,
    }
    
    if args.offload_device == "cpu":
        # ZeRO-Offload: CPU offloading
        config["zero_optimization"]["offload_optimizer"] = {
            "device": "cpu",
            "pin_memory": True,
            "buffer_count": 4,
            "fast_init": True
        }
        config["zero_optimization"]["offload_param"] = {
            "device": "cpu",
            "pin_memory": True,
            "buffer_count": 5,
            "buffer_size": 1e8,
        }
        
    elif args.offload_device == "nvme":
        # ZeRO-Infinity: NVMe offloading
        if not args.nvme_path:
            raise ValueError("--nvme_path required for NVMe offloading")
        
        os.makedirs(args.nvme_path, exist_ok=True)
        
        config["zero_optimization"]["offload_optimizer"] = {
            "device": "nvme",
            "nvme_path": args.nvme_path,
            "pin_memory": True,
            "buffer_count": 4,
            "fast_init": True
        }
        config["zero_optimization"]["offload_param"] = {
            "device": "nvme",
            "nvme_path": args.nvme_path,
            "pin_memory": True,
            "buffer_count": 5,
            "buffer_size": 1e8,
            "max_in_cpu": 1e9,
        }
        
        # Async I/O configuration for NVMe
        config["aio"] = {
            "block_size": 1048576,
            "queue_depth": 8,
            "thread_count": 1,
            "single_submit": False,
            "overlap_events": True
        }
    
    return config


def main():
    parser = argparse.ArgumentParser(description="ZeRO Offload Example")
    parser.add_argument("--offload_device", type=str, default="cpu",
                        choices=["cpu", "nvme"],
                        help="Device to offload to")
    parser.add_argument("--nvme_path", type=str, default=None,
                        help="Path for NVMe offloading")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--seq_length", type=int, default=512)
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--num_layers", type=int, default=24)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--local_rank", type=int, default=-1)
    
    args = parser.parse_args()
    
    # Create a larger model to demonstrate offloading benefits
    config = GPT2Config(
        vocab_size=50257,
        n_positions=args.seq_length,
        n_embd=args.hidden_size,
        n_layer=args.num_layers,
        n_head=args.num_heads,
    )
    
    # Calculate model size
    model = GPT2LMHeadModel(config)
    param_count = sum(p.numel() for p in model.parameters())
    param_size_gb = param_count * 2 / 1e9  # FP16
    
    print(f"Model: {param_count/1e6:.1f}M parameters ({param_size_gb:.2f} GB in FP16)")
    print(f"Offload device: {args.offload_device}")
    
    # Create dataset
    dataset = RandomTextDataset(
        vocab_size=config.vocab_size,
        seq_length=args.seq_length,
        num_samples=args.num_steps * args.batch_size * 2
    )
    
    # Get DeepSpeed config
    ds_config = get_offload_config(args)
    
    # Initialize DeepSpeed
    model_engine, optimizer, train_dataloader, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        training_data=dataset,
        config=ds_config,
    )
    
    # Training loop
    model_engine.train()
    
    for step, batch in enumerate(train_dataloader):
        if step >= args.num_steps:
            break
        
        input_ids = batch["input_ids"].to(model_engine.device)
        labels = batch["labels"].to(model_engine.device)
        
        outputs = model_engine(input_ids=input_ids, labels=labels)
        loss = outputs.loss
        
        model_engine.backward(loss)
        model_engine.step()
        
        if step % 10 == 0 and model_engine.local_rank == 0:
            # Get GPU memory usage
            gpu_mem = torch.cuda.max_memory_allocated() / 1e9
            print(f"Step {step}, Loss: {loss.item():.4f}, GPU Memory: {gpu_mem:.2f} GB")
    
    if model_engine.local_rank == 0:
        print(f"\nTraining complete with {args.offload_device.upper()} offloading!")
        print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
