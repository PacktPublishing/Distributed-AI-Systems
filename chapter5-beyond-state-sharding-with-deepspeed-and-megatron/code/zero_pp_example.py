"""
ZeRO++ Example: Communication-Optimized Training.

Demonstrates ZeRO++ features:
- qwZ (Quantized Weights): Reduces all-gather communication
- hpZ (Hierarchical Partitioning): Intra-node full replication
- qgZ (Quantized Gradients): Reduces gradient communication

Usage:
    # Basic ZeRO++ with quantized weights
    deepspeed --num_gpus=4 zero_pp_example.py --enable_qwz

    # ZeRO++ with hierarchical partitioning (multi-node simulation)
    deepspeed --num_gpus=4 zero_pp_example.py --enable_hpz

    # Full ZeRO++ (all optimizations)
    deepspeed --num_gpus=4 zero_pp_example.py --enable_qwz --enable_hpz --enable_qgz

Requirements:
    pip install deepspeed>=0.10.0 transformers
"""

import argparse
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


def get_zero_pp_config(args):
    """Build DeepSpeed config with ZeRO++ optimizations."""
    
    config = {
        "train_batch_size": args.batch_size * args.world_size,
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
        # ZeRO++ qwZ/qgZ dequantize to FP16, so we must use fp16 (not bf16)
        "fp16": {
            "enabled": True
        },
        "zero_optimization": {
            "stage": 3,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": 5e7,
            "stage3_prefetch_bucket_size": 5e7,
            "stage3_param_persistence_threshold": 1e5,
        },
        "gradient_clipping": 1.0,
        "steps_per_print": 10,
    }
    
    # qwZ: Quantized Weight Communication
    # Reduces all-gather communication by using INT8 quantization
    if args.enable_qwz:
        config["zero_optimization"]["zero_quantized_weights"] = True
        config["zero_optimization"]["zero_hpz_partition_size"] = args.world_size
        print("Enabled qwZ (Quantized Weights)")
    
    # hpZ: Hierarchical Partitioning
    # Keeps full copy within node, partitions across nodes
    # Reduces inter-node communication at cost of intra-node memory
    if args.enable_hpz:
        # Partition size = GPUs per node (simulated as world_size / 2)
        partition_size = max(1, args.world_size // 2)
        config["zero_optimization"]["zero_hpz_partition_size"] = partition_size
        print(f"Enabled hpZ (Hierarchical Partitioning, partition_size={partition_size})")
    
    # qgZ: Quantized Gradient Communication
    # Reduces reduce-scatter communication using quantization
    if args.enable_qgz:
        config["zero_optimization"]["zero_quantized_gradients"] = True
        print("Enabled qgZ (Quantized Gradients)")
    
    return config


def main():
    parser = argparse.ArgumentParser(description="ZeRO++ Example")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--seq_length", type=int, default=512)
    parser.add_argument("--hidden_size", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--enable_qwz", action="store_true",
                        help="Enable quantized weight communication")
    parser.add_argument("--enable_hpz", action="store_true",
                        help="Enable hierarchical partitioning")
    parser.add_argument("--enable_qgz", action="store_true",
                        help="Enable quantized gradient communication")
    parser.add_argument("--local_rank", type=int, default=-1)
    
    args = parser.parse_args()
    
    # Get world size from DeepSpeed
    deepspeed.init_distributed()
    args.world_size = torch.distributed.get_world_size()
    
    # Create model
    config = GPT2Config(
        vocab_size=50257,
        n_positions=args.seq_length,
        n_embd=args.hidden_size,
        n_layer=args.num_layers,
        n_head=args.num_heads,
    )
    model = GPT2LMHeadModel(config)
    
    param_count = sum(p.numel() for p in model.parameters())
    
    if args.local_rank == 0:
        print(f"\nModel: {param_count/1e6:.1f}M parameters")
        print(f"World size: {args.world_size}")
        print(f"ZeRO++ features:")
        print(f"  qwZ (Quantized Weights): {args.enable_qwz}")
        print(f"  hpZ (Hierarchical Partitioning): {args.enable_hpz}")
        print(f"  qgZ (Quantized Gradients): {args.enable_qgz}")
        print()
    
    # Create dataset
    dataset = RandomTextDataset(
        vocab_size=config.vocab_size,
        seq_length=args.seq_length,
        num_samples=args.num_steps * args.batch_size * args.world_size
    )
    
    # Get DeepSpeed config
    ds_config = get_zero_pp_config(args)
    
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
            print(f"Step {step}, Loss: {loss.item():.4f}")
    
    if model_engine.local_rank == 0:
        print(f"\nZeRO++ training complete!")
        print(f"Peak GPU memory: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
