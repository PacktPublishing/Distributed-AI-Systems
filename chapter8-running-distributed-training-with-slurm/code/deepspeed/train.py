#!/usr/bin/env python3
"""
DeepSpeed ZeRO-3 with CPU Offload Training Script for SLURM.

This script demonstrates DeepSpeed ZeRO-3 with CPU offload, which enables
training models larger than GPU memory by offloading parameters and optimizer
states to CPU memory.

Key features:
- No manual distributed setup needed (DeepSpeed handles it)
- Works with HuggingFace models
- ZeRO-3 shards parameters, gradients, and optimizer states
- CPU offload enables training models larger than GPU memory
"""

import os
import socket
import argparse
import torch
import torch.distributed as dist
import deepspeed
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.data import Dataset, DataLoader


class SimpleTextDataset(Dataset):
    """Simple text dataset for demonstration."""
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0)
        }


def main():
    parser = argparse.ArgumentParser(description='DeepSpeed ZeRO-3 Training')
    parser.add_argument('--model-name', type=str, default='Qwen/Qwen2.5-1.5B-Instruct',
                       help='HuggingFace model name')
    parser.add_argument('--deepspeed', action='store_true',
                       help='Enable DeepSpeed')
    parser.add_argument('--deepspeed-config', type=str, default='ds_zero3_offload.json',
                       help='Path to DeepSpeed config file')
    parser.add_argument('--epochs', type=int, default=1,
                       help='Number of epochs')
    parser.add_argument('--steps', type=int, default=10,
                       help='Number of training steps')
    parser.add_argument('--local_rank', type=int, default=-1,
                       help='Local rank (set by DeepSpeed)')
    args = parser.parse_args()
    
    # Initialize distributed training using SLURM environment variables
    # This must be done BEFORE calling deepspeed.initialize() when using srun
    # Fix IPv6 resolution issues
    os.environ['NCCL_SOCKET_IFNAME'] = '^docker,lo'
    os.environ['GLOO_SOCKET_IFNAME'] = '^docker,lo'
    
    # Check if we're in a SLURM environment
    if 'SLURM_PROCID' in os.environ:
        # SLURM mode - use SLURM environment variables
        rank = int(os.environ.get('SLURM_PROCID', 0))
        world_size = int(os.environ.get('SLURM_NTASKS', 1))
        local_rank = int(os.environ.get('SLURM_LOCALID', 0))
        
        # DeepSpeed requires LOCAL_RANK environment variable
        # Set it from SLURM_LOCALID
        os.environ['LOCAL_RANK'] = str(local_rank)
        
        # Get master address - use localhost for single-node multi-GPU setup
        master_addr = os.environ.get('MASTER_ADDR', '127.0.0.1')
        if master_addr == 'localhost':
            master_addr = '127.0.0.1'
        master_port = os.environ.get('MASTER_PORT', '29500')
        
        # For single-node multi-GPU setups, handle GPU mapping
        # Override CUDA_VISIBLE_DEVICES based on node name (node6 -> GPU 6, node7 -> GPU 7)
        node_name = os.environ.get('SLURMD_NODENAME', '')
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        
        # Always override CUDA_VISIBLE_DEVICES based on node name for virtual nodes
        if node_name.startswith('node'):
            try:
                gpu_num = int(node_name.replace('node', ''))
                os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_num)
                if rank == 0:
                    if cuda_visible:
                        print(f"Overriding CUDA_VISIBLE_DEVICES from '{cuda_visible}' to '{gpu_num}' for node {node_name}")
                    else:
                        print(f"Setting CUDA_VISIBLE_DEVICES={gpu_num} for node {node_name}")
            except ValueError:
                if not cuda_visible:
                    os.environ['CUDA_VISIBLE_DEVICES'] = str(local_rank)
        
        # Initialize process group if not already initialized
        if not dist.is_initialized():
            os.environ['MASTER_ADDR'] = master_addr
            os.environ['MASTER_PORT'] = str(master_port)
            
            init_method = f'tcp://{master_addr}:{master_port}'
            device_id = 0  # After CUDA_VISIBLE_DEVICES remapping, always use device 0
            torch.cuda.set_device(device_id)
            dist.init_process_group(
                backend='nccl',
                init_method=init_method,
                rank=rank,
                world_size=world_size,
                device_id=device_id
            )
    else:
        # Non-SLURM mode
        rank = int(os.environ.get('RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        # Ensure LOCAL_RANK is set for DeepSpeed
        if 'LOCAL_RANK' not in os.environ:
            os.environ['LOCAL_RANK'] = str(local_rank)
        
        # Initialize process group if not already initialized
        if not dist.is_initialized() and world_size > 1:
            master_addr = os.environ.get('MASTER_ADDR', '127.0.0.1')
            master_port = os.environ.get('MASTER_PORT', '29500')
            init_method = f'tcp://{master_addr}:{master_port}'
            torch.cuda.set_device(local_rank)
            dist.init_process_group(
                backend='nccl',
                init_method=init_method,
                rank=rank,
                world_size=world_size,
                device_id=local_rank
            )
    
    if rank == 0:
        print(f"Distributed setup: rank={rank}, world_size={world_size}")
        print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    
    # Load tokenizer and model (only on rank 0 to avoid duplicate downloads)
    if rank == 0:
        print(f"Loading model: {args.model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model - DeepSpeed will handle device placement
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=torch.float16,
        device_map=None  # DeepSpeed will handle device placement
    )
    
    # Initialize DeepSpeed
    if args.deepspeed:
        if rank == 0:
            print(f"Initializing DeepSpeed with config: {args.deepspeed_config}")
        
        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=args.deepspeed_config
        )
        
        if model_engine.global_rank == 0:
            print("=" * 60)
            print("DeepSpeed ZeRO-3 Initialization Complete")
            print("=" * 60)
            print(f"  ZeRO Stage: {model_engine.zero_optimization_stage()}")
            # Check parameter offload (ZeRO-3 only)
            param_offload = model_engine.zero_offload_param()
            param_offload_enabled = param_offload is not None and str(param_offload.device) != "none"
            print(f"  Parameter offloading: {param_offload_enabled}")
            if param_offload_enabled:
                print(f"    Parameter offload device: {param_offload.device}")
            # Check optimizer offload
            optimizer_offload_enabled = model_engine.zero_use_cpu_optimizer()
            print(f"  Optimizer offloading: {optimizer_offload_enabled}")
            if optimizer_offload_enabled:
                opt_offload = model_engine.zero_offload_optimizer()
                if opt_offload is not None:
                    print(f"    Optimizer offload device: {opt_offload.device}")
            print(f"  Global rank: {model_engine.global_rank}")
            print(f"  World size: {model_engine.world_size}")
            print(f"  Device: {model_engine.device}")
            print("=" * 60)
    else:
        # Non-DeepSpeed mode (for testing)
        model_engine = model
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        if rank == 0:
            print("WARNING: Running without DeepSpeed (for testing only)")
        # For non-DeepSpeed mode, set device manually
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_engine = model_engine.to(device)
    
    model_engine.train()
    
    # Create simple training data
    training_texts = [
        "DeepSpeed ZeRO-3 offload example for distributed training.",
        "This demonstrates CPU offloading for large language models.",
        "ZeRO-3 shards parameters, gradients, and optimizer states.",
        "CPU offload enables training models larger than GPU memory.",
        "DeepSpeed automatically handles distributed training setup.",
    ] * 10  # Repeat to have enough data
    
    dataset = SimpleTextDataset(training_texts, tokenizer)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    
    # Training loop
    # Use hasattr to check if global_rank exists (DeepSpeed engine) or use rank (non-DeepSpeed)
    is_rank_zero = (hasattr(model_engine, 'global_rank') and model_engine.global_rank == 0) or (not hasattr(model_engine, 'global_rank') and rank == 0)
    if is_rank_zero:
        print(f"\nStarting training for {args.steps} steps...\n")
    
    step = 0
    for epoch in range(args.epochs):
        for batch in dataloader:
            if step >= args.steps:
                break
            
            # Move inputs to device
            if hasattr(model_engine, 'device'):
                device = model_engine.device
            else:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            
            # Forward pass
            outputs = model_engine(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=input_ids  # Use input_ids as labels for language modeling
            )
            loss = outputs.loss
            
            # Backward pass and optimizer step
            if hasattr(model_engine, 'backward') and hasattr(model_engine, 'step'):
                # DeepSpeed mode
                model_engine.backward(loss)
                model_engine.step()
            else:
                # Non-DeepSpeed mode
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            
            if is_rank_zero:
                print(f"[Step {step}] Loss: {loss.item():.4f}")
            
            step += 1
            
            if step >= args.steps:
                break
    
    if is_rank_zero:
        print("\nTraining completed!")
        print(f"Total steps: {step}")
    
    # Cleanup distributed training
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
