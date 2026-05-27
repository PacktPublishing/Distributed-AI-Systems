#!/usr/bin/env python3
"""
Export Megatron-LM checkpoint to standard PyTorch format or HuggingFace format.

Usage:
    python convert_megatron_checkpoint.py \
        --checkpoint-dir /path/to/megatron/checkpoint \
        --output-dir /path/to/output \
        --format pytorch  # or 'huggingface'
"""

import argparse
import os
import sys
import torch
import torch.distributed as dist
from pathlib import Path

# Add Megatron-LM to path
MEGATRON_LM_PATH = "/home/wukong/workspace/coderepo/Megatron-LM"
if MEGATRON_LM_PATH not in sys.path:
    sys.path.insert(0, MEGATRON_LM_PATH)

from megatron.core import dist_checkpointing
from megatron.training import get_args, initialize_megatron
from megatron.training.utils import unwrap_model
from megatron.training.checkpointing import load_checkpoint
from model_provider import model_provider
from gpt_builders import gpt_builder
from functools import partial
from megatron.core.enums import ModelType


def export_to_pytorch(checkpoint_dir, output_path, model):
    """Export model to standard PyTorch state dict format."""
    print(f"Exporting to PyTorch format: {output_path}")
    
    # Get unwrapped model (remove DDP wrapper if present)
    if hasattr(model, 'module'):
        unwrapped = model.module
    else:
        unwrapped = model
    if isinstance(unwrapped, list):
        unwrapped = unwrapped[0]
    
    # Get state dict
    state_dict = unwrapped.state_dict()
    
    # Save as PyTorch checkpoint
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save({
        'model_state_dict': state_dict,
        'model_config': {
            'num_layers': get_args().num_layers,
            'hidden_size': get_args().hidden_size,
            'num_attention_heads': get_args().num_attention_heads,
            'vocab_size': get_args().vocab_size,
            'max_position_embeddings': get_args().max_position_embeddings,
        }
    }, output_path)
    
    print(f"Saved PyTorch checkpoint to {output_path}")
    print(f"Checkpoint size: {os.path.getsize(output_path) / 1e9:.2f} GB")


def export_to_huggingface(checkpoint_dir, output_dir, model):
    """Export model to HuggingFace format (simplified version)."""
    print(f"Exporting to HuggingFace format: {output_dir}")
    print("WARNING: Full HuggingFace export requires model-specific conversion.")
    print("This is a simplified export that saves the state dict in a compatible format.")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Get unwrapped model
    unwrapped = unwrap_model(model)[0]
    state_dict = unwrapped.state_dict()
    
    # Save state dict (can be loaded by transformers with custom loading code)
    output_path = os.path.join(output_dir, "pytorch_model.bin")
    torch.save(state_dict, output_path)
    
    print(f"Saved state dict to {output_path}")
    print(f"Note: You may need to manually convert layer names to HuggingFace format")
    print(f"      or use a conversion tool like Megatron-Bridge")


def main():
    parser = argparse.ArgumentParser(description="Export Megatron checkpoint")
    parser.add_argument('--checkpoint-dir', type=str, required=True,
                       help='Path to Megatron checkpoint directory')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Output directory for exported checkpoint')
    parser.add_argument('--format', type=str, default='pytorch',
                       choices=['pytorch', 'huggingface'],
                       help='Export format')
    parser.add_argument('--no-load-optim', action='store_true',
                       help='Do not load optimizer state')
    parser.add_argument('--no-load-rng', action='store_true',
                       help='Do not load RNG state')
    
    # Add model arguments (minimal set needed)
    parser.add_argument('--num-layers', type=int, default=32)
    parser.add_argument('--hidden-size', type=int, default=4096)
    parser.add_argument('--num-attention-heads', type=int, default=32)
    parser.add_argument('--vocab-size', type=int, default=128256)
    parser.add_argument('--max-position-embeddings', type=int, default=2048)
    parser.add_argument('--use-mcore-models', action='store_true')
    parser.add_argument('--bf16', action='store_true')
    
    args = parser.parse_args()
    
    # Initialize distributed (single process for export)
    if not dist.is_initialized():
        dist.init_process_group(
            backend='nccl' if torch.cuda.is_available() else 'gloo',
            init_method='tcp://localhost:29500',
            rank=0,
            world_size=1
        )
    
    # Initialize Megatron with minimal args
    initialize_megatron(
        extra_args_provider=None,
        args_defaults={
            'num_layers': args.num_layers,
            'hidden_size': args.hidden_size,
            'num_attention_heads': args.num_attention_heads,
            'vocab_size': args.vocab_size,
            'max_position_embeddings': args.max_position_embeddings,
            'use_mcore_models': args.use_mcore_models,
            'bf16': args.bf16,
            'load': args.checkpoint_dir,
            'no_load_optim': args.no_load_optim,
            'no_load_rng': args.no_load_rng,
        }
    )
    
    # Build model
    from megatron.training import get_model
    model = get_model(
        partial(model_provider, gpt_builder),
        ModelType.encoder_or_decoder,
        wrap_with_ddp=False
    )
    
    # Load checkpoint
    from megatron.training.checkpointing import load_checkpoint
    
    print(f"Loading checkpoint from {args.checkpoint_dir}...")
    load_checkpoint(model, None, None, strict=False)
    print("Checkpoint loaded successfully")
    
    # Export
    if args.format == 'pytorch':
        output_path = os.path.join(args.output_dir, "model.pt")
        export_to_pytorch(args.checkpoint_dir, output_path, model)
    elif args.format == 'huggingface':
        export_to_huggingface(args.checkpoint_dir, args.output_dir, model)
    
    # Cleanup
    dist.destroy_process_group()
    print("Export completed!")


if __name__ == '__main__':
    main()
