#!/usr/bin/env python3
"""
Export Megatron checkpoint to standard PyTorch format.

This script loads a Megatron distributed checkpoint and saves it as a standard PyTorch .pt file
that can be used by other frameworks like SGLang, LLM, or HuggingFace (with manual conversion).

Usage:
    python convert_megatron_to_pytorch.py \
        --checkpoint-dir checkpoints/gpt_8b/iter_0000010 \
        --output-path model.pt
"""

import argparse
import os
import sys
import torch
import torch.distributed as dist

# Parse arguments BEFORE importing Megatron (which parses all command line args)
parser = argparse.ArgumentParser(description="Export Megatron checkpoint to PyTorch format")
parser.add_argument('--checkpoint-dir', type=str, required=True,
                   help='Path to Megatron checkpoint directory (e.g., checkpoints/gpt_8b/iter_0000010)')
parser.add_argument('--output-path', type=str, required=True,
                   help='Output path for PyTorch checkpoint (e.g., model.pt)')

# Model configuration (should match training config)
parser.add_argument('--num-layers', type=int, default=32)
parser.add_argument('--hidden-size', type=int, default=4096)
parser.add_argument('--num-attention-heads', type=int, default=32)
parser.add_argument('--vocab-size', type=int, default=128256)
parser.add_argument('--max-position-embeddings', type=int, default=2048)
parser.add_argument('--ffn-hidden-size', type=int, default=14336)
parser.add_argument('--num-query-groups', type=int, default=8)
parser.add_argument('--kv-channels', type=int, default=128)

# Parse known args to avoid conflicts with Megatron's argument parser
script_args, unknown = parser.parse_known_args()

# Save original argv and replace with only Megatron-compatible args
original_argv = sys.argv.copy()
sys.argv = [sys.argv[0]] + unknown


def find_megatron_path():
    """Find Megatron-LM path automatically.
    
    Checks in order:
    1. MEGATRON_LM_PATH environment variable
    2. Parent directories (looking for Megatron-LM)
    3. Try importing megatron (if already in path)
    """
    from pathlib import Path
    
    # Check environment variable
    env_path = os.environ.get('MEGATRON_LM_PATH')
    if env_path and os.path.isdir(env_path):
        return env_path
    
    # Try to import megatron (might already be installed/available)
    try:
        import megatron
        # If import succeeds, check if it's from a source directory
        megatron_file = Path(megatron.__file__)
        # If it's from a source install, the parent should be Megatron-LM
        if 'Megatron-LM' in str(megatron_file):
            megatron_lm_path = megatron_file.parent.parent
            if (megatron_lm_path / 'megatron').exists():
                return str(megatron_lm_path)
        # If already importable, we don't need to add to path
        return None
    except ImportError:
        pass
    
    # Search in parent directories
    current = Path(__file__).resolve().parent
    # Check siblings and parents for Megatron-LM directory
    for parent in [current] + list(current.parents):
        # Check sibling directory
        megatron_lm_path = parent.parent / 'Megatron-LM'
        if megatron_lm_path.exists() and (megatron_lm_path / 'megatron').exists():
            return str(megatron_lm_path)
        
        # Check if current directory is inside Megatron-LM
        if 'Megatron-LM' in str(parent):
            megatron_lm_path = parent
            if (megatron_lm_path / 'megatron').exists():
                return str(megatron_lm_path)
        
        # Check if Megatron-LM is a sibling at this level
        megatron_lm_path = parent / 'Megatron-LM'
        if megatron_lm_path.exists() and (megatron_lm_path / 'megatron').exists():
            return str(megatron_lm_path)
    
    return None


# Add Megatron-LM to path if needed
MEGATRON_LM_PATH = find_megatron_path()
if MEGATRON_LM_PATH:
    if MEGATRON_LM_PATH not in sys.path:
        sys.path.insert(0, MEGATRON_LM_PATH)
elif 'megatron' not in sys.modules:
    # Try one more time to import
    try:
        import megatron
    except ImportError:
        print("WARNING: Could not find Megatron-LM path automatically.")
        print("Please set MEGATRON_LM_PATH environment variable or ensure Megatron-LM is installed.")
        print("Example: export MEGATRON_LM_PATH=/path/to/Megatron-LM")

# Set environment for single GPU export
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29500'

from megatron.core import dist_checkpointing
from megatron.training import get_args, initialize_megatron, get_model
from megatron.training.utils import unwrap_model
from megatron.training.checkpointing import load_checkpoint
from model_provider import model_provider
from gpt_builders import gpt_builder
from functools import partial
from megatron.core.enums import ModelType


def main():
    # Use script_args parsed before Megatron import
    args = script_args
    
    print("=" * 60)
    print("Megatron Checkpoint Exporter")
    print("=" * 60)
    print(f"Checkpoint dir: {args.checkpoint_dir}")
    print(f"Output path: {args.output_path}")
    print()
    
    # Initialize distributed (single process)
    if not dist.is_initialized():
        dist.init_process_group(
            backend='nccl' if torch.cuda.is_available() else 'gloo',
            init_method='tcp://127.0.0.1:29500',
            rank=0,
            world_size=1
        )
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
    
    try:
        # Initialize Megatron
        print("Initializing Megatron...")
        initialize_megatron(
            extra_args_provider=None,
            args_defaults={
                'num_layers': args.num_layers,
                'hidden_size': args.hidden_size,
                'num_attention_heads': args.num_attention_heads,
                'vocab_size': args.vocab_size,
                'max_position_embeddings': args.max_position_embeddings,
                'ffn_hidden_size': args.ffn_hidden_size,
                'num_query_groups': args.num_query_groups,
                'kv_channels': args.kv_channels,
                'use_mcore_models': True,
                'bf16': True,
                'load': args.checkpoint_dir,
                'no_load_optim': True,
                'no_load_rng': True,
                'tensor_model_parallel_size': 1,
                'pipeline_model_parallel_size': 1,
                'data_parallel_size': 1,
                'micro_batch_size': 1,
                'global_batch_size': 1,
                'seq_length': 2048,
                'group_query_attention': True,
                'position_embedding_type': 'rope',
                'rotary_base': 1000000,
                'rotary_percent': 1.0,
                'swiglu': True,
                'untie_embeddings_and_output_weights': True,
                'disable_bias_linear': True,
                'apply_layernorm_1p': True,
                'attention_backend': 'fused',
                'tokenizer_type': 'NullTokenizer',  # Use null tokenizer for export
                'gradient_accumulation_fusion': False,  # Disable gradient accumulation fusion
            }
        )
        
        # Force disable gradient accumulation fusion after initialization
        # (in case checkpoint overrides it)
        from megatron.training import get_args as get_megatron_args
        megatron_args = get_megatron_args()
        megatron_args.gradient_accumulation_fusion = False
        
        # Build model
        print("Building model...")
        model = get_model(
            partial(model_provider, gpt_builder),
            ModelType.encoder_or_decoder,
            wrap_with_ddp=False
        )
        
        # Load checkpoint
        print(f"Loading checkpoint from {args.checkpoint_dir}...")
        load_checkpoint(model, None, None, strict=False)
        print("Checkpoint loaded successfully!")
        
        # Get unwrapped model
        unwrapped = unwrap_model(model)
        if isinstance(unwrapped, list):
            unwrapped = unwrapped[0]
        
        # Get state dict
        print("Extracting state dict...")
        state_dict = unwrapped.state_dict()
        
        # Convert to bf16 to save space (original training was in bf16)
        print("Converting to bfloat16 to match training precision...")
        for key in state_dict:
            if isinstance(state_dict[key], torch.Tensor) and state_dict[key].dtype == torch.float32:
                state_dict[key] = state_dict[key].to(torch.bfloat16)
        
        # Save as PyTorch checkpoint
        print(f"Saving to {args.output_path}...")
        os.makedirs(os.path.dirname(args.output_path) if os.path.dirname(args.output_path) else '.', exist_ok=True)
        
        checkpoint = {
            'model_state_dict': state_dict,
            'model_config': {
                'num_layers': args.num_layers,
                'hidden_size': args.hidden_size,
                'num_attention_heads': args.num_attention_heads,
                'vocab_size': args.vocab_size,
                'max_position_embeddings': args.max_position_embeddings,
                'ffn_hidden_size': args.ffn_hidden_size,
                'num_query_groups': args.num_query_groups,
                'kv_channels': args.kv_channels,
            }
        }
        
        torch.save(checkpoint, args.output_path)
        
        file_size_gb = os.path.getsize(args.output_path) / 1e9
        print(f"✓ Saved PyTorch checkpoint to {args.output_path}")
        print(f"  File size: {file_size_gb:.2f} GB")
        print()
        print("You can now load this checkpoint with:")
        print(f"  checkpoint = torch.load('{args.output_path}')")
        print(f"  model.load_state_dict(checkpoint['model_state_dict'])")
        
    finally:
        # Cleanup
        if dist.is_initialized():
            dist.destroy_process_group()
    
    print("Export completed!")


if __name__ == '__main__':
    main()
