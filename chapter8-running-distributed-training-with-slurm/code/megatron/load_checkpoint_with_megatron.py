#!/usr/bin/env python3
"""
Load and use the exported Megatron checkpoint WITH Megatron framework.

NOTE: This script requires Megatron-LM to be installed/available.
For a standalone checkpoint loader (NO Megatron dependency), see load_checkpoint_standalone.py

This script demonstrates how to load the exported PyTorch checkpoint and use it 
for inference WITH the full Megatron framework.

Usage:
    python load_checkpoint_with_megatron.py --checkpoint-path exported_model.pt
"""

import argparse
import os
import sys
from pathlib import Path
import torch
import torch.distributed as dist


def find_megatron_path():
    """Find Megatron-LM path automatically.
    
    Checks in order:
    1. MEGATRON_LM_PATH environment variable
    2. Parent directories (looking for Megatron-LM)
    3. Try importing megatron (if already in path)
    """
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

# Set environment for single GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['MASTER_ADDR'] = '127.0.0.1'
os.environ['MASTER_PORT'] = '29500'

from megatron.training import get_args, initialize_megatron, get_model
from megatron.training.utils import unwrap_model
from model_provider import model_provider
from gpt_builders import gpt_builder
from functools import partial
from megatron.core.enums import ModelType


def main():
    parser = argparse.ArgumentParser(description="Load exported Megatron checkpoint")
    parser.add_argument('--checkpoint-path', type=str, required=True,
                       help='Path to exported PyTorch checkpoint (e.g., exported_model.pt)')
    
    # Model configuration (should match training/export config)
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
    
    print("=" * 60)
    print("Loading Exported Megatron Checkpoint")
    print("=" * 60)
    print(f"Checkpoint path: {script_args.checkpoint_path}")
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
        # Load checkpoint to get config
        print("Loading checkpoint to read config...")
        checkpoint = torch.load(script_args.checkpoint_path, map_location='cpu')
        model_config = checkpoint.get('model_config', {})
        
        # Use config from checkpoint if available, otherwise use command line args
        num_layers = model_config.get('num_layers', script_args.num_layers)
        hidden_size = model_config.get('hidden_size', script_args.hidden_size)
        num_attention_heads = model_config.get('num_attention_heads', script_args.num_attention_heads)
        vocab_size = model_config.get('vocab_size', script_args.vocab_size)
        max_position_embeddings = model_config.get('max_position_embeddings', script_args.max_position_embeddings)
        ffn_hidden_size = model_config.get('ffn_hidden_size', script_args.ffn_hidden_size)
        num_query_groups = model_config.get('num_query_groups', script_args.num_query_groups)
        kv_channels = model_config.get('kv_channels', script_args.kv_channels)
        
        print(f"Model config from checkpoint:")
        print(f"  Layers: {num_layers}")
        print(f"  Hidden size: {hidden_size}")
        print(f"  Attention heads: {num_attention_heads}")
        print(f"  Vocab size: {vocab_size}")
        print()
        
        # Initialize Megatron
        print("Initializing Megatron...")
        initialize_megatron(
            extra_args_provider=None,
            args_defaults={
                'num_layers': num_layers,
                'hidden_size': hidden_size,
                'num_attention_heads': num_attention_heads,
                'vocab_size': vocab_size,
                'max_position_embeddings': max_position_embeddings,
                'ffn_hidden_size': ffn_hidden_size,
                'num_query_groups': num_query_groups,
                'kv_channels': kv_channels,
                'use_mcore_models': True,
                'bf16': True,
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
                'tokenizer_type': 'NullTokenizer',
                'gradient_accumulation_fusion': False,
            }
        )
        
        # Force disable gradient accumulation fusion
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
        
        # Load checkpoint weights
        print(f"Loading weights from {script_args.checkpoint_path}...")
        state_dict = checkpoint['model_state_dict']
        
        # Get unwrapped model
        unwrapped = unwrap_model(model)
        if isinstance(unwrapped, list):
            unwrapped = unwrapped[0]
        
        # Load state dict
        missing_keys, unexpected_keys = unwrapped.load_state_dict(state_dict, strict=False)
        
        if missing_keys:
            print(f"Warning: Missing keys: {len(missing_keys)} keys")
            if len(missing_keys) <= 10:
                for key in missing_keys:
                    print(f"  - {key}")
        
        if unexpected_keys:
            print(f"Warning: Unexpected keys: {len(unexpected_keys)} keys")
            if len(unexpected_keys) <= 10:
                for key in unexpected_keys:
                    print(f"  - {key}")
        
        print("✓ Model loaded successfully!")
        print()
        
        # Move model to appropriate device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if isinstance(model, list):
            model = model[0]
        model = model.to(device)
        model.eval()
        
        print(f"Model is ready on {device}")
        print(f"Model dtype: {next(unwrapped.parameters()).dtype}")
        print()
        
        # Example: Get model info
        total_params = sum(p.numel() for p in unwrapped.parameters())
        trainable_params = sum(p.numel() for p in unwrapped.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params/1e9:.2f}B")
        print(f"Trainable parameters: {trainable_params/1e9:.2f}B")
        print()
        
        print("You can now use the model for inference!")
        print("Example:")
        print("  with torch.no_grad():")
        print("      output = model(input_ids)")
        
    finally:
        # Cleanup
        if dist.is_initialized():
            dist.destroy_process_group()
    
    print("Done!")


if __name__ == '__main__':
    main()
