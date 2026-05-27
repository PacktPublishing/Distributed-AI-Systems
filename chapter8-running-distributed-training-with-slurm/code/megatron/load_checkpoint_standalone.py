#!/usr/bin/env python3
"""
Simple example: Load exported checkpoint in Python (NO MEGATRON DEPENDENCY).

This is a minimal example showing how to load the exported checkpoint.
It only requires PyTorch - no Megatron installation needed.

Usage:
    python load_checkpoint_standalone.py [--checkpoint-path exported_model.pt]
"""

import argparse
import torch


def main():
    parser = argparse.ArgumentParser(description="Load exported checkpoint (no Megatron dependency)")
    parser.add_argument('--checkpoint-path', type=str, default='exported_model.pt',
                       help='Path to exported PyTorch checkpoint (default: exported_model.pt)')
    
    args = parser.parse_args()
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint_path}...")
    try:
        checkpoint = torch.load(args.checkpoint_path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return
    
    # Check contents
    print("\n" + "="*60)
    print("Checkpoint Contents")
    print("="*60)
    print(f"Keys: {list(checkpoint.keys())}")
    
    # Model configuration
    if 'model_config' in checkpoint:
        config = checkpoint['model_config']
        print(f"\nModel Configuration:")
        print("-" * 60)
        for key, value in sorted(config.items()):
            print(f"  {key:30s}: {value}")
    
    # State dict info
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
        print(f"\nState Dict Info:")
        print("-" * 60)
        print(f"  Number of keys: {len(state_dict)}")
        
        # Show first and last few keys
        keys = list(state_dict.keys())
        print(f"\n  First 5 keys:")
        for key in keys[:5]:
            tensor = state_dict[key]
            if isinstance(tensor, torch.Tensor):
                print(f"    {key}")
                print(f"      shape: {tensor.shape}, dtype: {tensor.dtype}, size: {tensor.numel() * tensor.element_size() / 1e6:.2f} MB")
            else:
                print(f"    {key}: {type(tensor)}")
        
        if len(keys) > 10:
            print(f"\n  ... ({len(keys) - 10} more keys) ...")
            print(f"\n  Last 5 keys:")
            for key in keys[-5:]:
                tensor = state_dict[key]
                if isinstance(tensor, torch.Tensor):
                    print(f"    {key}")
                    print(f"      shape: {tensor.shape}, dtype: {tensor.dtype}, size: {tensor.numel() * tensor.element_size() / 1e6:.2f} MB")
                else:
                    print(f"    {key}: {type(tensor)}")
        
        # Calculate total parameters and size
        total_params = 0
        total_size_bytes = 0
        dtype_counts = {}
        
        for key, tensor in state_dict.items():
            if isinstance(tensor, torch.Tensor):
                params = tensor.numel()
                size = params * tensor.element_size()
                total_params += params
                total_size_bytes += size
                dtype = str(tensor.dtype)
                dtype_counts[dtype] = dtype_counts.get(dtype, 0) + params
        
        print(f"\n  Parameter Statistics:")
        print(f"    Total parameters: {total_params/1e9:.2f}B")
        print(f"    Total size: {total_size_bytes/1e9:.2f} GB")
        print(f"\n  Dtype distribution:")
        for dtype, count in sorted(dtype_counts.items()):
            size_gb = count * 2 / 1e9 if 'bfloat16' in dtype or 'bf16' in dtype else count * 4 / 1e9
            print(f"    {dtype:20s}: {count/1e9:8.2f}B params (~{size_gb:.2f} GB)")
    
    print("\n" + "="*60)
    print("✓ Checkpoint loaded successfully!")
    print("="*60)
    print("\nThis checkpoint is independent and can be used with:")
    print("  - PyTorch models (with matching architecture)")
    print("  - HuggingFace transformers (with conversion)")
    print("  - SGLang, vLLM, or other inference frameworks")
    print("\nExample usage:")
    print("  import torch")
    print("  checkpoint = torch.load('exported_model.pt', map_location='cpu')")
    print("  model.load_state_dict(checkpoint['model_state_dict'])")
