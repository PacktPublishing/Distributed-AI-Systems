#!/usr/bin/env python3
"""
Checkpoint utility script for distributed training.

This script can be called to save checkpoints during training,
especially useful when handling preemption signals from SLURM.
"""

import os
import argparse
import torch
import torch.distributed as dist


def save_checkpoint(checkpoint_dir, epoch=None):
    """Save a checkpoint manually."""
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory {checkpoint_dir} does not exist")
        return
    
    # Find the latest checkpoint
    latest_checkpoint = os.path.join(checkpoint_dir, 'latest.pt')
    
    if os.path.exists(latest_checkpoint):
        # Copy latest to a manual checkpoint
        import shutil
        manual_checkpoint = os.path.join(checkpoint_dir, f'manual_checkpoint.pt')
        shutil.copy(latest_checkpoint, manual_checkpoint)
        print(f"Saved manual checkpoint to {manual_checkpoint}")
    else:
        print("No checkpoint found to save")


def main():
    parser = argparse.ArgumentParser(description='Checkpoint Utility')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', 
                       help='Checkpoint directory')
    parser.add_argument('--epoch', type=int, default=None, 
                       help='Epoch number (optional)')
    args = parser.parse_args()
    
    save_checkpoint(args.checkpoint_dir, args.epoch)


if __name__ == '__main__':
    main()
