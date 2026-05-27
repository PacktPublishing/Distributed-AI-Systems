import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from transformers import AutoTokenizer, GPT2TokenizerFast
from transformers import T5Tokenizer, T5ForConditionalGeneration
import functools
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from transformers.models.t5.modeling_t5 import T5Block

# FSDP2 imports
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

from functools import partial
from torch.utils.data import DataLoader
from pathlib import Path
from summarization_dataset import wikihow
import policies
import model_checkpointing
from model_checkpointing.checkpoint_handler import (
    save_fsdp2_checkpoint,
    load_fsdp2_checkpoint,
)
from configs import fsdp_config, train_config
from utils import (bfloat_support,
                   cleanup, get_date_of_run,
                   format_metrics_to_gb,
                   train, validation)
from transformers.models.t5.modeling_t5 import T5Block
from typing import Type
import time
import tqdm
from datetime import datetime


def get_policies(cfg, rank):
    """establish current policies for mixed precision in FSDP2"""

    mp_policy = None

    # mixed precision -----
    if cfg.mixed_precision:
        bfloat_available = bfloat_support()
        if bfloat_available and not cfg.use_fp16:
            # FSDP2 uses MixedPrecisionPolicy (no buffer_dtype parameter)
            mp_policy = MixedPrecisionPolicy(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.bfloat16,
            )
            if rank == 0:
                print(f"bFloat16 enabled for mixed precision - using MixedPrecisionPolicy")
        elif cfg.use_fp16:
            mp_policy = MixedPrecisionPolicy(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
            )
            if rank == 0:
                print(f"FP16 enabled.")
        else:
            print(
                f"bFloat16 support not present. Will use FP32, and not mixed precision"
            )

    return mp_policy


def fsdp_main(args):

    # Use model_name from args if provided, otherwise use config default
    model_name = args.model_name if args.model_name else train_config.model_name
    
    # Validate that model name is a FLAN-T5 model
    if not ('flan-t5' in model_name.lower() or 'flan_t5' in model_name.lower()):
        raise ValueError(
            f"Model name must be a FLAN-T5 model (must contain 'flan-t5'). "
            f"Got: {model_name}. "
            f"Valid examples: google/flan-t5-small, google/flan-t5-base, "
            f"google/flan-t5-large, google/flan-t5-xl, google/flan-t5-xxl"
        )
    
    local_rank = int(os.environ['LOCAL_RANK'])
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    # Initialize process group for FSDP2
    if torch.accelerator.is_available():
        device_type = torch.accelerator.current_accelerator().type
        device = torch.device(f"{device_type}:{local_rank}")
        torch.accelerator.device_index(local_rank)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    backend = torch.distributed.get_default_backend_for_device(device)
    torch.distributed.init_process_group(backend=backend, device_id=device)
    
    torch.cuda.set_device(local_rank)

    # Create datasets (loads from local CSV files)
    if rank == 0:
        print("Loading training dataset...")
    tokenizer = T5Tokenizer.from_pretrained(model_name, legacy=False)
    train_dataset = wikihow(tokenizer, 'train', 1500, 512, 150, False, data_dir='data/')
    if rank == 0:
        print(f"Training dataset size: {len(train_dataset)}")
    
    if rank == 0:
        print("Loading validation dataset...")
    val_dataset = wikihow(tokenizer, 'validation', 300, 512, 150, False, data_dir='data/')
    if rank == 0:
        print(f"Validation dataset size: {len(val_dataset)}")

    sampler1 = DistributedSampler(train_dataset, rank=rank, num_replicas=world_size, shuffle=True)
    sampler2 = DistributedSampler(val_dataset, rank=rank, num_replicas=world_size)

    train_kwargs = {'batch_size': args.batch_size, 'sampler': sampler1}
    test_kwargs = {'batch_size': args.test_batch_size, 'sampler': sampler2}
    cuda_kwargs = {'num_workers': 2,
                    'pin_memory': True,
                    'shuffle': False}
    train_kwargs.update(cuda_kwargs)
    test_kwargs.update(cuda_kwargs)

    train_loader = torch.utils.data.DataLoader(train_dataset,**train_kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, **test_kwargs)

    # Set up FSDP2 mixed precision policy
    mp_policy = get_policies(train_config, rank)

    # Load model with FSDP2
    # FSDP2: Load model normally, then apply fully_shard to each block and root
    if rank == 0:
        print(f"Loading model {model_name}...")
    
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    model = model.to(device)
    
    # Apply FSDP2 fully_shard to each T5Block (encoder and decoder blocks)
    fsdp_kwargs = {}
    if mp_policy is not None:
        fsdp_kwargs["mp_policy"] = mp_policy
    
    # Shard encoder blocks
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'block'):
        for block in model.encoder.block:
            fully_shard(block, **fsdp_kwargs)
    
    # Shard decoder blocks
    if hasattr(model, 'decoder') and hasattr(model.decoder, 'block'):
        for block in model.decoder.block:
            fully_shard(block, **fsdp_kwargs)
    
    # Shard the entire model (root)
    fully_shard(model, **fsdp_kwargs)
    
    if rank == 0:
        print("Model sharded with FSDP2")

    # Enabling activation checkpointing for FSDP2
    if fsdp_config.fsdp_activation_checkpointing:
        policies.apply_fsdp_checkpointing(model)

    # Set up optimizer and scheduler
    optimizer = optim.AdamW(model.parameters(), lr=train_config.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)
    best_val_loss = float("inf")
    curr_val_loss = float("inf")
    file_save_name = "T5-model-"
    
    # Setup checkpoint directory for FSDP2 DCP
    checkpoint_dir = Path.cwd() / "checkpoints" / model_name.replace("/", "_")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if rank == 0:
        time_of_run = get_date_of_run()
        dur = []
        train_acc_tracking = []
        val_acc_tracking = []
        training_start_time = time.time()
        # Training-only timing (excludes setup, data loading, model loading)
        training_only_start_time = None

    if rank == 0 and args.track_memory:
        mem_alloc_tracker = []
        mem_reserved_tracker = []

    # Start training-only timer before first epoch
    if rank == 0:
        training_only_start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        is_last_epoch = (epoch == args.epochs)
        train_accuracy = train(args, model, rank, world_size, train_loader, optimizer, epoch, sampler=sampler1, is_last_epoch=is_last_epoch)
        if args.run_validation:
            curr_val_loss = validation(model, rank, world_size, val_loader)
        scheduler.step()

        if rank == 0:

            print(f"--> epoch {epoch} completed...entering save and stats zone")

            dur.append(time.time() - t0)
            train_acc_tracking.append(train_accuracy.item())

            if args.run_validation:
                val_acc_tracking.append(curr_val_loss.item())

            if args.track_memory:
                mem_alloc_tracker.append(
                    format_metrics_to_gb(torch.cuda.memory_allocated())
                )
                mem_reserved_tracker.append(
                    format_metrics_to_gb(torch.cuda.memory_reserved())
                )

        if train_config.save_model and curr_val_loss < best_val_loss:
            # FSDP2 checkpointing using Distributed Checkpointing (DCP)
            # Each GPU saves its shard in parallel, avoiding OOM
            save_fsdp2_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                rank=rank,
                checkpoint_dir=checkpoint_dir,
                save_optimizer=True,
            )

        if curr_val_loss < best_val_loss:

            best_val_loss = curr_val_loss
            if rank==0:
                print(f"-->>>> New Val Loss Record: {best_val_loss}")

    # Training-only timing summary (excludes setup, data loading, model loading)
    if rank == 0 and training_only_start_time is not None:
        training_only_total_time = time.time() - training_only_start_time
        print("\n" + "="*70)
        print("Training-Only Time Summary (excludes setup/data loading)")
        print("="*70)
        print(f"Total training time: {training_only_total_time:.2f} seconds ({training_only_total_time/60:.2f} minutes)")
        print(f"Average time per epoch: {sum(dur)/len(dur):.2f} seconds")
        print(f"Total epochs: {args.epochs}")
        print("="*70)

    # Barrier
    dist.barrier()
    cleanup()


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch FLAN-T5 FSDP2 Example')
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 4)')
    parser.add_argument('--test-batch-size', type=int, default=4, metavar='N',
                        help='input batch size for testing (default: 4)')
    parser.add_argument('--epochs', type=int, default=4, metavar='N',
                        help='number of epochs to train (default: 4)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--track_memory', action='store_false', default=True,
                        help='track the gpu memory')
    parser.add_argument('--run_validation', action='store_false', default=True,
                        help='running the validation')
    parser.add_argument('--model-name', type=str, default=None,
                        help=f'model name to use (default: {train_config.model_name})')
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    fsdp_main(args)
