"""
# Basic training
python T5_training_Single.py

# With mixed precision
python T5_training_Single.py --use-amp

# Custom batch size and epochs
python T5_training_Single.py --batch-size 8 --epochs 5

# Save model checkpoints
python T5_training_Single.py --save-model

# Track GPU memory
python T5_training_Single.py --track-memory
"""
import os
import argparse
import torch
import torch.optim as optim
from transformers import T5Tokenizer, T5ForConditionalGeneration
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from summarization_dataset import wikihow
from configs import train_config
from utils import setup_model, get_date_of_run, format_metrics_to_gb, bfloat_support
import time
import tqdm
from datetime import datetime

g_gigabyte = 1024**3


def train_single_gpu(model, train_loader, optimizer, epoch, device, use_amp=False, is_last_epoch=False):
    """Training function for single GPU"""
    model.train()
    total_loss = 0.0
    total_samples = 0
    
    inner_pbar = tqdm.tqdm(
        range(len(train_loader)), colour="blue", desc=f"Training Epoch {epoch}"
    )
    
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    # Use bfloat16 if available, otherwise FP16 (for fair comparison with FSDP)
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16 if use_amp else None
    
    total_iterations = len(train_loader)
    for iteration, batch in enumerate(train_loader):
        # Print memory before last iteration of last epoch
        if is_last_epoch and iteration == total_iterations - 1:
            if device.type == 'cuda' and torch.cuda.is_available():
                torch.cuda.synchronize()  # Ensure all operations are complete
                allocated_gb = torch.cuda.memory_allocated(device) / (1024**3)
                reserved_gb = torch.cuda.memory_reserved(device) / (1024**3)
                max_allocated_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
                print(f"\n{'='*70}")
                print(f"Memory Usage BEFORE Last Iteration (Epoch {epoch}, Iteration {iteration+1}/{total_iterations}):")
                print(f"{'='*70}")
                print(f"GPU {device.index if device.index is not None else 0}:")
                print(f"  Allocated: {allocated_gb:.2f} GB")
                print(f"  Reserved:  {reserved_gb:.2f} GB")
                print(f"  Max Allocated (peak): {max_allocated_gb:.2f} GB")
                print(f"{'='*70}\n")
        # Move batch to device
        for key in batch.keys():
            batch[key] = batch[key].to(device)
        
        optimizer.zero_grad()
        
        if use_amp and scaler is not None:
            # Use automatic mixed precision (bfloat16 if available, matching FSDP)
            with torch.amp.autocast(device_type='cuda', dtype=amp_dtype):
                output = model(
                    input_ids=batch["source_ids"],
                    attention_mask=batch["source_mask"],
                    labels=batch["target_ids"]
                )
                loss = output["loss"]
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard training
            output = model(
                input_ids=batch["source_ids"],
                attention_mask=batch["source_mask"],
                labels=batch["target_ids"]
            )
            loss = output["loss"]
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        total_samples += len(batch)
        inner_pbar.update(1)
    
    inner_pbar.close()
    avg_loss = total_loss / total_samples
    print(f"Train Epoch: {epoch}, Loss: {avg_loss:.4f}")
    return torch.tensor(avg_loss)


def validation_single_gpu(model, val_loader, device, use_amp=False):
    """Validation function for single GPU"""
    model.eval()
    total_loss = 0.0
    total_samples = 0
    
    # Use bfloat16 if available, otherwise FP16 (for fair comparison with FSDP)
    amp_dtype = torch.bfloat16 if (use_amp and torch.cuda.is_bf16_supported()) else torch.float16 if use_amp else None
    
    inner_pbar = tqdm.tqdm(
        range(len(val_loader)), colour="green", desc="Validation"
    )
    
    with torch.no_grad():
        for batch in val_loader:
            # Move batch to device
            for key in batch.keys():
                batch[key] = batch[key].to(device)
            
            if use_amp and amp_dtype is not None:
                # Use automatic mixed precision (bfloat16 if available, matching FSDP)
                with torch.amp.autocast(device_type='cuda', dtype=amp_dtype):
                    output = model(
                        input_ids=batch["source_ids"],
                        attention_mask=batch["source_mask"],
                        labels=batch["target_ids"]
                    )
            else:
                output = model(
                    input_ids=batch["source_ids"],
                    attention_mask=batch["source_mask"],
                    labels=batch["target_ids"]
                )
            
            loss = output["loss"]
            total_loss += loss.item()
            total_samples += len(batch)
            inner_pbar.update(1)
    
    inner_pbar.close()
    avg_loss = total_loss / total_samples
    print(f"Validation Loss: {avg_loss:.4f}")
    return torch.tensor(avg_loss)


def save_checkpoint(model, optimizer, epoch, loss, save_path):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def main(args):
    """Main training function for single GPU"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cpu':
        print("Warning: CUDA not available, using CPU. Training will be very slow.")
    
    # Setup model and tokenizer
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
    
    model, tokenizer = setup_model(model_name)
    model = model.to(device)
    
    # Create datasets (loads from local CSV files)
    print("Loading training dataset...")
    train_dataset = wikihow(tokenizer, 'train', 1500, 512, 150, False, data_dir='data/')
    print(f"Training dataset size: {len(train_dataset)}")
    
    print("Loading validation dataset...")
    val_dataset = wikihow(tokenizer, 'validation', 300, 512, 150, False, data_dir='data/')
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Create data loaders (no distributed sampler needed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.test_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Setup optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=train_config.lr,
        weight_decay=train_config.weight_decay
    )
    
    scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)
    
    # Determine if we should use mixed precision
    # Default to True to match FSDP's bfloat16 default (for fair comparison)
    use_amp = args.use_amp and device.type == 'cuda' and torch.cuda.is_available()
    
    if use_amp:
        bfloat_available = torch.cuda.is_bf16_supported()
        if bfloat_available:
            print("bFloat16 enabled for mixed precision - using bfSixteen policy")
        else:
            print("Using Automatic Mixed Precision (AMP) with FP16 for training")
    else:
        print("Using FP32 precision for training")
    
    # Training tracking
    time_of_run = get_date_of_run()
    dur = []
    train_loss_tracking = []
    val_loss_tracking = []
    training_start_time = time.time()
    # Training-only timing (excludes setup, data loading, model loading)
    training_only_start_time = None
    best_val_loss = float("inf")
    
    if args.track_memory and device.type == 'cuda':
        mem_alloc_tracker = []
        mem_reserved_tracker = []
    
    # Start training-only timer before first epoch
    training_only_start_time = time.time()
    
    # Training loop
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        
        # Train
        is_last_epoch = (epoch == args.epochs)
        train_loss = train_single_gpu(
            model, train_loader, optimizer, epoch, device, use_amp=use_amp, is_last_epoch=is_last_epoch
        )
        
        # Validate
        if args.run_validation:
            curr_val_loss = validation_single_gpu(
                model, val_loader, device, use_amp=use_amp
            )
        else:
            curr_val_loss = torch.tensor(float("inf"))
        
        scheduler.step()
        
        # Track metrics
        dur.append(time.time() - t0)
        train_loss_tracking.append(train_loss.item())
        
        if args.run_validation:
            val_loss_tracking.append(curr_val_loss.item())
        
        if args.track_memory and device.type == 'cuda':
            mem_alloc_tracker.append(
                format_metrics_to_gb(torch.cuda.memory_allocated())
            )
            mem_reserved_tracker.append(
                format_metrics_to_gb(torch.cuda.memory_reserved())
            )
            print(f"GPU Memory - Allocated: {mem_alloc_tracker[-1]:.4f} GB, "
                  f"Reserved: {mem_reserved_tracker[-1]:.4f} GB")
        
        print(f"--> Epoch {epoch} completed in {dur[-1]:.2f} seconds")
        
        # Save checkpoint if validation loss improved
        if args.run_validation and curr_val_loss.item() < best_val_loss:
            best_val_loss = curr_val_loss.item()
            if args.save_model:
                checkpoint_path = f"T5-model-epoch-{epoch}-loss-{best_val_loss:.4f}.pt"
                save_checkpoint(model, optimizer, epoch, best_val_loss, checkpoint_path)
            print(f"-->>>> New Best Val Loss Record: {best_val_loss:.4f}")
    
    # Training summary
    total_time = time.time() - training_start_time
    training_only_total_time = time.time() - training_only_start_time if training_only_start_time else total_time
    
    print("\n" + "="*50)
    print("Training Summary")
    print("="*50)
    print(f"Total time (including setup): {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Training-only time (excludes setup/data loading): {training_only_total_time:.2f} seconds ({training_only_total_time/60:.2f} minutes)")
    print(f"Average time per epoch: {sum(dur)/len(dur):.2f} seconds")
    print(f"Total epochs: {args.epochs}")
    print(f"Final training loss: {train_loss_tracking[-1]:.4f}")
    if args.run_validation:
        print(f"Best validation loss: {best_val_loss:.4f}")
    print("="*50)


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch T5 Single GPU Training Example')
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 4)')
    parser.add_argument('--test-batch-size', type=int, default=4, metavar='N',
                        help='input batch size for testing (default: 4)')
    parser.add_argument('--epochs', type=int, default=4, metavar='N',
                        help='number of epochs to train (default: 4)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--num-workers', type=int, default=2, metavar='N',
                        help='number of data loading workers (default: 2)')
    parser.add_argument('--use-amp', action='store_true', default=True,
                        help='use automatic mixed precision with bfloat16 (default: True, matches FSDP)')
    parser.add_argument('--no-amp', action='store_false', dest='use_amp',
                        help='disable automatic mixed precision (use FP32)')
    parser.add_argument('--track-memory', action='store_true', default=False,
                        help='track GPU memory usage (default: False)')
    parser.add_argument('--no-run-validation', action='store_false', default=True,
                        dest='run_validation',
                        help='disable validation after each epoch (default: validation enabled)')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='save model checkpoint when validation loss improves (default: False)')
    parser.add_argument('--model-name', type=str, default=None,
                        help=f'model name to use (default: {train_config.model_name})')
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    main(args)

