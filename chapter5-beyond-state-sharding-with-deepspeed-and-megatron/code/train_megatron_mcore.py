"""
Megatron Core Training Example

This script demonstrates training a GPT model using Megatron Core with
tensor parallelism. It shows how to:
1. Initialize distributed training with Megatron's parallel state
2. Create a GPT model using TransformerConfig
3. Wrap with DistributedDataParallel for gradient synchronization
4. Use Megatron's forward/backward scheduling

Requirements:
    pip install megatron-core

Usage:
    # Single node, 2 GPUs with tensor parallelism
    torchrun --nproc_per_node=2 train_megatron_mcore.py

    # Single node, 4 GPUs with tensor parallelism
    torchrun --nproc_per_node=4 train_megatron_mcore.py --tp-size 4

    # Multi-node (2 nodes, 4 GPUs each)
    torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
        --master_addr=<master_ip> --master_port=29500 \
        train_megatron_mcore.py --tp-size 4
"""

import os
import argparse
import torch
from torch.optim import Adam
from megatron.core import parallel_state
from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
from megatron.core.distributed import DistributedDataParallel
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.distributed.finalize_model_grads import finalize_model_grads


def parse_args():
    parser = argparse.ArgumentParser(description="Megatron Core Training Example")
    parser.add_argument("--tp-size", type=int, default=2,
                        help="Tensor model parallel size")
    parser.add_argument("--pp-size", type=int, default=1,
                        help="Pipeline model parallel size")
    parser.add_argument("--num-layers", type=int, default=12,
                        help="Number of transformer layers")
    parser.add_argument("--hidden-size", type=int, default=768,
                        help="Hidden size")
    parser.add_argument("--num-attention-heads", type=int, default=12,
                        help="Number of attention heads")
    parser.add_argument("--seq-length", type=int, default=512,
                        help="Sequence length")
    parser.add_argument("--micro-batch-size", type=int, default=4,
                        help="Micro batch size")
    parser.add_argument("--iterations", type=int, default=100,
                        help="Number of training iterations")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    return parser.parse_args()


def initialize_distributed(tensor_model_parallel_size, pipeline_model_parallel_size):
    """Initialize torch.distributed and Megatron-Core model parallel groups."""
    parallel_state.destroy_model_parallel()
    
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(
        backend="nccl", rank=rank, world_size=world_size,
        device_id=torch.device(f"cuda:{local_rank}")
    )
    
    # Initialize Megatron model parallelism
    parallel_state.initialize_model_parallel(
        tensor_model_parallel_size, pipeline_model_parallel_size
    )
    
    return rank, world_size, local_rank


def model_provider(args):
    """Build and return a GPT model using Megatron Core."""
    transformer_config = TransformerConfig(
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        num_attention_heads=args.num_attention_heads,
        use_cpu_initialization=True,
        pipeline_dtype=torch.bfloat16,
    )
    
    gpt_model = GPTModel(
        config=transformer_config,
        transformer_layer_spec=get_gpt_layer_local_spec(),
        vocab_size=50257,
        max_sequence_length=args.seq_length,
    )
    
    return gpt_model


def create_dummy_data_iterator(args, rank):
    """Create a dummy data iterator for demonstration."""
    while True:
        # Generate random data for demonstration
        tokens = torch.randint(0, 50257, (args.micro_batch_size, args.seq_length))
        attention_mask = torch.ones(args.micro_batch_size, 1, args.seq_length, args.seq_length)
        position_ids = torch.arange(args.seq_length).unsqueeze(0).expand(args.micro_batch_size, -1)
        labels = torch.randint(0, 50257, (args.micro_batch_size, args.seq_length))
        loss_mask = torch.ones(args.micro_batch_size, args.seq_length)
        
        yield {
            "tokens": tokens,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "labels": labels,
            "loss_mask": loss_mask,
        }


def forward_step_func(data_iterator, model):
    """Forward step function for training."""
    def loss_func(loss_mask, output_tensor):
        losses = output_tensor.float()
        loss_mask = loss_mask.view(-1).float()
        loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()
        return loss, {"lm loss": loss}
    
    data = next(data_iterator)
    tokens = data["tokens"].cuda()
    attention_mask = data["attention_mask"].cuda()
    position_ids = data["position_ids"].cuda()
    labels = data["labels"].cuda()
    loss_mask = data["loss_mask"].cuda()
    
    output_tensor = model(tokens, position_ids, attention_mask, labels=labels)
    
    return output_tensor, lambda: loss_func(loss_mask, output_tensor)


def main():
    args = parse_args()
    
    # Initialize distributed training
    rank, world_size, local_rank = initialize_distributed(
        tensor_model_parallel_size=args.tp_size,
        pipeline_model_parallel_size=args.pp_size
    )
    model_parallel_cuda_manual_seed(123)
    
    if rank == 0:
        print(f"Initialized distributed training:")
        print(f"  World size: {world_size}")
        print(f"  Tensor parallel size: {args.tp_size}")
        print(f"  Pipeline parallel size: {args.pp_size}")
    
    # Create model
    gpt_model = model_provider(args)
    gpt_model.cuda()
    
    # Count parameters
    num_params = sum(p.numel() for p in gpt_model.parameters())
    if rank == 0:
        print(f"  Model parameters: {num_params:,}")
    
    # Wrap with DistributedDataParallel
    config = gpt_model.config
    ddp_config = DistributedDataParallelConfig(
        grad_reduce_in_fp32=False,
        overlap_grad_reduce=True,
        use_distributed_optimizer=True,
    )
    gpt_model = DistributedDataParallel(
        config=config,
        ddp_config=ddp_config,
        module=gpt_model,
    )
    
    # Optimizer
    optim = Adam(gpt_model.parameters(), lr=args.lr)
    
    # Get forward/backward function
    forward_backward_func = get_forward_backward_func()
    
    # Create data iterator
    train_iterator = create_dummy_data_iterator(args, rank)
    
    if rank == 0:
        print(f"\nStarting training for {args.iterations} iterations...")
    
    # Training loop
    for iteration in range(args.iterations):
        optim.zero_grad()
        
        # Forward and backward pass
        losses_reduced = forward_backward_func(
            forward_step_func=forward_step_func,
            data_iterator=train_iterator,
            model=gpt_model,
            num_microbatches=1,
            seq_length=args.seq_length,
            micro_batch_size=args.micro_batch_size,
            decoder_seq_length=args.seq_length,
            forward_only=False,
        )
        
        # Finalize gradients
        finalize_model_grads([gpt_model])
        
        optim.step()
        
        if iteration % 10 == 0 and parallel_state.get_tensor_model_parallel_rank() == 0:
            print(f"Iteration {iteration}: Loss: {losses_reduced}")
    
    if rank == 0:
        print("\nTraining complete!")
    
    # Cleanup
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
