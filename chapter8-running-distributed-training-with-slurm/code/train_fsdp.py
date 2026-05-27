#!/usr/bin/env python3
"""
PyTorch FSDP (Fully Sharded Data Parallel) training script for SLURM.

This script demonstrates FSDP training with automatic SLURM environment detection.
Works with both single-node multi-GPU and multi-node setups.
"""

import os
import socket
import argparse
import multiprocessing
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import CPUOffload
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler


class SimpleDataset(Dataset):
    """Simple synthetic dataset for demonstration."""
    def __init__(self, size=1000, input_dim=10):
        self.size = size
        self.input_dim = input_dim
        self.data = torch.randn(size, input_dim)
        self.targets = torch.randint(0, 2, (size,))
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]


class MyLargeModel(nn.Module):
    """Large model for FSDP demonstration."""
    def __init__(self, input_dim=10, hidden_dim=256, num_layers=4, num_classes=2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
        
        self.layers.append(nn.Linear(hidden_dim, num_classes))
        self.dropout = nn.Dropout(0.2)
    
    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = F.relu(layer(x))
            x = self.dropout(x)
        x = self.layers[-1](x)
        return x


def setup_distributed():
    """Initialize distributed training using SLURM or torchrun environment variables."""
    # Fix IPv6 resolution issues by forcing IPv4
    # These environment variables prevent c10d from trying to resolve IPv6 addresses
    os.environ['NCCL_SOCKET_IFNAME'] = '^docker,lo'
    os.environ['GLOO_SOCKET_IFNAME'] = '^docker,lo'
    
    # Check if MASTER_ADDR is already set (e.g., by user as 127.0.0.1)
    # If set, use it directly to avoid hostname resolution
    user_master_addr = os.environ.get('MASTER_ADDR', '')
    
    # Check if we're in a SLURM environment
    if 'SLURM_PROCID' in os.environ:
        # SLURM mode: use SLURM environment variables
        rank = int(os.environ['SLURM_PROCID'])
        world_size = int(os.environ['SLURM_NTASKS'])
        local_rank = int(os.environ['SLURM_LOCALID'])
        
        # For single-node multi-GPU setups, determine the actual GPU device
        # SLURM may set CUDA_VISIBLE_DEVICES incorrectly for virtual nodes (e.g., both get 0)
        # We need to override it based on node name (node6 -> GPU 6, node7 -> GPU 7)
        node_name = os.environ.get('SLURMD_NODENAME', '')
        
        # Check current CUDA_VISIBLE_DEVICES
        cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        
        # For single-node multi-GPU setups with virtual nodes (node6, node7)
        # Extract GPU number from node name and override CUDA_VISIBLE_DEVICES
        # This is necessary because SLURM may set it incorrectly (e.g., both nodes get 0)
        if node_name.startswith('node'):
            try:
                gpu_num = int(node_name.replace('node', ''))
                # Override CUDA_VISIBLE_DEVICES even if SLURM already set it
                # This ensures each process only sees its assigned GPU
                os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_num)
                if rank == 0:
                    if cuda_visible:
                        print(f"Overriding CUDA_VISIBLE_DEVICES from '{cuda_visible}' to '{gpu_num}' for node {node_name}")
                    else:
                        print(f"Setting CUDA_VISIBLE_DEVICES={gpu_num} for node {node_name}")
            except ValueError:
                # If node name doesn't match pattern, keep existing or use LOCAL_RANK
                if not cuda_visible:
                    os.environ['CUDA_VISIBLE_DEVICES'] = str(local_rank)
                if rank == 0:
                    print(f"Warning: Could not extract GPU number from node name '{node_name}', using CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'LOCAL_RANK')}")
        else:
            # Fallback: use LOCAL_RANK if node name doesn't match pattern
            if not cuda_visible:
                os.environ['CUDA_VISIBLE_DEVICES'] = str(local_rank)
            if rank == 0:
                print(f"Warning: Node name '{node_name}' doesn't match 'node*' pattern, using CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'LOCAL_RANK')}")
        
        # Get master address from SLURM
        # If user set MASTER_ADDR, use it (e.g., 127.0.0.1 for single-node setups)
        if user_master_addr:
            master_addr = user_master_addr
            # Convert localhost to 127.0.0.1 to avoid resolution
            if master_addr == 'localhost':
                master_addr = '127.0.0.1'
        elif 'SLURM_JOB_NODELIST' in os.environ:
            master_hostname = os.popen(
                f"scontrol show hostnames {os.environ['SLURM_JOB_NODELIST']} | head -n 1"
            ).read().strip()
            # For single-node multi-GPU setups, use localhost to avoid IPv6 resolution
            # Check if all nodes are on the same physical host
            all_hostnames = os.popen(
                f"scontrol show hostnames {os.environ['SLURM_JOB_NODELIST']}"
            ).read().strip().split('\n')
            all_hostnames = [h for h in all_hostnames if h]
            
            # If only one unique hostname or all resolve to same IP, use localhost
            try:
                current_host_ip = socket.gethostbyname(socket.gethostname().split('.')[0])
                all_same_host = True
                for hostname in all_hostnames:
                    try:
                        node_ip = socket.gethostbyname(hostname.split('.')[0])
                        if node_ip != current_host_ip and node_ip != '127.0.0.1':
                            all_same_host = False
                            break
                    except socket.gaierror:
                        all_same_host = False
                        break
                
                if all_same_host:
                    master_addr = '127.0.0.1'  # Use localhost for single-node setups
                else:
                    # Multi-node: resolve to IPv4 explicitly
                    try:
                        addr_info = socket.getaddrinfo(master_hostname.split('.')[0], None, socket.AF_INET)
                        master_addr = addr_info[0][4][0]
                    except (socket.gaierror, IndexError):
                        master_addr = '127.0.0.1'  # Fallback
            except (socket.gaierror, socket.herror):
                # If resolution fails, use localhost for single-node setups
                master_addr = '127.0.0.1'
        else:
            master_addr = '127.0.0.1'  # Default to localhost
        
        master_port = os.environ.get('MASTER_PORT', '29500')
    else:
        # Non-SLURM mode: use torchrun environment variables
        rank = int(os.environ.get('RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', 1))
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        # Check if MASTER_ADDR is set, otherwise use localhost
        master_addr = os.environ.get('MASTER_ADDR', '127.0.0.1')
        if master_addr == 'localhost':
            master_addr = '127.0.0.1'
        master_port = os.environ.get('MASTER_PORT', '29500')
    
    # If world_size is 1, we're running single process - skip distributed setup
    if world_size == 1:
        print("Running in single-process mode (no distributed training)")
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        return 0, 1, 0, device, False
    
    # Set up the process group for distributed training
    # Always use IPv4 address to avoid IPv6 resolution warnings
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = str(master_port)
    
    # Use explicit init_method with IPv4 address to prevent hostname resolution
    init_method = f'tcp://{master_addr}:{master_port}'
    
    dist.init_process_group(
        backend='nccl',
        init_method=init_method,
        rank=rank,
        world_size=world_size
    )
    
    # Set device - after CUDA_VISIBLE_DEVICES is set, each process only sees one GPU (at index 0)
    # So we use device 0 for the device index, but they're actually different physical GPUs
    actual_device_id = 0  # After CUDA_VISIBLE_DEVICES remapping, always use device 0
    torch.cuda.set_device(actual_device_id)
    device = torch.device(f'cuda:{actual_device_id}')
    
    # Verify GPU assignment - each process should see exactly one GPU
    num_visible_gpus = torch.cuda.device_count()
    if num_visible_gpus != 1:
        if rank == 0:
            print(f"WARNING: Expected 1 GPU after CUDA_VISIBLE_DEVICES remapping, but found {num_visible_gpus}")
            print(f"  This may cause duplicate GPU errors. CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
    
    return rank, world_size, local_rank, device, True


def cleanup_distributed(is_distributed):
    """Clean up distributed training."""
    if is_distributed:
        dist.destroy_process_group()


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='FSDP Training with SLURM')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size per GPU')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--dataset-size', type=int, default=1000, help='Dataset size')
    parser.add_argument('--input-dim', type=int, default=10, help='Input dimension')
    parser.add_argument('--hidden-dim', type=int, default=256, help='Hidden dimension')
    parser.add_argument('--num-layers', type=int, default=4, help='Number of layers')
    parser.add_argument('--num-workers', type=int, default=None, 
                       help='Number of DataLoader workers (default: auto-detect from SLURM CPU allocation)')
    args = parser.parse_args()
    
    # Setup distributed training
    rank, world_size, local_rank, device, is_distributed = setup_distributed()
    
    if rank == 0:
        print(f'Training configuration:')
        print(f'  World size: {world_size}')
        print(f'  Rank: {rank}, Local rank: {local_rank}')
        print(f'  Device: {device}')
        print(f'  Distributed: {is_distributed}')
        print(f'  Using FSDP: {is_distributed}')
        print(f'  CUDA_VISIBLE_DEVICES: {os.environ.get("CUDA_VISIBLE_DEVICES", "not set")}')
        print(f'  Actual GPU device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"}')
        print(f'  Epochs: {args.epochs}')
        print(f'  Batch size per GPU: {args.batch_size}')
        print(f'  Total batch size: {args.batch_size * world_size}')
    
    # Print GPU info for all ranks to debug device assignment
    print(f'[Rank {rank}] Node: {os.environ.get("SLURMD_NODENAME", "unknown")}, '
          f'CUDA_VISIBLE_DEVICES: {os.environ.get("CUDA_VISIBLE_DEVICES", "not set")}, '
          f'Device: {device}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"}')
    
    # Determine number of workers for DataLoader based on available CPUs
    # User can override with --num-workers argument
    if args.num_workers is not None:
        num_workers = args.num_workers
    else:
        # Auto-detect: Check SLURM CPU allocation first, then fall back to system detection
        if 'SLURM_CPUS_PER_TASK' in os.environ:
            # SLURM explicitly allocated CPUs per task
            cpus_per_task = int(os.environ['SLURM_CPUS_PER_TASK'])
            # Use CPUs - 1 to leave one for the main process, but at least 1
            num_workers = max(1, cpus_per_task - 1)
        elif 'SLURM_JOB_CPUS_PER_NODE' in os.environ:
            # SLURM allocated CPUs per node, divide by number of tasks on this node
            cpus_per_node = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])
            tasks_per_node = int(os.environ.get('SLURM_NTASKS_PER_NODE', world_size))
            cpus_per_task = cpus_per_node // tasks_per_node if tasks_per_node > 0 else cpus_per_node
            num_workers = max(1, cpus_per_task - 1)
        else:
            # Fall back to system CPU count
            # Use os.cpu_count() or multiprocessing.cpu_count()
            try:
                system_cpus = os.cpu_count() or multiprocessing.cpu_count()
                # Use a reasonable fraction (e.g., half) but at least 1
                num_workers = max(1, system_cpus // 2)
            except:
                num_workers = 1
    
    if rank == 0:
        print(f'  DataLoader num_workers: {num_workers} (auto-detected from available CPUs)')
    
    # Create model and move to device before wrapping with FSDP
    # This avoids the warning about CPU initialization and improves performance
    model = MyLargeModel(input_dim=args.input_dim, hidden_dim=args.hidden_dim, num_layers=args.num_layers)
    model = model.to(device)
    
    if is_distributed:
        # Wrap with FSDP - model is already on GPU, so pass device_id for proper sharding
        # device_id should be the device index (0 after CUDA_VISIBLE_DEVICES remapping)
        device_id = device.index if device.index is not None else 0
        model = FSDP(
            model,
            auto_wrap_policy=size_based_auto_wrap_policy,
            cpu_offload=CPUOffload(offload_params=False),  # Set to True to offload to CPU
            device_id=device_id,  # Specify device_id to avoid CPU initialization warning
        )
    # Single GPU mode - model already moved to device above
    
    # Create dataset and dataloader
    dataset = SimpleDataset(size=args.dataset_size, input_dim=args.input_dim)
    
    if is_distributed:
        # Use DistributedSampler for multi-process training
        sampler = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True
        )
    else:
        # Use regular DataLoader for single process
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
    
    # Setup optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    if rank == 0:
        print('\nStarting training...\n')
    
    for epoch in range(args.epochs):
        # Set epoch for distributed sampler (important for shuffling)
        if is_distributed:
            sampler.set_epoch(epoch)
        
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        if rank == 0:
            print(f"Epoch {epoch} completed, Average Loss: {avg_loss:.4f}")
    
    if rank == 0:
        print('\nTraining completed!')
    
    # Cleanup
    cleanup_distributed(is_distributed)


if __name__ == '__main__':
    main()
