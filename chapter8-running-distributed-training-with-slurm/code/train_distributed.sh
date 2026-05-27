#!/bin/bash
#SBATCH --job-name=distributed-training
#SBATCH --nodes=2                    # 2 nodes (one GPU per node)
#SBATCH --gres=gpu:1                # 1 GPU per node
#SBATCH --ntasks-per-node=1         # 1 task per node
#SBATCH --cpus-per-task=28
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=train_distributed_%j.out
#SBATCH --error=train_distributed_%j.err
#SBATCH --signal=SIGUSR1@90         # Send signal 90 seconds before time limit

# Setup distributed training environment
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

# NCCL settings
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=^docker,lo

echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "World size: $WORLD_SIZE, Rank: $RANK, Local rank: $LOCAL_RANK"
echo "Node list: $SLURM_JOB_NODELIST"

# Handle checkpoint signal
trap 'echo "Checkpointing..."; python checkpoint.py --checkpoint_dir ./checkpoints' SIGUSR1

# Run training with torchrun
torchrun \
    --nproc_per_node=1 \
    --nnodes=$SLURM_JOB_NUM_NODES \
    --node_rank=$SLURM_NODEID \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train_distributed.py \
    --resume \
    --checkpoint_dir ./checkpoints \
    --epochs 100 \
    --batch-size 32 \
    --lr 0.001
