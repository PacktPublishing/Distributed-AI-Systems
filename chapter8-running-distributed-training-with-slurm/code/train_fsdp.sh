#!/bin/bash
#SBATCH --job-name=fsdp-training
#SBATCH --nodes=2                    # 2 nodes (one GPU per node)
#SBATCH --gres=gpu:1                # 1 GPU per node
#SBATCH --ntasks-per-node=1         # 1 task per node
#SBATCH --cpus-per-task=4
#SBATCH --mem=50G
#SBATCH --time=1:00:00
#SBATCH --output=logs/train_fsdp_%j_%N.out
#SBATCH --error=logs/train_fsdp_%j_%N.err

# This script runs FSDP training with train_fsdp.py across 2 nodes (node6 and node7)
# Each node gets 1 GPU, for a total of 2 GPUs
#
# Usage:
#   mkdir -p logs  # Create logs directory before submitting
#   sbatch train_fsdp.sh
#
# Logs:
#   - Logs are written to the logs/ directory
#   - Each node writes to separate log files:
#     - logs/train_fsdp_<job_id>_node6.out/err (for node6, rank 0)
#     - logs/train_fsdp_<job_id>_node7.out/err (for node7, rank 1)
#   - View logs: tail -f logs/train_fsdp_<job_id>_node6.out
#   - View all logs: tail -f logs/train_fsdp_<job_id>_*.out

# Activate conda environment
# Initialize conda (adjust path if needed)
if [ -f ~/miniconda3/etc/profile.d/conda.sh ]; then
    source ~/miniconda3/etc/profile.d/conda.sh
elif [ -f ~/anaconda3/etc/profile.d/conda.sh ]; then
    source ~/anaconda3/etc/profile.d/conda.sh
fi
conda activate research

# Get the directory where this script is located and create logs directory
# Use the actual script directory (where sbatch was submitted from)
SCRIPT_DIR="/home/wukong/workspace/coderepo/08/code"
cd "$SCRIPT_DIR"
mkdir -p logs

echo "=========================================="
echo "FSDP Training with train_fsdp.py"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "Total tasks: $SLURM_NTASKS"
echo "Node list: $SLURM_JOB_NODELIST"
echo "Current node: $SLURMD_NODENAME"
echo "Logs directory: $SCRIPT_DIR/logs"
echo ""

# Get master node address - use localhost for single-node multi-GPU setup
# For true multi-node, you would use the first node's hostname
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID

echo "Master address: $MASTER_ADDR"
echo "Master port: $MASTER_PORT"
echo "World size: $WORLD_SIZE"
echo "Rank: $RANK"
echo "Local rank: $LOCAL_RANK"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo ""

# Display node and GPU information
echo "Node information:"
srun hostname
echo ""

echo "GPU information:"
srun nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

# NCCL settings
export NCCL_DEBUG=WARN  # Set to INFO for more details
export NCCL_SOCKET_IFNAME=^docker,lo
export NCCL_IB_DISABLE=0  # Enable InfiniBand if available

echo "=========================================="
echo "Starting FSDP training on node $SLURMD_NODENAME (rank $RANK)..."
echo "=========================================="
echo ""

# Get Python command from conda environment
PYTHON_CMD=$(which python)
echo "Python command: $PYTHON_CMD"
echo "Python version: $($PYTHON_CMD --version)"
echo ""

# Use absolute path to train_fsdp.py
TRAIN_SCRIPT="$SCRIPT_DIR/train_fsdp.py"
echo "Train script path: $TRAIN_SCRIPT"
echo "Verifying train_fsdp.py exists:"
ls -la "$TRAIN_SCRIPT" || echo "ERROR: train_fsdp.py not found at $TRAIN_SCRIPT"
echo ""

# Run FSDP training using srun directly (train_fsdp.py handles SLURM env vars)
# Use --chdir to ensure correct working directory and --label to identify ranks
# Since train_fsdp.py already handles SLURM environment variables, we don't need torchrun
srun --chdir="$SCRIPT_DIR" --label "$PYTHON_CMD" "$TRAIN_SCRIPT" \
    --epochs 10 \
    --batch-size 32 \
    --lr 0.001 \
    --dataset-size 1000 \
    --input-dim 10

echo ""
echo "=========================================="
echo "FSDP training completed on node $SLURMD_NODENAME (rank $RANK)!"
echo "=========================================="
