#!/bin/bash
#SBATCH --array=0-9                 # Run 10 jobs (0-9)
#SBATCH --job-name=hyperparam-tune
#SBATCH --nodes=1                    # Single node
#SBATCH --gres=gpu:1                # 1 GPU per job
#SBATCH --cpus-per-task=14
#SBATCH --mem=100G
#SBATCH --time=4:00:00
#SBATCH --output=train_array_%A_%a.out
#SBATCH --error=train_array_%A_%a.err

# Each array task gets different hyperparameters
# Array of learning rates
LR_ARRAY=(0.001 0.0001 0.00001 0.000001)
LR=${LR_ARRAY[$((SLURM_ARRAY_TASK_ID % 4))]}

# Array of batch sizes
BATCH_SIZE_ARRAY=(32 64 128 256)
BATCH_IDX=$((SLURM_ARRAY_TASK_ID / 4))
BATCH_SIZE=${BATCH_SIZE_ARRAY[$((BATCH_IDX % 4))]}

echo "=========================================="
echo "Job Array Task: $SLURM_ARRAY_TASK_ID"
echo "Learning Rate: $LR"
echo "Batch Size: $BATCH_SIZE"
echo "=========================================="

# Run training with hyperparameters
python train.py \
    --lr $LR \
    --batch-size $BATCH_SIZE \
    --epochs 10
