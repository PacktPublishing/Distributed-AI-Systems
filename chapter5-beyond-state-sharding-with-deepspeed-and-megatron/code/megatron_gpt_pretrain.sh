#!/bin/bash
# Megatron-LM GPT Pretraining Script
#
# This script demonstrates how to run Megatron-LM for GPT pretraining
# with various parallelism configurations.
#
# Prerequisites:
#   1. Clone Megatron-LM: git clone https://github.com/NVIDIA/Megatron-LM.git
#   2. Install dependencies: pip install -r requirements.txt
#   3. Prepare data using Megatron's preprocessing tools
#
# Usage:
#   # Single node, 8 GPUs with tensor parallelism
#   bash megatron_gpt_pretrain.sh
#
#   # Multi-node with SLURM (modify SLURM parameters below)
#   sbatch megatron_gpt_pretrain.sh

# Exit on error
set -e

# ============================================================================
# Configuration
# ============================================================================

# Paths (modify these for your setup)
MEGATRON_PATH="${MEGATRON_PATH:-/path/to/Megatron-LM}"
DATA_PATH="${DATA_PATH:-/path/to/data/gpt2_text_document}"
VOCAB_FILE="${VOCAB_FILE:-/path/to/gpt2-vocab.json}"
MERGE_FILE="${MERGE_FILE:-/path/to/gpt2-merges.txt}"
CHECKPOINT_PATH="${CHECKPOINT_PATH:-./checkpoints/gpt}"
TENSORBOARD_PATH="${TENSORBOARD_PATH:-./tensorboard/gpt}"

# Model configuration (GPT-2 Small for demonstration)
NUM_LAYERS=12
HIDDEN_SIZE=768
NUM_ATTENTION_HEADS=12
SEQ_LENGTH=1024

# Parallelism configuration
TENSOR_PARALLEL_SIZE=2    # Split layers across GPUs
PIPELINE_PARALLEL_SIZE=1  # Split model depth across GPUs
WORLD_SIZE=2              # Total GPUs

# Training configuration
MICRO_BATCH_SIZE=4
GLOBAL_BATCH_SIZE=32
TRAIN_ITERS=10000
LR=6e-4
MIN_LR=6e-5
WARMUP_ITERS=500

# ============================================================================
# Derived Configuration
# ============================================================================

# Calculate data parallel size
DATA_PARALLEL_SIZE=$((WORLD_SIZE / (TENSOR_PARALLEL_SIZE * PIPELINE_PARALLEL_SIZE)))

# Gradient accumulation steps
GRADIENT_ACCUMULATION_STEPS=$((GLOBAL_BATCH_SIZE / (MICRO_BATCH_SIZE * DATA_PARALLEL_SIZE)))

echo "============================================"
echo "Megatron GPT Pretraining Configuration"
echo "============================================"
echo "Model: ${NUM_LAYERS} layers, ${HIDDEN_SIZE} hidden, ${NUM_ATTENTION_HEADS} heads"
echo "Parallelism: TP=${TENSOR_PARALLEL_SIZE}, PP=${PIPELINE_PARALLEL_SIZE}, DP=${DATA_PARALLEL_SIZE}"
echo "Batch: micro=${MICRO_BATCH_SIZE}, global=${GLOBAL_BATCH_SIZE}, accum=${GRADIENT_ACCUMULATION_STEPS}"
echo "============================================"

# ============================================================================
# Launch Training
# ============================================================================

DISTRIBUTED_ARGS="
    --nproc_per_node ${WORLD_SIZE}
    --nnodes 1
    --node_rank 0
    --master_addr localhost
    --master_port 6000
"

GPT_ARGS="
    --num-layers ${NUM_LAYERS}
    --hidden-size ${HIDDEN_SIZE}
    --num-attention-heads ${NUM_ATTENTION_HEADS}
    --seq-length ${SEQ_LENGTH}
    --max-position-embeddings ${SEQ_LENGTH}
    --micro-batch-size ${MICRO_BATCH_SIZE}
    --global-batch-size ${GLOBAL_BATCH_SIZE}
    --lr ${LR}
    --min-lr ${MIN_LR}
    --train-iters ${TRAIN_ITERS}
    --lr-decay-iters ${TRAIN_ITERS}
    --lr-decay-style cosine
    --lr-warmup-iters ${WARMUP_ITERS}
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --clip-grad 1.0
    --bf16
    --use-flash-attn
"

PARALLELISM_ARGS="
    --tensor-model-parallel-size ${TENSOR_PARALLEL_SIZE}
    --pipeline-model-parallel-size ${PIPELINE_PARALLEL_SIZE}
    --sequence-parallel
    --use-distributed-optimizer
"

DATA_ARGS="
    --data-path ${DATA_PATH}
    --vocab-file ${VOCAB_FILE}
    --merge-file ${MERGE_FILE}
    --split 969,30,1
"

OUTPUT_ARGS="
    --log-interval 10
    --save-interval 1000
    --eval-interval 500
    --eval-iters 10
    --save ${CHECKPOINT_PATH}
    --load ${CHECKPOINT_PATH}
    --tensorboard-dir ${TENSORBOARD_PATH}
"

# Run training
torchrun ${DISTRIBUTED_ARGS} \
    ${MEGATRON_PATH}/pretrain_gpt.py \
    ${GPT_ARGS} \
    ${PARALLELISM_ARGS} \
    ${DATA_ARGS} \
    ${OUTPUT_ARGS}
