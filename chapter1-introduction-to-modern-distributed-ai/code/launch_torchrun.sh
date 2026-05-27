#!/usr/bin/env bash
# Launch Chapter 1 multi-GPU example

# Set OMP_NUM_THREADS to control OpenMP threads per process
# This prevents oversubscription of CPU cores when using multiple GPUs
# Must be set BEFORE torchrun to avoid the warning message
export OMP_NUM_THREADS=4

torchrun --nproc_per_node=2 code/multi_gpu_ddp.py
