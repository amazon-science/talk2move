#!/bin/bash
# export NCCL_BLOCKING_WAIT=1

GPUS_PER_NODE=8
NUM_MACHINES=2
NUM_PROCESSES=$((NUM_MACHINES * GPUS_PER_NODE))
MASTER_PORT=19001
MASTER_ADDR=enter_your_master_addr

RANK=$1

# Launch command using torchrun (native PyTorch distributed)
torchrun \
    --nproc_per_node=${GPUS_PER_NODE} \
    --nnodes=${NUM_MACHINES} \
    --node_rank=${RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    scripts/train_qwenimage_edit.py \
    --config config/grpo.py:counting_qwenimage_edit