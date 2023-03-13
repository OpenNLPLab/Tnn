#!/usr/bin/env bash
export NCCL_LL_THRESHOLD=0

ARCH=$3
GPUS=$1
batch_size=$2
PORT=$(( $RANDOM + 2000 ))
export MASTER_PORT=${MASTER_PORT:-$PORT}
LR=$5
DATASET=$6

OUTPUT_DIR=./checkpoints/$4
RESUME=./checkpoints/$4/checkpoint_best.pth

PROG=path_to_im/classification/main.py
DATA=path_to_data

python $PROG \
    --data-set $DATASET --data-path $DATA \
    --batch-size $batch_size --dist-eval --output_dir $OUTPUT_DIR \
    --resume $RESUME --model $ARCH --epochs 300 --fp32-resume --lr $LR \
    --weight-decay $7 \
    --warmup-epochs $8 \
    --clip-grad ${9} \
    --broadcast_buffers
