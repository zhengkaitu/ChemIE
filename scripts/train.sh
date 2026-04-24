#!/bin/bash

NUM_NODES=1
NUM_GPUS_PER_NODE=1
NODE_RANK=0
MASTER_PORT=$(shuf -n 1 -i 10000-65535)

ACCUM_STEP=1

LOAD_PATH=ckpts/swin_base_char_aux_1m680k.pth
mkdir -p "$SAVE_PATH"

set -x

python train.py \
    --data_path data \
    --dataset_dir_train "$DATASET_DIR_TRAIN" \
    --train_file "$TRAIN_FILE" \
    --val_file "$VAL_FILE" \
    --vocab_file molscribe/vocab/vocab_chars.json \
    --coords_file train_file \
    --formats chartok_coords,edges \
    --coord_bins 64 --sep_xy \
    --input_size 384 \
    --encoder swin_base \
    --decoder transformer \
    --num_bond_type 11 \
    --load_path "$LOAD_PATH" \
    --encoder_lr "$LR" \
    --decoder_lr "$LR" \
    --save_path "$SAVE_PATH" --save_mode last \
    --label_smoothing 0.1 \
    --epochs "$EPOCH" \
    --batch_size $((BATCH_SIZE / NUM_GPUS_PER_NODE / ACCUM_STEP)) \
    --gradient_accumulation_steps $ACCUM_STEP \
    --use_checkpoint \
    --warmup 0.02 \
    --print_freq 200 \
    --do_train \
    --do_val \
    --fp16 --backend nccl 2>&1
