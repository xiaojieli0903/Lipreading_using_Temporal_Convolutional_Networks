#!/usr/bin/env bash

MODEL_JSON_PATH=configs/lrw_resnet18_mstcn.json
ANNONATION_DIRECTORY=/userhome/datasets/lipread_mp4/
MOUTH_ROIS_DIRECTORY=/userhome/datasets/LRW/visual_data

GPUS=$1
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    main.py --modality video --config-path $MODEL_JSON_PATH --annonation-direc $ANNONATION_DIRECTORY --data-dir $MOUTH_ROIS_DIRECTORY \
    --model-path /userhome/train_logs/tcn/lrw_resnet18_mstcn/2022-10-13T05:34:07/ckpt.pth --init-epoch 1
