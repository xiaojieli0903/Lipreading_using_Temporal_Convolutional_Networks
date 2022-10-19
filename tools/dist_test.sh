MODEL_JSON_PATH=$1
ANNONATION_DIRECTORY=/userhome/datasets/lipread_mp4/
MOUTH_ROIS_DIRECTORY=/userhome/datasets/LRW/visual_data
MODEL_PATH=$2

GPUS=1
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}


YTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    main.py --modality video \
    --config-path $MODEL_JSON_PATH \
    --annonation-direc $ANNONATION_DIRECTORY \
    --data-dir $MOUTH_ROIS_DIRECTORY \
    --model-path $MODEL_PATH \
    --test
