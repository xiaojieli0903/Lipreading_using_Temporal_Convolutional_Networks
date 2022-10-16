MODEL_JSON_PATH=configs/lrw_resnet18_mstcn.json
ANNONATION_DIRECTORY=/userhome/datasets/lipread_mp4/
MOUTH_ROIS_DIRECTORY=/userhome/datasets/LRW/visual_data
python main.py --modality video \
    --config-path $MODEL_JSON_PATH \
    --annonation-direc $ANNONATION_DIRECTORY \
    --data-dir $MOUTH_ROIS_DIRECTORY \
    --alpha 0 --epochs 20 \
    --exp-name _nomixup_20epochs_loadpretrain \
    --model-path $1 \
    --init-epoch 0 --allow-size-mismatch
