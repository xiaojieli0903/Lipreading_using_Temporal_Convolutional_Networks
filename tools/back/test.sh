MODEL_JSON_PATH=configs/lrw_resnet18_dctcn_boundary.json
ANNONATION_DIRECTORY=/home/gao2/disk/datasets/lrw/lipread_mp4/
MOUTH_ROIS_DIRECTORY=/home/gao2/disk/datasets/lrw/visual_data
MODEL_PATH=/disk/gao2/work_dirs/train_logs/tcn/lrw_resnet18_dctcn_boundary/2022-09-13T13\:12\:09_bs32/ckpt.best.pth
CUDA_VISIBLE_DEVICES=3 python main.py --modality video \
                                      --config-path $MODEL_JSON_PATH \
                                      --annonation-direc $ANNONATION_DIRECTORY \
                                      --data-dir $MOUTH_ROIS_DIRECTORY \
                                      --model-path $MODEL_PATH \
                                      --test --batch-size 8
