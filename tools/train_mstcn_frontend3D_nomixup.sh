MODEL_JSON_PATH=configs/lrw_resnet18_mstcn.json
ANNONATION_DIRECTORY=./datasets/lipread_mp4/
MOUTH_ROIS_DIRECTORY=./datasets/visual_data
CUDA_VISIBLE_DEVICES=3 python main.py --modality video \
                                      --config-path $MODEL_JSON_PATH \
                                      --annonation-direc $ANNONATION_DIRECTORY \
                                      --data-dir $MOUTH_ROIS_DIRECTORY \
				      --alpha 0 \
				      --exp-name _nomixup --model-path /disk/gao2/work_dirs/train_logs/tcn/lrw_resnet18_mstcn_nomixup/2022-10-09T22:18:52/ckpt.pth --init-epoch 1
