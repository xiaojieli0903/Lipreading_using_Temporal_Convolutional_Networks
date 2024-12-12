MODEL_JSON_PATH=configs/lrw_resnet18_mstcn_frontend3D_predict-future.json
ANNONATION_DIRECTORY=./datasets/lipread_mp4/
MOUTH_ROIS_DIRECTORY=./datasets/visual_data
CUDA_VISIBLE_DEVICES=2 python main.py --modality video \
                                      --config-path $MODEL_JSON_PATH \
                                      --annonation-direc $ANNONATION_DIRECTORY \
                                      --data-dir $MOUTH_ROIS_DIRECTORY \
				      --alpha 0 \
				      --exp-name _nomixup
#				      --model-path train_logs/tcn/lrw_resnet18_mstcn/2022-09-24T11:43:09/ckpt.pth \
#				      --init-epoch 1
