MODEL_JSON_PATH=$1
ANNONATION_DIRECTORY=./datasets/lipread_mp4/
MOUTH_ROIS_DIRECTORY=./datasets/visual_data
MODEL_PATH=$2
CUDA_VISIBLE_DEVICES=1 python main.py --modality video \
                                      --config-path $MODEL_JSON_PATH \
                                      --annonation-direc $ANNONATION_DIRECTORY \
                                      --data-dir $MOUTH_ROIS_DIRECTORY \
                                      --model-path $MODEL_PATH \
                                      --extract-feats --mouth-patch-path /disk/gao2/datasets/lrw/test_list.txt \
				      --output-layer $3
