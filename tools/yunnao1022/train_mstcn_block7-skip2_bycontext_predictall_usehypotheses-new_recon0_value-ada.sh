MODEL_JSON_PATH=configs/lrw_resnet18_mstcn_frontend3D_block7-skip2_mvm_bycontext_predictall_usehypotheses_value-ada.json
ANNONATION_DIRECTORY=./datasets/lipread_mp4/
MOUTH_ROIS_DIRECTORY=./datasets/visual_data
CUDA_VISIBLE_DEVICES=1 python main.py --modality video \
                                      --config-path $MODEL_JSON_PATH \
                                      --annonation-direc $ANNONATION_DIRECTORY \
                                      --data-dir $MOUTH_ROIS_DIRECTORY \
				      --alpha 0 --epochs 20 \
				      --exp-name _nomixup_20epochs_recon0 \
				      --recon-loss-weight 0
