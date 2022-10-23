MODEL_JSON_PATH=configs/lrw_resnet18_mstcn_frontend3D_block7-skip2_mvm_bycontext_predictall_usehypotheses_use_kd.json
ANNONATION_DIRECTORY=./datasets/lipread_mp4/
MOUTH_ROIS_DIRECTORY=./datasets/visual_data
CUDA_VISIBLE_DEVICES=2 python main.py --modality video \
                                      --config-path $MODEL_JSON_PATH \
                                      --annonation-direc $ANNONATION_DIRECTORY \
                                      --data-dir $MOUTH_ROIS_DIRECTORY \
				      --alpha 0 --epochs 20 \
				      --exp-name _nomixup_20epochs  --interval 50 \
#				      --model-path /disk/gao2/work_dirs/train_logs/tcn/lrw_resnet18_mstcn_frontend3D_block7-skip2_mvm_bycontext_predictall_usehypotheses_nomixup_20epochs/ckpt.pth --init-epoch 0 --allow-size-mismatch
#				      --model-path /disk/gao2/work_dirs/train_logs/tcn/lrw_resnet18_mstcn_frontend3D_block7-skip2_mvm_bycontext_predictall_usehypotheses_use_kd_nomixup_20epochs/ckpt.pth --init-epoch 1
