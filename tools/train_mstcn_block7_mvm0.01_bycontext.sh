MODEL_JSON_PATH=configs/lrw_resnet18_mstcn_frontend3D_predict-future-type1-block7_usememory_mvm_fixmemory_bycontext.json
python main.py --modality video \
                                      --config-path $MODEL_JSON_PATH \
				      --alpha 0 --epochs 20  --predict-loss-weight 1 --loss-average-dim 0 \
				      --exp-name _nomixup_20epochs_lw1_avg0_cosine_addloss_contrastive0.01 --detach-target \
				      --predict-loss-type 'cosine' --add-memory-loss --contrastive-loss-weight 0.01

