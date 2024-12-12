MODEL_JSON_PATH=configs/lrw_resnet18_mstcn_frontend3D_predict-future-type1-block5_usememory_mvm.json
ANNONATION_DIRECTORY=/userhome/datasets/lipread_mp4/
MOUTH_ROIS_DIRECTORY=/userhome/datasets/LRW/visual_data
python main.py --modality video \
	--config-path $MODEL_JSON_PATH \
	--annonation-direc $ANNONATION_DIRECTORY \
	--data-dir $MOUTH_ROIS_DIRECTORY \
	--alpha 0 --epochs 10  --predict-loss-weight 0 --loss-average-dim 0 \
	--exp-name _supervised_pretrain --detach-target --cls-loss-weight 1 \
