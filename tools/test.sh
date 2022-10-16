MODEL_JSON_PATH=$1
<<<<<<< HEAD
ANNONATION_DIRECTORY=/userhome/datasets/lipread_mp4/
MOUTH_ROIS_DIRECTORY=/userhome/datasets/LRW/visual_data
=======
ANNONATION_DIRECTORY=./datasets/lipread_mp4/
MOUTH_ROIS_DIRECTORY=./datasets/visual_data
>>>>>>> 61d758d54daedb313c978e0c1beb1b2aa33e6dba
MODEL_PATH=$2
CUDA_VISIBLE_DEVICES=1 python main.py --modality video \
                                      --config-path $MODEL_JSON_PATH \
                                      --annonation-direc $ANNONATION_DIRECTORY \
                                      --data-dir $MOUTH_ROIS_DIRECTORY \
                                      --model-path $MODEL_PATH \
                                      --test
