CUDA_VISIBLE_DEVICES=2 python mmg_main.py \
    --seed 0 \
    --weights ./weights/los_7.h5 \
    --task los_7 \
    --mode test \
    --save_features 0