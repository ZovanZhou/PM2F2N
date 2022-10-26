CUDA_VISIBLE_DEVICES=2 python mmg_main.py \
    --seed 0 \
    --epoch 300 \
    --patience 20 \
    --weights ./weights/mort_icu.h5 \
    --task mort_icu \
    --mode train \
    --save_features 0