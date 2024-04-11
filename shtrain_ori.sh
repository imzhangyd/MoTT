for lr_m in 10.0 8.0 6.0 4.0 2.0
do 
    CUDA_VISIBLE_DEVICES=1 python train_tracking_eval_tracks10.py \
    --use_tb \
    --train \
    --tracking \
    --eval \
    --lr_mul $lr_m
done