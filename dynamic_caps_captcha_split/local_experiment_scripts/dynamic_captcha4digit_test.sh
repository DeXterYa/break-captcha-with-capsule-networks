#!/bin/sh

cd ..
python train.py --batch_size 128 --continue_from_epoch -1 --seed 0 \
                --epochs 20 --experiment_name 'dynamic_captcha3digit_split_grad_trainval60k' \
                --device "cuda" --weight_decay_coefficient 0. --num_primary_channel 128\
                --dropout 'False' --test_name '/test' --stride 1 --random_rotation 'True' \
                --num_train_val 60000 --coord 'False' --attention 'False' --random_order 'True'
