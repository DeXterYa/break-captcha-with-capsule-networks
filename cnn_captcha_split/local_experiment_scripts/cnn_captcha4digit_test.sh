#!/bin/sh

cd ..
python train.py --batch_size 128 --continue_from_epoch -1 --seed 1 \
                --epochs 20 --experiment_name 'cnn_captcha3digit_split_60k_seed0' \
                --device "cuda" --weight_decay_coefficient 0. --random_rotation 'False'\
                --num_train_val 60000 --coord 'False' --random_order 'True'