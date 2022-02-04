#!/bin/sh

cd ..
python train.py --batch_size 128 --continue_from_epoch -1 --seed 0 \
                --epochs 20 --experiment_name 'cnn_captcha4digit_coord_newdata_4k_seed0' \
                --device "cuda" --weight_decay_coefficient 0.\
                --num_train_val 40000 --coord 'False'