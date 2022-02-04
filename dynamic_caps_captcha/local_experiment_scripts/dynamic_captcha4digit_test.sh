#!/bin/sh

cd ..
python train.py --batch_size 128 --continue_from_epoch -1 --seed 0 \
                --epochs 20 --experiment_name 'dynamic_captcha4digit_trainval60k' \
                --device "cuda" --weight_decay_coefficient 0.\
                --dropout 'True' --test_name '/test' --stride 1\
                --num_train_val 60000 --coord 'False' --attention 'False'
