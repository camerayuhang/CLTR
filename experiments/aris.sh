#!/bin/bash

python -m train_distributed --gpu_id '0' \
--gray_aug --gray_p 0.1 --scale_aug --scale_type 1 --scale_p 0.3 --epochs 1500 --lr_step 600 --lr 5e-5 \
--batch_size 8 --num_patch 1 --threshold 0.35 --test_per_epoch 1 --num_queries 500 \
--dataset aris --crop_size 320 --local_rank 0 --backbone resnet18 --pre None --test_patch --save