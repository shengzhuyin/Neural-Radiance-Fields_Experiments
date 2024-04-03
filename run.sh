#!/bin/bash

# export CUDA_VISIBLE_DEVICES=0
# torchrun --standalone --nproc_per_node=gpu run_nerf.py --config configs/photoshapes/config.txt --skip_loading
# python3 -m torch.distributed.run --standalone --nproc_per_node=gpu run_nerf.py --config configs/photoshapes/config.txt --skip_loading --i_testset 5000 --load_from /scratch/users/akshat7/cv/editnerf/logs/photoshapes/001000.tar 1>training.log 2>&1

# python3 -m torch.distributed.run --standalone --nproc_per_node=gpu run_nerf.py --config configs/plane_dataset/config.txt --skip_loading 1>plane_training.log 2>&1

# python3 -u ./run_nerf.py --config configs/plane_dataset/config.txt --skip_loading 1>plane_training.log 2>&1
python3 -u ./run_nerf.py --config configs/photoshapes/config.txt --load_it 910000 1>photoshapes_training.log 2>&1

# torch version 1.13.1+cu116
# pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116 --upgrade
# pip install urllib3==1.26.6 --upgrade
