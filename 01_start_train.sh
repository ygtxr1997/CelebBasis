#!/bin/bash

export PYTHONPATH="${PWD}"

SD_PATH=$1  # /Your/Path/To/sd-v1-4-full-ema.ckpt

# Usage Example:
# ./start_train.sh /gavin/pretrained/sd-v1-4-full-ema.ckpt

python main_id_embed.py --base configs/stable-diffusion/aigc_id.yaml \
               -t \
               --actual_resume "${SD_PATH}" \
               -n celebbasis \
               --gpus 0,