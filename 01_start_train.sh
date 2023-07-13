#!/bin/bash

export PYTHONPATH="${PWD}"

SD_PATH=$1  # /Your/Path/To/sd-v1-4-full-ema.ckpt
CONFIG_FILE=${2:-configs/stable-diffusion/aigc_id.yaml}

# Usage Example:
# ./start_train.sh ./weights/sd-v1-4-full-ema.ckpt

python main_id_embed.py --base "${CONFIG_FILE}" \
               -t \
               --actual_resume "${SD_PATH}" \
               -n celebbasis \
               --gpus 0,