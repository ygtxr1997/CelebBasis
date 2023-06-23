#!/bin/bash

export PYTHONPATH="${PWD}"

# 1. Input args
SD_PATH=$1  # /Your/Path/To/sd-v1-4-full-ema.ckpt
PROJECT_FOLDER=$2  # project folder name under ./logs/, e.g. training2023-06-20T14-58-59_celebbasis
# ------------------------------------------------------

# 2. Edit or modify the following settings as you need
step_list=(799)  # e.g. (99 199 299 399)
# ------------------------------------------------------

# Usage Example:
# ./03_extract.sh "./weights/sd-v1-4-full-ema.ckpt" "training2023-06-21T14-44-46_celebbasis"


project_folder="${PROJECT_FOLDER}"
project=${project_folder%_celebbasis}
cfg_file="logs/${project_folder}/configs/${project}-project.yaml"
echo "$cfg_file"


for step_id in "${step_list[@]}"; do
  echo embeddings_gs-"$step_id".pt
  python scripts/extract_pt.py  \
          --embedding_path logs/"$project_folder"/checkpoints/embeddings_gs-"$step_id".pt \
          --ckpt "${SD_PATH}" \
          --config "$cfg_file"  \
          --seed 42
done
