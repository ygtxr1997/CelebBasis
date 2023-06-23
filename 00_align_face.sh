#!/bin/bash

export PYTHONPATH="${PWD}"

IN_FOLDER=$1  # /Your/Path/To/Images/ori
OUT_FOLDER=$2  # /Your/Path/To/Images/ffhq

# Usage Example:
# bash ./00_align_face.sh /gavin/datasets/aigc_id/dataset_myself/ori /gavin/datasets/aigc_id/dataset_myself/ffhq

# build nms.c for PIPNet
cd ./evaluation/face_align/PIPNet/FaceBoxesV2/utils/ && chmod +x make.sh && bash ./make.sh && cd - || exit

# run align and crop
python ./evaluation/face_align/PIPNet/start_align.py  \
  --in_folder "${IN_FOLDER}"  \
  --out_folder "${OUT_FOLDER}"
