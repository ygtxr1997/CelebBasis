#!/bin/bash

export PYTHONPATH="${PWD}"

# 1. Input args
SD_PATH=$1  # /Your/Path/To/sd-v1-4-full-ema.ckpt
PROMPT_FILE=$2  # /Your/Path/To/Prompt_File.txt, e.g. ./infer_images/example_prompt.txt
PROJECT_FOLDER=$3  # project folder name under ./logs/, e.g. training2023-06-20T14-58-59_celebbasis
N_SAMPLES=${4:-8}  # n_samples per text (equals to batch_size), default: 8
# ------------------------------------------------------

# 2. Edit or modify the following settings as you need
step_list=(799)  # default: (799), e.g. (99 199 299 399)
eval_id1_list=(0)  # the id of the 1st person, e.g. (0 1 2 3 4)
eval_id2_list=(1)  # the id of the 2nd person, e.g. (0 1 2 3 4)
# ------------------------------------------------------

# Usage Example:
# ./start_test.sh "./weights/sd-v1-4-full-ema.ckpt" "./infer_images/example_prompt.txt" "training2023-06-21T16-40-29_celebbasis"


#################### BEGIN #######################
project_folder="${PROJECT_FOLDER}"
project=${project_folder%_celebbasis}
cfg_file="logs/${project_folder}/configs/${project}-project.yaml"
echo "$cfg_file"


for (( i = 0 ; i < ${#eval_id1_list[@]} ; i++ )) do
  eval_id1=${eval_id1_list[$i]}
  eval_id2=${eval_id2_list[$i]}

  for step_id in "${step_list[@]}"; do
    echo embeddings_gs-"$step_id".pt
    python scripts/stable_txt2img.py --ddim_eta 0.0 \
            --n_samples "${N_SAMPLES}" \
            --n_iter 1 \
            --scale 10.0 \
            --ddim_steps 50 \
            --embedding_path logs/"$project_folder"/checkpoints/embeddings_gs-"$step_id".pt \
            --ckpt "${SD_PATH}" \
            --from-file "${PROMPT_FILE}" \
            --outdir outputs/"$project_folder"  \
            --config "$cfg_file"  \
            --eval_id1 "${eval_id1}"  \
            --eval_id2 "${eval_id2}"  \
            --img_suffix "${eval_id1}_${eval_id2}_${step_id}"  \
            --seed 42
  done

done
#################### END #######################

#prompt_list=(
#"Elon Musk talks with Mr. Bean"
#"Mr. Bean talks with Elon Musk"
#"Mr. Bean shakes hands with Anne Hathaway"
#"Elon Musk talks with Anne Hathaway"
#"sks and ata sit on the sofa "
#"sks is talking with \`"
#"a man whose face is sks is sitting in front of another man whose face is ks"
#"a man whose face is sks is talking with another man whose face is \`"
#"a man whose face is sks is talking to another man whose face is *"
#"a man whose face is sks is talking to a woman whose face is *"
#"a man whose face is sks is talking to a woman whose face is Anne Hathaway"
#"a man whose face is * is talking to a man whose face is Mr. Bean"
#"sks shakes hands with Mr. Bean"
#"* talks with Anne Hathaway"
#"* shakes hands with Anne Hathaway"
#"Anne Hathaway shakes hands with *"
#"sks shakes hands with *"
#"* shakes hands with sks"
#"ks and sks are playing football"
#"sks and ks are playing football, ks and sks are playing football"
#)

prompt_list=(
#"a photo of sks face"
#"sks is playing basketball"
#"a man having sks face is playing guitar"
#"a man having sks face is playing skateboard"
#"a man having sks face is eating bread in front of the Eiffel Tower"
#"sks shakes hands with Elon Musk"
##"Elon Musk talks with sks"
#"Anne Hathaway talks to sks"
##"sks talks with Anne Hathaway"
#"a man having sks face shakes hands with another person having ks face"
#"a man having ks face shakes hands with another person having sks face"

# **** ID Embedding ****
"a photo of sks person"
#"sks person is playing basketball"
#"a sks person is playing guitar"
#"a sks person is playing skateboard"
#"a sks person is eating bread in front of the Eiffel Tower"
#"sks person shakes hands with Elon Musk"
##"Elon Musk talks with sks"
#"Anne Hathaway talks to sks person"
##"sks talks with Anne Hathaway"
#"a sks person shakes hands with a ks person"
#"a ks person shakes hands with a sks person"
# **** END ****

# ***** TWO CELEBS *****
##"a photo of Elon Musk"      # [49406,320,1125,539,20406,19063,49407]
##"a photo of Anne Hathaway"  # [49406,320,1125,539,4083,31801,49407]
#"a photo of ata tre"
#"ata tre is playing basketball"
#"ata tre is playing guitar"
#"ata tre is playing skateboard"
#"ata tre is eating bread in front of the Eiffel Tower"
#"ata tre talks with Barack Obama"
#"ata tre shakes hands with Robert Downey"
#
#"a photo of sks ks"
#"sks ks is playing basketball"
#"sks ks is playing guitar"
#"sks ks is playing skateboard"
#"sks ks is eating bread in front of the Eiffel Tower"
#"sks ks talks with Anne Hathaway"
#"sks ks shakes hands with Elon Musk"
#
#"ata tre shakes hands with sks ks"
#"sks ks talks to ata tre"
##"Elon Musk shakes hands with Steve Jobs"
##"ata tre shakes hands with Steve Jobs"  # Steve:3803, Jobs:3735
##"Elon Musk shakes hands with Robert Downey"
##"ata tre shakes hands with Robert Downey"  # Robert:3929, Downey:29429
##"Elon Musk shakes hands with Barack Obama"
##"ata tre shakes hands with Barack Obama"  # Barack:22481, Obama:4276
# ***** TWO CELEBS END *****


# ***** MULTI CELEBS *****
#"Michael Jordan"
#"Oprah Winfrey"
# ***** MULTI CELEBS END *****


# ***** WORDS INTERPOLATION *****
#"a photo of Tom Cruise"
#"a picture of Tom Cruise"
# ***** WORDS INTERPOLATION END *****


#"a man with sks face talks to another man with ks face"
#"sks and ks sit on a sofa"
#"ks is talking with ata"
#"ata is talking with tre"
#"tre is talking with ry"
#"ry is talking with bop"
#"bop shakes hands with rn"
#"rn shakes hands with &"
#"& shakes hands with *"
#"* shakes hands with \`"

)

#for step_id in "${prompt_list[@]}"; do
#    echo "$step_id"
#    python scripts/stable_txt2img.py --ddim_eta 0.0 \
#        --n_samples 8 \
#        --n_iter 1 \
#        --scale 10.0 \
#        --ddim_steps 50 \
#        --embedding_path logs/training2023-03-29T11-19-46_textualinversion/checkpoints/embeddings_gs-49999.pt \
#        --ckpt ./weights/sd-v1-4-full-ema.ckpt \
#        --prompt "$step_id"
#done


#python scripts/stable_txt2img.py --ddim_eta 0.0 \
#    --n_samples 8 \
#    --n_iter 1 \
#    --scale 10.0 \
#    --ddim_steps 50 \
#    --embedding_path logs/training2023-04-04T17-14-14_textualinversion/checkpoints/embeddings_gs-9999.pt \
#    --ckpt ./weights/sd-v1-4-full-ema.ckpt \
#    --from-file ./infer_images/personalize.txt


#file_list=("000" "001" "002" "003" "004" "005" "006" "007" "008" "009" "010" "011" "012")
#for a_prompt in "${file_list[@]}"; do
#  echo ./infer_images/wiki_names_"$a_prompt".txt
#  python scripts/stable_txt2img.py --ddim_eta 0.0 \
#      --n_samples 8 \
#      --n_iter 1 \
#      --scale 10.0 \
#      --ddim_steps 50 \
#      --embedding_path logs/training2023-04-17T22-27-34_textualinversion/checkpoints/embeddings_gs-7999.pt \
#      --ckpt ./weights/sd-v1-4-full-ema.ckpt \
#      --from-file ./infer_images/wiki_names_"$a_prompt".txt
#done


#python scripts/stable_txt2img.py --ddim_eta 0.0 \
#    --n_samples 8 \
#    --n_iter 1 \
#    --scale 10.0 \
#    --ddim_steps 50 \
#    --embedding_path logs/training2023-03-30T21-21-35_textualinversion/checkpoints/embeddings_gs-2999.pt \
#    --ckpt ./weights/sd-v1-4-full-ema.ckpt \
#    --from-file ./infer_images/interpolation.txt
