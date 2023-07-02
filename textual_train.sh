export PYTHONPATH="${PWD}"

train_set1=("sup01" "sup03" "sup05" "sup07" "sup09" "sup11" "sup13")
# train_set2=("sup02" "sup04" "sup06" "sup08" "sup10" "sup12" "sup14")
train_set2=("sup09" "sup10")
train_len=${#train_set2[@]}
echo "Training datasets cnt: ${train_len}"

for (( i=0; i<train_len; i++)) do
  folder1=${train_set1[i]}
  folder2=${train_set2[i]}
#  python main.py --base configs/stable-diffusion/v1-finetune.yaml \
#               -t \
#               --actual_resume ./weights/sd-v1-4-full-ema.ckpt \
#               -n textualinversion_v \
#               --gpus 0, \
#               --data_root /gavin/datasets/aigc_id/dataset_for_baseline/"${folder1}" \
#               --init_word person
  python main.py --base configs/stable-diffusion/v1-finetune.yaml \
               -t \
               --actual_resume ./weights/sd-v1-4-full-ema.ckpt \
               -n textualinversion_v \
               --gpus 0, \
               --data_root /gavin/datasets/aigc_id/dataset_for_baseline/"${folder2}" \
               --init_word person
done

