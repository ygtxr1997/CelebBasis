export PYTHONPATH="${PWD}"

step_list=(
#999
2999
)
prompt_list=(
#"a * person"
"a * person is playing basketball"
"a * person is playing the guitar"
"a * person shakes hands with Barack Obama"
"a * person talks with Elon Musk"
"a * person talks with Anne Hathaway"
##"a sks person shakes hands with a ks person"
)
project_folder="id82023-05-15T17-12-44_textualinversion_v"

#for val1 in "${prompt_list[@]}"; do
#  echo "$val1"
#  for val2 in "${step_list[@]}"; do
#    echo embeddings_gs-"$val2".pt
#    python scripts/stable_txt2img.py --ddim_eta 0.0 \
#        --n_samples 8 \
#        --n_iter 1 \
#        --scale 10.0 \
#        --ddim_steps 50 \
#        --embedding_path logs/"$project_folder"/checkpoints/embeddings_gs-"$val2".pt \
#        --ckpt ./weights/sd-v1-4-full-ema.ckpt \
#        --config "configs/stable-diffusion/v1-inference.yaml" \
#        --outdir outputs/"$project_folder"  \
#        --prompt "$val1"
#  done
#done


for val2 in "${step_list[@]}"; do
  echo embeddings_gs-"$val2".pt
  python scripts/stable_txt2img.py --ddim_eta 0.0 \
      --n_samples 8 \
      --n_iter 1 \
      --scale 10.0 \
      --ddim_steps 50 \
      --embedding_path logs/"$project_folder"/checkpoints/embeddings_gs-"$val2".pt \
      --ckpt ./weights/sd-v1-4-full-ema.ckpt \
      --config "configs/stable-diffusion/v1-inference.yaml" \
      --outdir outputs/"$project_folder"  \
      --from-file /gavin/datasets/aigc_id/dataset_for_baseline/exp_baseline_ti.txt
done
