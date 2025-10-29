# End2End
pretrain_dir="models/qwen2.5-full/sft/toolscaler_ret-2@2"
checkpoint_dir="models/qwen2.5-full/sft/toolscaler_dfs-2@2"
flash_attention="True"
run_name="gist2@2-end2end"
datasets="/data/toolscaler_hierarchical_2@2_G123_dfs.json"
dataset_nums="10000000"
max_length="6144"
batch_size="1"
lr="4e-5"
accumulation_steps="64"
epochs="1"
add_virtual_tokens="False"
template="qwen2.5"
save_strategy="steps"
save_steps="500"
zero="z3_offload"

chat="True"

cmd="deepspeed --include=localhost:0,1,2,3 --master_port 25025 train.py \
  --model_name_or_path ${pretrain_dir} \
  --add_virtual_tokens ${add_virtual_tokens} \
  --flash_attention ${flash_attention} \
  --deepspeed src/configs/ds_${zero}_config.json \
  --chat ${chat} \
  --template ${template} \
  --architecture causal \
  --output_dir ${checkpoint_dir} \
  --save_strategy ${save_strategy} \
  --save_steps ${save_steps} \
  --gather_weights True \
  --learning_rate ${lr} \
  --warmup_ratio 0.03 \
  --datasets ${datasets} \
  --dataset_nums ${dataset_nums} \
  --per_device_train_batch_size ${batch_size} \
  --gradient_accumulation_steps ${accumulation_steps} \
  --max_length ${max_length} \
  --num_train_epochs ${epochs} \
  --gradient_checkpointing False \
  --bf16 True \
  --logging_steps 1 \
  --report_to wandb \
  --run_name ${run_name}"

echo $cmd
eval $cmd