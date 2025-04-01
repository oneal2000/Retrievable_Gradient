#!/bin/bash

# 默认参数
default_model="/liuzyai04/thuir/LLM/Meta-Llama-3-8B-Instruct"
default_peft="lora"
default_dataset="/liuzyai04/thuir/yanjunxi/RG/data_with_psg/ChemProt/total_augment.json"
default_sample=500
default_passage_cnt=3
default_batch_size=1
default_epochs=3
default_lr=3e-4
default_block_size=3000
default_lora_rank="2"
default_lora_alpha="32"

# 解析命令行参数
while [[ "$#" -gt 0 ]]; do
  case $1 in
    --model) model="$2"; shift 2;;
    --peft) peft="$2"; shift 2;;
    --dataset) dataset="$2"; shift 2;;
    --sample) sample="$2"; shift 2;;
    --passage_cnt) passage_cnt="$2"; shift 2;;
    --per_device_train_batch_size) batch_size="$2"; shift 2;;
    --num_train_epochs) epochs="$2"; shift 2;;
    --learning_rate) lr="$2"; shift 2;;
    --block_size) block_size="$2"; shift 2;;
    --lora_rank) lora_rank="$2"; shift 2;;
    --lora_alpha) lora_alpha="$2"; shift 2;;
    *) echo "Unknown parameter: $1"; exit 1;;
  esac
done

# 设置默认值
model=${model:-$default_model}
peft=${peft:-$default_peft}
dataset=${dataset:-$default_dataset}
sample=${sample:-$default_sample}
passage_cnt=${passage_cnt:-$default_passage_cnt}
batch_size=${batch_size:-$default_batch_size}
epochs=${epochs:-$default_epochs}
lr=${lr:-$default_lr}
block_size=${block_size:-$default_block_size}
lora_rank=${lora_rank:-$default_lora_rank}
lora_alpha=${lora_alpha:-$default_lora_alpha}

# 运行 encode.py
nohup python src/encode.py \
  --model "$model" \
  --peft "$peft" \
  --dataset "$dataset" \
  --sample "$sample" \
  --passage_cnt "$passage_cnt" \
  --per_device_train_batch_size "$batch_size" \
  --num_train_epochs "$epochs" \
  --learning_rate "$lr" \
  --block_size "$block_size" \
  ${lora_rank:+--lora_rank "$lora_rank"} \
  ${lora_alpha:+--lora_alpha "$lora_alpha"} & 
 