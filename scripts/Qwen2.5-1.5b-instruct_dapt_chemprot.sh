##!/bin/bash

#finetune
python src/dapt.py \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --peft lora \
    --dataset data_with_psg/ChemProt/total_augment.json \
    --sample 500 \
    --num_train_epochs 3 \
    --learning_rate 3e-4 \
    --lora_rank 2 \
    --lora_alpha 32 \
# test
python src/inference.py \
    --model gradient/Qwen2.5-1.5B-Instruct/lora/rank=2_alpha=32/Qwen2.5-1.5B-Instruct_DAPT \
    --peft lora \
    --max_new_tokens 20 \
    --dataset data_with_psg/ChemProt/total_augment.json \
    --sample 500 \
    --num_train_epochs 3 \
    --learning_rate 3e-4 \
    --passage_cnt 3 \
    --inference_method "wo_param" \
    --lora_rank 2 \
    --lora_alpha 32 \
    --density 0.2

