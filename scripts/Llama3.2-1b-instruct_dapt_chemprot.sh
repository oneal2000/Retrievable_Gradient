##!/bin/bash

#finetune
python src/dapt.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --peft lora \
    --dataset data_with_psg/ChemProt/total_augment.json \
    --sample 500 \
    --num_train_epochs 3 \
    --learning_rate 3e-4 \
    --lora_rank 2 \
    --lora_alpha 32 \

# test
python src/inference.py \
    --model gradient/Llama-3.2-1B-Instruct/lora/rank=2_alpha=32/Llama-3.2-1B-Instruct_DAPT \
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

