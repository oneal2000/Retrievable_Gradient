##!/bin/bash

# test
python src/inference.py \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
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

