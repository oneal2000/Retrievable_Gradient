#!/bin/bash

# Step 1: Initialize the LoRA Adapter
# This script sets up the initial adapter weights for the specified model.
python src/init_adapter.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --peft lora \
    --lora_rank 2 \
    --lora_alpha 32

# Step 2: Generate Gradients
# This step encodes document-specific gradients using LoRA-based fine-tuning.
python src/encode.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --peft lora \
    --dataset data_with_psg/ChemProt/total_augment.json \
    --sample 500 \ 
    --per_device_train_batch_size 1 \  
    --num_train_epochs 3 \ 
    --learning_rate 3e-4 \  
    --block_size 3000 \  # Maximum input sequence length
    --passage_cnt 3 \  
    --lora_rank 2 \
    --lora_alpha 32

# Step 3: Run Inference
# This script retrieves and applies stored gradients to perform inference.
python src/inference.py \
    --model meta-llama/Llama-3.2-1B-Instruct \
    --peft lora \
    --max_new_tokens 20 \ 
    --dataset data_with_psg/ChemProt/total_augment.json \
    --sample 500 \  
    --num_train_epochs 3 \
    --learning_rate 3e-4 \
    --passage_cnt 3 \
    --inference_method "cat_all_one" \ 
    --lora_rank 2 \
    --lora_alpha 32 \
    --density 0.2  # Density parameter for inference
