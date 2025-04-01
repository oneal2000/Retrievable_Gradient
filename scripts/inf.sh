#!/bin/bash

# 设置默认参数
MODEL="/liuzyai04/thuir/yanjunxi/gradient/Meta-Llama-3-8B-Instruct/lora/rank=2_alpha=32/Meta-Llama-3-8B-Instruct_DAPT"
PEFT="lora"
MAX_NEW_TOKENS=20
DATASET="/liuzyai04/thuir/yanjunxi/RG/data_with_psg/ChemProt/total_augment.json"
SAMPLE=500
NUM_TRAIN_EPOCHS=3
LEARNING_RATE=3e-4
PASSAGE_CNT=3
INFERENCE_METHOD="wo_param"

# LoRA 参数
LORA_RANK=2
LORA_ALPHA=32
DENSITY=0.2

# 运行 Python 推理脚本
nohup python src/inference.py \
    --model $MODEL \
    --peft $PEFT \
    --max_new_tokens $MAX_NEW_TOKENS \
    --dataset $DATASET \
    --sample $SAMPLE \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --passage_cnt $PASSAGE_CNT \
    --inference_method $INFERENCE_METHOD \
    --lora_rank $LORA_RANK \
    --lora_alpha $LORA_ALPHA \
    --density $DENSITY &


