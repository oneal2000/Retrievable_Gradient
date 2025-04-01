# target_modules=['down_proj', 'gate_proj', 'up_proj']

import os
import json
import argparse
import torch
from peft import TaskType, get_peft_model, LoraConfig, IA3Config
from transformers import AutoModelForCausalLM

from utils import get_model_path


def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--peft", type=str, required=True)

    # LoRA Config
    parser.add_argument("--lora_rank", type=int, default=None)
    parser.add_argument("--lora_alpha", type=int, default=None)
    # lora_dropout = 0

    return parser

if __name__ == "__main__":
    parser = get_argument_parser()
    args = parser.parse_args()
    print(args)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        # device_map="auto", 
        trust_remote_code=True
    )

    if args.peft == "lora": 
        assert args.lora_rank and args.lora_alpha
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            target_modules=['down_proj', 'gate_proj', 'up_proj'],
            inference_mode=False,
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0,
        )
        output_name = f"rank={args.lora_rank}_alpha={args.lora_alpha}"
    elif args.peft == "ia3": 
        peft_config = IA3Config(
            task_type=TaskType.CAUSAL_LM,
            target_modules=['down_proj', 'gate_proj', 'up_proj'],
        )
        output_name = "none"
    else:
        raise ValueError(f"Invalid Peft {args.peft}")
    
    model = get_peft_model(model, peft_config)
    model.is_parallelizable = True
    model.model_parallel = True
    save_path = os.path.join(
        "./gradient", 
        args.model.split('/')[-1],
        args.peft, 
        f"rank={args.lora_rank}_alpha={args.lora_alpha}" if args.peft == "lora" else "none",
        "base_weight",
    )
    print(f'Saving to {save_path}')
    os.makedirs(save_path, exist_ok=True)
    model.save_pretrained(save_path)