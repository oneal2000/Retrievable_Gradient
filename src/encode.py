import os
import gc
import json
import logging
import argparse
import torch
from tqdm import tqdm
from peft import PeftModel
from torch.utils.data import Dataset
from transformers import DefaultDataCollator
from typing import Dict, List
from transformers import AutoModelForCausalLM, AutoTokenizer

import numpy as np
import random

# 设置随机种子
seed = 42 
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--peft", type=str, default="lora")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--sample", type=int, default=-1)
    parser.add_argument("--passage_cnt", type=int, default=-1)

    # Train
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--block_size", type=int, default=3000)

    # Lora
    parser.add_argument("--lora_rank", type=int, default=None)
    parser.add_argument("--lora_alpha", type=int, default=None)

    return parser


class TrainingData(Dataset):
    ignored_id = -100

    def __init__(self, prompt_ids, tokenizer, max_length):
        self.max_length = max_length
        self.dataset = []
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        for input_ids in prompt_ids:
            labels = input_ids.copy()
            if len(input_ids) > max_length:
                input_ids = input_ids[:max_length]
                labels = labels[:max_length]
            attention_mask = [1] * len(input_ids) + [0] * (max_length - len(input_ids))
            input_ids += [pad_token_id] * (max_length - len(input_ids))
            # my question: why do we need to pad the labels with -100?
            labels += [self.ignored_id] * (max_length - len(labels))
            self.dataset.append({
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention_mask,
            })
        self.total_len = len(self.dataset)
    
    def __len__(self):
        return self.total_len
    
    def __getitem__(self, idx) -> Dict[str, list]:
        return self.dataset[idx]


class TrainingDataCollator(DefaultDataCollator):
    def __init__(self, tokenizer, device):
        super().__init__()
        self.tokenizer = tokenizer
        self.device = device
    
    def __call__(self, examples: List[Dict[str, list]]) -> Dict[str, torch.Tensor]:
        input_ids, labels, attention_mask = tuple(
            map(lambda x: [example[x] for example in examples], ["input_ids", "labels", "attention_mask"])
        )
        return {
            "input_ids": torch.tensor(input_ids).to(self.device),
            "labels": torch.tensor(labels).to(self.device),
            "attention_mask": torch.tensor(attention_mask).to(self.device),
        }
        

def train(augment, args, model, tokenizer, 
           init_adapter_path,save_path):
    prompt_ids = [tokenizer.encode(augment, add_special_tokens=False)]
    train_data = TrainingData(prompt_ids, tokenizer, args.block_size) #??
    train_dataloader = torch.utils.data.DataLoader( #??
        train_data,
        batch_size=args.per_device_train_batch_size,
        collate_fn=TrainingDataCollator(tokenizer, model.device),
        shuffle=False,
    )
    model = PeftModel.from_pretrained(model, init_adapter_path, is_trainable=True)
    model.is_parallelizable = True
    model.model_parallel = True
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(model_parameters, lr=args.learning_rate)
    for epoch in range(args.num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
    os.makedirs(save_path, exist_ok=True)

    model.save_pretrained(save_path)
    model = model.unload()
    torch.cuda.empty_cache()
    gc.collect()
    return model


def main():
    parser = get_argument_parser()
    args = parser.parse_args()
    print(args)
    with open(args.dataset, 'r') as f:
        dataset = json.load(f)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, 
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        device_map="auto", 
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    #if args.with_cot:
    #    prompt_template.get_fewshot(args.dataset)
    init_adapter_path = os.path.join(
        "./gradient", 
        args.model.split('/')[-1],
        args.peft, 
        f"rank={args.lora_rank}_alpha={args.lora_alpha}" if args.peft == "lora" else "none",
        "base_weight",
    )

    print(f"### Solving {args.dataset} ###")
    output_dir = os.path.join(
        "./gradient", 
        args.model.split('/')[-1],
        args.peft,
        f"rank={args.lora_rank}_alpha={args.lora_alpha}" if args.peft == "lora" else "none", 
        args.dataset.split('/')[-2], #dataset
        f"lr={args.learning_rate}_epoch={args.num_train_epochs}",
        f"pcnt={args.passage_cnt}",
        args.dataset.split('/')[-1], #filename
    )
    os.makedirs(output_dir, exist_ok=True)
    fulldata = dataset if args.sample == -1 else dataset[:args.sample]


    for did, data in tqdm(enumerate(fulldata), total=len(fulldata)):
        augment = data["aug_passages"]
        if args.passage_cnt != -1:
            augment = augment[:args.passage_cnt]

        for pid in range(len(augment)):
            save_path = os.path.join(output_dir, f"data_{did}", f"passage_{pid}")
            if os.path.exists(os.path.join(save_path, "adapter_model.safetensors")):
                continue
            model = train(augment[pid], args, model, tokenizer, init_adapter_path, save_path)

if __name__ == "__main__":
    main()