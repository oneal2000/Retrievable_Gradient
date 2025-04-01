import os
import gc
import json
import logging
import argparse
import torch
from tqdm import tqdm
from peft import get_peft_model,LoraConfig
from torch.utils.data import Dataset
from transformers import DefaultDataCollator
from typing import Dict, List
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import random

# 设置随机种子
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
writer = SummaryWriter(log_dir="./")

def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--peft", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--sample", type=int, default=-1)

    # Train
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--block_size", type=int, default=3000)

    # Lora
    parser.add_argument("--lora_rank", type=int, default=None)
    parser.add_argument("--lora_alpha", type=int, default=None)

    return parser

class CombinedTrainingData(Dataset):
    ignored_id = -100

    def __init__(self, data, tokenizer, max_length):
        self.max_length = max_length
        self.dataset = []
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

        for item in data:
            augment = item["passages"]
            for passage in augment:
                input_ids = tokenizer.encode(passage, add_special_tokens=False)
                labels = input_ids.copy()
                if len(input_ids) > max_length:
                    input_ids = input_ids[:max_length]
                    labels = labels[:max_length]
                attention_mask = [1] * len(input_ids) + [0] * (max_length - len(input_ids))
                input_ids += [pad_token_id] * (max_length - len(input_ids))
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

def train_combined(args, model, tokenizer, dataset,  output_dir):
    train_data = CombinedTrainingData(dataset, tokenizer, args.block_size)
    train_dataloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.per_device_train_batch_size,
        collate_fn=TrainingDataCollator(tokenizer, model.device),
        shuffle=True,
    )
    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.is_parallelizable = True
    model.model_parallel = True

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(model_parameters, lr=args.learning_rate)

    for epoch in tqdm(range(args.num_train_epochs), desc='Epochs', position=0):
        for step, batch in enumerate(tqdm(train_dataloader, desc=f'Training epoch {epoch+1}', position=1, leave=False)):
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            print("loss:",loss.item(), flush=True)
            loss.backward()
            optimizer.step()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            writer.add_scalar('Loss/train', loss.item(), epoch * len(train_dataloader) + step)


    os.makedirs(output_dir, exist_ok=True)
    model = model.merge_and_unload()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    torch.cuda.empty_cache()
    gc.collect()
    return model


def main():
    parser = get_argument_parser()
    args = parser.parse_args()

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

    output_dir = os.path.join(
        "../gradient",
        args.model.split('/')[-1],
        args.peft,
        f"rank={args.lora_rank}_alpha={args.lora_alpha}" if args.peft == "lora" else "none",
        f"{args.model.split('/')[-1]}_DAPT",
    )
    
    train_combined(args, model, tokenizer, dataset, output_dir)
    writer.close()
if __name__ == "__main__":
    main()
