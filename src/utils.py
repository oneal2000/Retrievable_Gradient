import os
import re
import json
import torch
import string
import numpy as np
from collections import Counter
from typing import List, Union
from transformers import AutoModelForCausalLM, AutoTokenizer

# from data import HotpotQA, WikiMultiHopQA, TriviaQA, BaseDataset
from src.prompt_template import get_prompt

ROOT_DIR = "/liuzyai04/thuir/tyc/DAN"
DATA_ROOT_DIR = f"{ROOT_DIR}/data_new/"

class BaseDataset:
    @classmethod
    def normalize_answer(cls, s):
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)
        def white_space_fix(text):
            return ' '.join(text.split())
        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)
        def lower(text):
            return text.lower()
        return white_space_fix(remove_articles(remove_punc(lower(s))))

    @classmethod
    def exact_match_score(
        cls,
        prediction: str,
        ground_truth: Union[str, List[str]],
        ground_truth_id: Union[str, List[str]] = None
    ):
        ground_truths = {ground_truth} if isinstance(ground_truth, str) else set(ground_truth)
        if ground_truth_id and isinstance(ground_truth_id, str):
            ground_truths.update(cls.get_all_alias(ground_truth_id))

        correct = np.max([int(cls.normalize_answer(prediction) == cls.normalize_answer(gt)) for gt in ground_truths])
        return {'correct': correct, 'incorrect': 1 - correct}

    @classmethod
    def f1_score(
        cls,
        prediction: str,
        ground_truth: Union[str, List[str]],
        ground_truth_id: Union[str, List[str]] = None
    ):
        ground_truths = {ground_truth} if isinstance(ground_truth, str) else set(ground_truth)
        if ground_truth_id and isinstance(ground_truth_id, str):
            ground_truths.update(cls.get_all_alias(ground_truth_id))
            
        final_metric = {'f1': 0, 'precision': 0, 'recall': 0}
        for ground_truth in ground_truths:
            normalized_prediction = cls.normalize_answer(prediction)
            normalized_ground_truth = cls.normalize_answer(ground_truth)
            if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
                continue
            if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
                continue
            prediction_tokens = normalized_prediction.split()
            ground_truth_tokens = normalized_ground_truth.split()
            common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                continue

            precision = 1.0 * num_same / len(prediction_tokens)
            recall = 1.0 * num_same / len(ground_truth_tokens)
            f1 = (2 * precision * recall) / (precision + recall)
            for k in ['f1', 'precision', 'recall']:
                final_metric[k] = max(eval(k), final_metric[k])
        return final_metric


def load_data(data_name, data_type, model_name):
    solve_dataset = []
    input_dir = os.path.join(DATA_ROOT_DIR, data_name, model_name)
    files = [f for f in os.listdir(input_dir) if "_fail" not in f]

    if len(files) > 1: # more types in dataset
        # 在这里加入了强行用 data_type == "total" 测试多种类数据集全集的行为
        # 行为是提取整个数据集中的部分构建 total.json
        if data_type == "total":
            all_data = {}
            for filename in files:
                with open(os.path.join(input_dir, filename), "r") as fin:
                    all_data[filename] = json.load(fin)
            print(all_data.keys())
            total_data = []
            idx = {filename: 0 for filename in files}
            for data in all_data["total.json"]:
                typ = data["type"] + ".json"
                if idx[typ] == len(all_data[typ]):
                    break 
                aim_data = all_data[typ][idx[typ]]
                assert aim_data["question"] == data["question"]
                idx[typ] += 1
                total_data.append(aim_data)
            return [["total.json", total_data]]

            # with open(os.path.join(input_dir, "total.json"), "r") as fin:
            #     solve_dataset.append(("total.json", json.load(fin)))
            # return solve_dataset

        for filename in files:
            if filename != "total.json":
                with open(os.path.join(input_dir, filename), "r") as fin:
                    solve_dataset.append((filename, json.load(fin)))
        if data_type is None:
            return solve_dataset
        else:
            data_type = data_type + ".json"
            if data_type not in [v[0] for v in solve_dataset]:
                raise ValueError(f"Invalid {data_type} in Dataset {data_name}")
            tmp = []
            for filename, dataset in solve_dataset:
                if filename == data_type:
                    tmp.append((filename, dataset))
            return tmp
    else:
        with open(os.path.join(input_dir, "total.json"), "r") as fin:
            solve_dataset.append(("total.json", json.load(fin)))
        return solve_dataset
    

def get_model_path(model_name):
    BASE_DIR = "/liuzyai04/thuir/tyc/LLM/"
    if model_name == "llama3.1-8b-instruct":
        return BASE_DIR + "Meta-Llama-3.1-8B-Instruct"
    elif model_name == "llama3-8b-instruct": 
        return "/liuzyai04/thuir/wcy/LLM/Meta-Llama-3-8B-Instruct"
    elif model_name == "llama2-7b-chat":
        return "/liuzyai04/thuir/wcy/LLM/llama2-7b-chat"
    elif model_name == "qwen2.5-0.5b-instruct":
        return BASE_DIR + "Qwen2.5-0.5B-Instruct" 
    elif model_name == "qwen2.5-1.5b-instruct":
        return BASE_DIR + "Qwen2.5-1.5B-Instruct"  
    elif model_name == "mistral-7b-instruct":
        return BASE_DIR + "Mistral-7B-Instruct-v0.3"
    elif model_name == "llama3.2-1b-instruct":
        return BASE_DIR + "Llama-3.2-1B-Instruct"
    elif model_name == "qwen2-7b-instruct":
        return "/liuzyai04/thuir/wcy/LLM/qwen2-7B-instruct"
    else:
        return model_name


def get_model(model_name, max_new_tokens=20):
    model_path = get_model_path(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        device_map="auto", 
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    generation_config = dict(
        num_beams=1, 
        do_sample=False,
        max_new_tokens=max_new_tokens,
        return_dict_in_generate=True,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
    )
    return model, tokenizer, generation_config


def read_complete(filepath):
    try:
        with open(filepath, "r") as fin:
            data = json.load(fin)
        return data, len(data)
    except:
        return [], 0
    

    
def evaluate(pred, ground_truth, with_cot=False):
    if not with_cot:
        pred = pred.strip()
        stop_list = [".", "\n", ","]
        for stop in stop_list:
            end_pos = pred.find(stop)
            if end_pos != -1:
                pred = pred[:end_pos].strip()
    else:
        if "the answer is" in pred:
            pred = pred[pred.find("the answer is") + len("the answer is"):]
        pred = pred.strip()
        stop_list = [".", "\n", ","]
        for stop in stop_list:
            end_pos = pred.find(stop)
            if end_pos != -1:
                pred = pred[:end_pos].strip() 

    em = BaseDataset.exact_match_score(
        prediction=pred,
        ground_truth=ground_truth,
    )["correct"]
    f1_score = BaseDataset.f1_score(
        prediction=pred,
        ground_truth=ground_truth,
    )
    f1, prec, recall = f1_score["f1"], f1_score["precision"], f1_score["recall"]
    return {
        "eval_predict": pred,
        "em": str(em),
        "f1": str(f1),
        "prec": str(prec),
        "recall": str(recall),
    }


def predict(model, tokenizer, generation_config, question, with_cot, passages = None):
    model.eval()
    input_ids = get_prompt(
        tokenizer, 
        question, 
        passages = passages, 
        with_cot = with_cot)
    input_len = len(input_ids)
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(model.device)
    with torch.no_grad():
        output = model.generate(
            input_ids, 
            attention_mask = torch.ones(input_ids.shape).to(model.device),
            **generation_config)
    output = output.sequences[0][input_len:]
    text = tokenizer.decode(output, skip_special_tokens=True)
    return text

def pad2sameLen(
    values,
    pad_idx=0,
    left_pad=False
):
    """Convert a list of 1d tensors into a padded 2d tensor.
    ensuring same lengths
    """
    size = max(v.shape[-1] for v in values)
    if left_pad:
        res=torch.stack([torch.nn.functional.pad(v,(size-v.shape[-1],0),value=pad_idx) for v in values])
    else:
        res=torch.stack([torch.nn.functional.pad(v,(0,size-v.shape[-1]),value=pad_idx) for v in values])
    return res