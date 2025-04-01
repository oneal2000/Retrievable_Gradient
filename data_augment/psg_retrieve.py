# 数据增强大集结版本
# 读入数据、BM25、翻译、QA生成 大一统

import os
import json
import random
import argparse
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer
from data_augment.retrieve.retriever import BM25

random.seed(42)

def read_complete(filepath):
    try:
        with open(filepath, "r") as fin:
            data = json.load(fin)
        return data, len(data)
    except:
        return [], 0

def load_ChemProt():
    data_path = "./medicine-tasks/ChemProt/"
    with open(data_path + 'test.json', 'r') as fin:
        dataset = json.load(fin)
    new_dataset = []
    for did, data in enumerate(dataset):
        val = {
            'qid': data['id'], 
            'question': data['input'],
            'options': data['options'],
            'index': data['gold_index']
        }
        new_dataset.append(val)
    ret = {"total": new_dataset}
    return ret


def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--sample", type=int, required=True)
    parser.add_argument("--data_type", type=str, default=None)
    parser.add_argument("--topk", type=int, default=3) 
    parser.add_argument("--index_name", type=str, default=None, help="index name")
    return parser

def main():
    parser = get_argument_parser()
    args = parser.parse_args()
    print(args)

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    tokenizer.pad_token = tokenizer.eos_token
    bm25_retriever = BM25(
        tokenizer = tokenizer, 
        index_name = args.index_name , 
        keys = {'title': 'title', 'body': 'txt', 'description': 'description', 'tags': 'tags'},
        engine = "elasticsearch",
    )

    def bm25_retrieve(question, topk):
        docs_ids, docs = bm25_retriever.retrieve(
            [question], 
            topk=topk, 
            max_query_length=256
        )
        return docs[0].tolist()

    output_dir = os.path.join("./data_with_psg/", 
                              args.dataset)
    os.makedirs(output_dir, exist_ok=True)

    print("### Loading dataset ###")
    load_func = globals()[f"load_{args.dataset}"]
    load_dataset = load_func()
    dt = args.data_type
    if len(load_dataset) == 1:
        assert dt is None or dt == "total", f"Invalid data_type in dataset {args.dataset}"
        solve_dataset = load_dataset
    else:
        assert dt is None or dt in load_dataset, f"Invalid data_type in dataset {args.dataset}"
        if dt:
            solve_dataset = {dt: load_dataset[dt]}
        else:
            solve_dataset = {k: v for k, v in load_dataset.items() if k != "total"}
        with open(os.path.join(output_dir, "total.json"), "w") as fout:
            json.dump(load_dataset["total"], fout, indent=4)

    for filename, dataset in solve_dataset.items():
        print(f"### Solving {filename} ###")
        output_file = os.path.join(output_dir, filename + ".json")
        before_data, start_from = read_complete(output_file)
        ret = before_data
        dataset = dataset[:args.sample]
        for did, data in tqdm(enumerate(dataset)):
            if did < start_from:
                data = before_data[did]
            else:
                ret.append(data)

            passages = data["passages"] if "passages" in data else []
            if len(passages) < args.topk:
                passages = bm25_retrieve(data["question"], topk=args.topk)
            passages = passages[:args.topk]
            data["passages"] = passages
            ret[did] = data
            with open(output_file, "w") as fout:
                json.dump(ret, fout, indent=4)
        
if __name__ == "__main__":
    main()