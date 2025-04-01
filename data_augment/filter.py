## Filter the enormous initial corpus to find those related passages
import json
import os
import argparse
from tqdm import tqdm
from data_augment.retrieve.retriever import BM25
from transformers import AutoTokenizer

def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--sample", type=int, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    return parser

def main():
    parser = get_argument_parser()
    args = parser.parse_args()
    print(args)
    ## loading the retrieve module
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    tokenizer.pad_token = tokenizer.eos_token
    bm25_retriever = BM25(
        tokenizer = tokenizer, 
        index_name = "pubmed", 
        engine = "elasticsearch",
    )

    def bm25_retrieve(question, topk):
        docs_ids, docs = bm25_retriever.retrieve(
            [question], 
            topk=topk, 
            max_query_length=256
        )
        return docs[0].tolist()
    
    ## loading the test set
    data_path = args.dataset
    output_path = args.output_dir
    with open(data_path + 'test.json', 'r') as fin:
        dataset = json.load(fin)

    all_passages = set() 
    for did, data in enumerate(tqdm(dataset, desc="Retrieving passages", ncols=100)):
        passages = bm25_retrieve(data["input"], topk=args.sample)
        all_passages.update(passages)

    output_path = os.path.join(output_path, "pubmed.jsonl")
    os.makedirs(output_path, exist_ok=True)
    with open(output_path, 'w') as fout:
        for idx, passage in enumerate(all_passages):
            json.dump({"id": idx, "text": passage}, fout)
            fout.write('\n')
    print(f"The count of passages is: {len(all_passages)}")
        
    


if __name__ == "__main__":
    main()