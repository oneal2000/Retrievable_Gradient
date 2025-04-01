import json
from tqdm import tqdm
import os
from openai import OpenAI
import itertools
import argparse
from concurrent.futures import ThreadPoolExecutor
import logging

API_KEYS = [
    "sk-xxxxxx",
    "sk-xxxxxx"
]


tag_instruction = """
Given the following document text:\n\
{passage}\n\n\
Please accomplish the following tasks clearly:\n\
1. Write a concise, clear description (1-2 sentences) summarizing the core knowledge or topics covered in this document.\n\
2. Generate 5 keyphrases (tags) that accurately represent the main content of the document.\n\
Output your result in this format strictly:\n\
{{\n\
    \"description\": \"This is an example of description.\",\n\
    \"tags\": [tag1, tag2, tag3, tag4, tag5] \n\
}}\n\n\
"""
def gpt_generate(content, api_key):
    client = OpenAI(api_key=api_key, base_url="https://api.aihao123.cn/luomacode-api/open-api/v1")
    completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": content}
        ],
        model="gpt-3.5-turbo"
    )
    return completion

def check_output(gen):
    if (
        "description" in gen and isinstance(gen["description"], str) and
        "tags" in gen and isinstance(gen["tags"], list) and len(gen["tags"]) == 5 and
        all(isinstance(tag, str) for tag in gen["tags"])
    ):
        return True, gen["description"], gen["tags"]
    else:
        return False, None, None

def process_data(data, api_key):
    passage = data['text']
    prompt = tag_instruction.format(passage=passage)
    max_attempts = 30 
    attempts = 0
    
    while attempts < max_attempts:
        try:
            output = gpt_generate(prompt, api_key).choices[0].message.content
            gen = json.loads(output)
            ret, des, tags = check_output(gen)
            if ret:
                return {"id": data['id'], "text": data['text'], "description": des, "tags": tags}
        except (json.JSONDecodeError, KeyError):
            attempts += 1 
    return {"id": data['id'], "text": data['text'], "error": "Failed after 30 attempts"}

def load_processed_ids(output_path):
    processed_ids = set()
    if os.path.exists(output_path):
        with open(output_path, 'r') as fin:
            for line in fin:
                try:
                    record = json.loads(line)
                    processed_ids.add(record['id'])
                except json.JSONDecodeError:
                    continue
    return processed_ids

def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    return parser

def main():
    parser = get_argument_parser()
    args = parser.parse_args()
    print(args)

    filtered_corpus_path = args.input_dir
    output_path = args.output_dir
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    dataset = []
    with open(filtered_corpus_path, 'r') as fin:
        for line in fin:
            dataset.append(json.loads(line))

    processed_ids = load_processed_ids(output_path)
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        key_cycle = itertools.cycle(API_KEYS)
        futures = [
            executor.submit(process_data, data, next(key_cycle)) for data in dataset if data['id'] not in processed_ids
        ]

        with open(output_path, "a") as fout:
            for future in tqdm(futures, total=len(futures), desc="Processing Data"):
                try:
                    result = future.result(timeout=30)
                    if result:
                        json.dump(result, fout)
                        fout.write('\n')
                except Exception as e:
                    logging.error(f"Error processing: {e}")

if __name__ == "__main__":
    main()