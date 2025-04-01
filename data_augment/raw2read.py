import argparse
import os
from tqdm import tqdm
import glob
import json
from tqdm.contrib.concurrent import process_map
from data_augment.read import TYPES, type_map, get_max_workers
from pysbd import Segmenter
import copy
import functools

def search(entry, overall_cls, segmenter, inited_type_map, args):
    # collect text title, which is the 1st sentence in the raw text
    title = entry['text'].split('\n')[0]
    context_wo_title = '\n'.join(entry['text'].split('\n')[1:]) 
    # NOTE: if the context has no title, use the following code:
    # title = None
    # context_wo_title = entry['text']
    
    # truncate the context to meet the max_seq_len
    context_wo_title = overall_cls.truncate_sentence(context_wo_title, max_len=overall_cls.max_seq_len-200)

    # mine task examples from the raw text
    sents = segmenter.segment(context_wo_title)
    overall_entry={'text_id': entry['text_id']}
    for type in TYPES:
        type_cls = inited_type_map[type]
        overall_entry[type], mined_num = type_cls.mine(text=context_wo_title, domain=args.domain_name, title=title, sents=copy.deepcopy(sents)) 
                            # mined_num is the number of mined examples per task type
    
    # create the reading comprehension text
    read_compre, count_dict = overall_cls.format_recomprehension(copy.deepcopy(overall_entry))
                # count_dict includes the number of comprehension tasks per task type
                # you may use `mined_num` and `count_dict` for data analysis

    return {'read_compre': read_compre, 'file_name': entry['file_name'], 'did': entry['did']}

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', 
                        type=str, help='directory of the input raw texts', 
                        default='./data_with_psg/ChemProt/total.json')
    parser.add_argument('--output_dir', 
                        type=str, help='directory of the output reading comprehension texts', 
                        default='./data_with_psg/ChemProt/total_augment.json')
    parser.add_argument("--general_spm_path", 
                        type=str, help='path to the sentencepiece model of the general LLM',
                        default='./data_samples/general.spm')
    parser.add_argument("--domain_spm_path", 
                        type=str, help='path to the sentencepiece model trained from the target domain corpora',
                        default='./data_samples/domain.spm')
    parser.add_argument("--domain_name", 
                        type=str, help='target domain name, e.g., `biomedicine`, `finance` or `law`',
                        default='biomedicine')
                 
    args = parser.parse_args()

    # get max worker for multi-process
    max_workers=get_max_workers()
    print(f'max_workers: {max_workers}')

    # load sentences in the input file
    print('loading raw texts in the input folder...')
    path=args.input_dir
    print(f'paths: {path}')


    raw_texts = []
    file_name = os.path.basename(path)
    with open(path, 'r') as f:
        dataset = json.load(f)
    for did, data in enumerate(dataset):
        for pid, passage in enumerate(data["passages"]):
            text = passage.strip()
            raw_texts.append({'text':text, 'text_id': did*3 + pid, 'file_name': file_name,'did': did})
    
    # init type_map
    inited_type_map = {}
    for type in TYPES:
        type_cls = type_map.cls_dic[type]()
        type_cls.init_spm(args.general_spm_path, args.domain_spm_path)
        inited_type_map[type] = type_cls

    overall_cls = type_map.cls_dic['overall']()
    overall_cls.init_spm(args.general_spm_path, args.domain_spm_path)

    # to chunk text to sentences   
    segmenter = Segmenter(language='en',clean=False)

    partial_search = functools.partial(search, overall_cls=overall_cls, segmenter=segmenter,inited_type_map=inited_type_map, args=args) 
    print('transferring raw texts into reading comprehension...')
    read_compre =list(process_map(partial_search, raw_texts, max_workers=max_workers, chunksize=8192))

    print('saving reading comprehension texts...')
    # sort by text_id to align with the order of raw texts
    for entry in read_compre:
        path = os.path.join(args.output_dir) 
        if 'aug_passages' not in dataset[entry['did']]:  
            dataset[entry['did']]['aug_passages'] = []  
        dataset[entry['did']]['aug_passages'].append(entry['read_compre'])
        with open(path, "w") as fout:
            json.dump(dataset, fout, indent=4)
    
    print(f'saved to {args.output_dir}')
    