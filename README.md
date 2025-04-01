# Retrievable Gradient

## Overview

**Welcome to the Official Repository of Retrievable Gradient!**

If you find our project insightful or useful, we would greatly appreciate your support—consider giving us a ⭐ to help spread the word!

#### What is Retrievable Gradient?

Retrievable Gradient introduces a novel framework for continual pretraining by preserving and indexing gradients instead of directly modifying model parameters. During inference, only the relevant gradients are retrieved and temporarily applied, enabling Large Language Models (LLMs) to acquire new knowledge while retaining their original capabilities.


#### What’s Included?

- End-to-end implementation of the **Retrievable Gradient** pipeline.
- Preprocessed benchmark datasets for experiments, along with scripts for customization and adding new datasets..

## Reproduce Our Results

This repository provides a step-by-step guide to evaluating **Retrievable Gradient** on various QA datasets. Follow these steps to reproduce our results:

- **[Store the Domain Corpus](#Access_Domain_Corpus)**: Select the appropriate storage granularity based on cost and efficiency.
- **[Run the Data Augmentation Module](#Self-Augmentation)**: Use data augmentation to improve training effect and generate labels to reduce the retrieval noise.
- **[Generate Retrievable Gradient](#Gradient_Generating)**: Train and store the gradients of different documents.
- **[Inference](#Inference)**: Retreive the gradient of relevant documents, apply them to the LLM, and use the updated LLM for inference.


### Install Environment

```
conda create -n rg-env python=3.10.4
conda activate rg-env
pip install -r requirements.txt
```
### Access Domain Corpus
This section provides instructions on preparing and accessing the domain corpus for **Retrievable Gradient**. Depending on available resources, you can choose between fine-grained document-level processing or a more efficient clustered approach.

#### **Scenario 1: Sufficient Resources** – Compute a Gradient for Each Document  

You can follow these steps to process it yourself or extract our `corpus.tar.gz` file using `tar -xvzf corpus.tar.gz`, which will create a folder named `corpus`. 

1. Download the Pubmed corpus from the [Huggingface](https://github.com/facebookresearch/DPR/blob/main/dpr/data/download_data.py#L32) using the following command

```bash
huggingface-cli download yanjx21/PubMed --repo-type dataset --local-dir corpus/rough 
```

2. Use Elasticsearch to index the Wikipedia dump

```bash
wget -O elasticsearch-8.15.0.tar.gz https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.15.0-linux-x86_64.tar.gz  # download Elasticsearch
tar zxvf elasticsearch-8.15.0.tar.gz
rm elasticsearch-8.15.0.tar.gz 
cd elasticsearch-8.15.0
nohup bin/elasticsearch &  # run Elasticsearch in background
cd ..
python prep_elastic.py --data_path corpus/rough/data.jsonl --index_name pubmed  # build index
```

3. Download downstream tasks from the [Huggingface](https://huggingface.co/datasets/AdaptLLM/medicine-tasks) using the following command

```bash
huggingface-cli download AdaptLLM/medicine-tasks --repo-type dataset --local-dir ./
```

4. Retrieve and filter relevant documents for downstream tasks, generate tags and descriptions to prevent the retrieval noise
```bash
python /data_augment/filter.py --dataset medicine-tasks/ChemProt/ --output_dir corpus/filtered/ChemProt/ --sample 400
python /data_augment/tag.py --input_dir corpus/filtered/ChemProt/pubmed.jsonl --output_dir corpus/taged/ChemProt/pubmed.jsonl # you need to change your api-keys here
```

####  **Scenario 2: Limited Resources** – Cluster the Corpus into N Categories
If computational resources are limited, you can cluster the corpus instead of computing a gradient for each document.
- Compute dense embeddings for all documents using a pre-trained model (e.g., sentence-transformers).
- Apply K-means clustering to group documents into N clusters.
- Select a representative document for each cluster, either by choosing the centroid or sampling a few documents from each group.

### Self-Augmentation

You can directly use the pre-augmented data available in the `data_with_psg` directory. 

If you want to perform data augmentation yourself, please process it as follows.

1. Use Elasticsearch to index your downstream dump
```bash
python tag_elastic.py --data_path corpus/taged/ChemProt/pubmed.jsonl --index_name pubmed_chemprot  # build index
```

2.  Retrieve Relevant Passages for the Downstream Dataset
```bash
python data_augment/psg_retrieve.py \
    --dataset corpus/taged/ChemProt/pubmed.jsonl \
    --sample 500 \
    --data_type total \
    --topk 3 \ # the number of passages you retrieve
    --index_name pubmed_chemprot 
```


3. Data Augmentation
```bash
python data_augment/raw2read.py \
    --input_dir ./data_with_psg/ChemProt/total.json \
    --output_dir ./data_with_psg/ChemProt/total_augment.json \
```


If you want to apply data augmentation to a new dataset, the default data format for the augmented data is JSON. Each element in the array should include a 'passages' array, as shown in the example below.

```json
[
    {   
        "passages": "list[string]",
    }
]
```

### Gradient Generating

To generate document-specific gradients (LoRA) for a given dataset, use the `src/encode.py` script with the following parameters:  
| **Parameter**                  | **Description & Options**                                  |
| ------------------------------ | --------------------------------------------------------- |
| `model`                   | Supported models: `llama3.2-1b-instruct`, `qwen2.5-1.5b-instruct`, `llama3-8b-instruct` |
| `dataset`                      | Path to the downstream dataset |
| `sample`                       | Number of questions to process |
| `per_device_train_batch_size`, `num_train_epochs`, `learning_rate` | Training parameters |
| `lora_rank`, `lora_alpha`       | LoRA parameters (dropout is set to 0) |

When running the script for the first time with a specific LoRA configuration, initialize the adapter with the following command:  

```bash
python src/init_adapter.py --lora_rank 2 --lora_alpha 32
```

This command creates an initial random parameter, base_weight, which serves as the starting point for all subsequent training runs.

#### Generated Parameter Storage Structure
All generated parameters are stored in the `gradient` folder. 
The specific location of the parameter files is as follows:

```plain
gradient/
├── {model}/
│   └── rank={lora_rank}_alpha={lora_alpha}/
│       ├── base_weight/
│       └── {dataset}/
│           └── lr={learning_rate}_epoch={num_train_epochs}/
│               └── dataset_name/
│                   └── data_{did}/
│                       └── passage_{pid}/
|                           └── parameters
```

### Inference

To generate parameterized representations of documents (LoRA) for a given dataset, use the `src/inference.py` script with the following parameters:  


| **Parameter**                  | **Description & Options**                                  |
| ------------------------------ | --------------------------------------------------------- |
| `model`                        | Supported models: `llama3.2-1b-instruct`, `qwen2.5-1.5b-instruct`, `llama3-8b-instruct` |
| `dataset`                      | Path to the downstream dataset |
| `sample`                       | Number of questions to process |
| `max_new_tokens`               | Maximum number of tokens to generate |
| `per_device_train_batch_size`, `num_train_epochs`, `learning_rate` | Training parameters |
| `lora_rank`, `lora_alpha`       | LoRA parameters (dropout is set to 0) |
| `inference_method`              | `"wo_param"` (original model) or `"cat_all_one"` (our method) |

#### **Generated Results Storage**  

All generated outputs are stored in the `output` directory with the following structure: 

```plain
output/
├── {model}/
│   └── rank={lora_rank}_alpha={lora_alpha}/
│       ├── base_weight/
│       └── {dataset}/
│           └── lr={learning_rate}_epoch={num_train_epochs}/
│               └── {inference_method}/
│                   ├── config.json
│                   ├── predict.json
│                   └── result.txt
```

### Experimental Results and corresponding Startup Scripts 

The following table presents the experimental results, where each number corresponds to a startup script. Clicking on the number allows you to view the corresponding script:

#### PubMed

| Model                   | Base | DAPT | RG |
|-------------------------|------------|------------|----------|
| llama3.2-1b-instruct   |  [17](scripts/Llama3.2-1b-instruct_base_chemprot.sh)    |  [14.4](scripts/Llama3.2-1b-instruct_dapt_chemprot.sh)    | **[17.8](scripts/Llama3.2-1b-instruct_rg_chemprot.sh)**    |
| qwen2.5-1.5b-instruct  |  [32.4](scripts/Qwen2.5-1.5b-instruct_base_chemprot.sh)    |  **[36.2](scripts/Qwen2.5-1.5b-instruct_dapt_chemprot.sh)**    | [33](scripts/Qwen2.5-1.5b-instruct_rg_chemprot.sh)   |
| llama3-8b-instruct     |  **[43](scripts/Llama3-8B-Instruct_base_chemprot.sh)**    |   [37.6](scripts/Llama3-8B-Instruct_dapt_chemprot.sh)   | [41.6](scripts/Llama3-8B-Instruct_rg_chemprot.sh)   |
