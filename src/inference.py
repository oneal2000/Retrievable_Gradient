import os
import gc
import json
import argparse
import torch
from tqdm import tqdm
from peft import PeftModel
from sklearn.metrics import f1_score
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.utils import pad2sameLen


def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--peft", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--sample", type=int, default=-1)
    parser.add_argument("--num_train_epochs", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--passage_cnt", type=int, default=-1)

    parser.add_argument("--inference_method", type=str, required=True) # also combination_type

    # LoRA
    parser.add_argument("--lora_rank", type=int)
    parser.add_argument("--lora_alpha", type=int)
    parser.add_argument("--density", type=float, default=0.2)
    parser.add_argument("--weight_type", type=str, default="average")

    return parser

LORA_MERGE_METHOD_WITH_DENSITY = ["ties", "ties_svd", "dare_ties", "dare_linear", "dare_ties_svd", "dare_linear_svd", "magnintude_prune", "magnitude_prune_svd"]

def main():
    parser = get_argument_parser()
    args = parser.parse_args()
    print(args)
    if args.peft == "lora":
        assert args.lora_rank and args.lora_alpha, "No Config for LoRA"
        if args.inference_method in LORA_MERGE_METHOD_WITH_DENSITY:
            assert 0 <= args.density <= 1, f"Invalid density {args.density}"
        else:
            args.density = -1
    
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

    # if args.with_cot:
    #     prompt_template.get_fewshot(args.dataset)

    load_adapter_path = os.path.join(
        "../gradient", 
        args.model.split('/')[-1],
        args.peft,
        f"rank={args.lora_rank}_alpha={args.lora_alpha}" if args.peft == "lora" else "none", 
        args.dataset.split('/')[-2], #dataset
        f"lr={args.learning_rate}_epoch={args.num_train_epochs}",
        f"pcnt={args.passage_cnt}",
    )
    output_root_dir = os.path.join(
        "./output",
        args.dataset.split('/')[-2], #dataset
        args.model.split('/')[-1], 
        args.peft,
        f"rank={args.lora_rank}_alpha={args.lora_alpha}" if args.peft == "lora" else "none",
        f"lr={args.learning_rate}_epoch={args.num_train_epochs}",
        f"pcnt={args.passage_cnt}",
        args.inference_method, 
    )
    if args.peft == "lora":
        print("density:", str(args.density))
        output_root_dir = os.path.join(output_root_dir, str(args.density))

    filename = args.dataset.split('/')[-1]
    print(f"### Solving {args.dataset} ###")
    output_dir = os.path.join(output_root_dir, filename)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "config.json"), "w") as fout:
        json.dump(vars(args), fout, indent=4)

    predict_file = os.path.join(output_dir, "predict.json")
    try:
        with open(predict_file, "r") as fin:
            data = json.load(fin)
        ret, start_with =  data, len(data)
    except:
        ret, start_with =  [], 0

    fulldata = dataset[start_with:] if args.sample == -1 else dataset[start_with:args.sample]
    for test_id, data in tqdm(enumerate(fulldata), total=len(fulldata)):
        test_id = test_id + start_with
        assert test_id == len(ret), f"test_id {test_id} != len(ret) {len(ret)}"

        question = data["question"]
        options = data["options"]
        labels = data["index"]
        passages = data["passages"] #without_aug
        if args.passage_cnt != -1:
            passages = passages[:args.passage_cnt]
        
        if args.inference_method == "prompting":
            PROMPT = "You should answer the question by referring to the knowledge provided below and integrating your own knowledge.\n\
            {passages}\n\
            Question: {question}"
            p=""
            for i, passage in enumerate(passages):
                p += f"Passage {i+1}: {passage}\n\n"
            question = PROMPT.format(passages=p, question=question)

        # text_to_instance_choice
        questions = [question] * len(options)
        answers = [' ' + o for o in options]
        input_ids_list = []
        input_atten_mask_list = []
        loss_mask_list = []

        for i in range(len(questions)):
            tokenized_qa = tokenizer.encode_plus(questions[i] + answers[i], truncation=False, return_tensors="pt")
            tokenized_q = tokenizer.encode_plus(questions[i], truncation=False, return_tensors="pt")
            q_mask = tokenized_q.attention_mask.squeeze()
            if len(q_mask.shape) == 0:
                q_mask = torch.tensor([1]).to(q_mask)

            input_ids = tokenized_qa.input_ids.squeeze()
            input_atten_mask = tokenized_qa.attention_mask.squeeze()

            loss_mask = torch.tensor([0] * q_mask.shape[-1] + [1] * (input_ids.shape[-1] - q_mask.shape[-1])).to(input_ids)
            
            input_ids_list.append(input_ids)
            input_atten_mask_list.append(input_atten_mask)
            loss_mask_list.append(loss_mask)


        input_ids = pad2sameLen(input_ids_list, pad_idx=tokenizer.pad_token_id).unsqueeze(0)
        input_atten_mask =  pad2sameLen(input_atten_mask_list, pad_idx=0).unsqueeze(0)
        loss_mask =  pad2sameLen(loss_mask_list, pad_idx=0).unsqueeze(0)

        def get_choice_losses(model, input_ids, input_atten_mask, loss_mask, labels):
            bsz, option_num, seq_len = input_ids.shape
            if len(questions) is not None:
                assert option_num == len(questions)
            model.eval()
            with torch.no_grad():
                output = model(
                    input_ids=input_ids.reshape(bsz * option_num, seq_len),
                    attention_mask=input_atten_mask.reshape(bsz * option_num, seq_len),
                )
            logits = output.logits.reshape(bsz, option_num, seq_len, -1)
            logits = logits[:, :, :-1, :]
            targets = input_ids[:, :, 1:].unsqueeze(-1)
            logit_probs = torch.nn.functional.log_softmax(logits.float(), dim=-1)            
            loss_mask = loss_mask[:, :, 1:]
            loss = - torch.gather(logit_probs, -1, targets).squeeze(-1) * loss_mask
            loss = loss.sum(-1) / loss_mask.sum(-1)
            preds = torch.argmin(loss, dim=-1).tolist()
            normed_loss = torch.nn.functional.normalize(loss, p=1, dim=-1)

            # roughly get pred_probs for all classes
            pred_probs = 2/option_num - normed_loss
            pred_probs = pred_probs.tolist()
            assert len(preds) == 1
            return {"infer_q": question, "infer_a": options, "preds": preds, "labels": labels, "pred_probs": pred_probs}


        if args.inference_method == "wo_param" or args.inference_method == "prompting":
            ret.append(get_choice_losses(model,input_ids, input_atten_mask, loss_mask, labels))

        elif args.peft == "lora":
            if args.inference_method == "merge_and_unload":
                # merge_and_unload
                for pid in range(len(passages)):
                    model = PeftModel.from_pretrained(model, 
                                                        os.path.join(load_adapter_path, filename, f"data_{test_id}", f"passage_{pid}"))
                    model = model.merge_and_unload()
                ret.append(get_choice_losses(model,input_ids, input_atten_mask, loss_mask, labels))
                del model
                torch.cuda.empty_cache()
                gc.collect() 
                model = AutoModelForCausalLM.from_pretrained(
                    args.model, 
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True,
                    device_map="auto", 
                    trust_remote_code=True
                )
                continue
            elif args.inference_method == "total":
                # TO TRAIN, no total now
                model = PeftModel.from_pretrained(
                    model, 
                    os.path.join(load_adapter_path, filename, f"data_{test_id}", "total"),
                    is_trainable = False
                ) 
                ret.append(get_choice_losses(model,input_ids, input_atten_mask, loss_mask, labels))
                model = model.unload()
                torch.cuda.empty_cache()
                gc.collect()
                continue
            elif args.inference_method in ["0", "1"]:
                # my question :这里是为什么要加入这样的设置
                model = PeftModel.from_pretrained(
                    model, 
                    os.path.join(load_adapter_path, filename, f"data_{test_id}", f"passage_{args.inference_method}"),
                    is_trainable = False
                ) 
                ret.append(get_choice_losses(model,input_ids, input_atten_mask, loss_mask, labels))
                model = model.unload()
                torch.cuda.empty_cache()
                gc.collect()
                continue 
            elif args.inference_method == "cat_all_one":
                # my question : Diff from merge and unload?
                model = PeftModel.from_pretrained(
                    model, 
                    os.path.join(load_adapter_path, filename, f"data_{test_id}", "passage_0"),
                    adapter_name = "0", 
                    is_trainable = False
                )
                for pid in range(1, len(passages)):
                    model.load_adapter(os.path.join(load_adapter_path, filename, f"data_{test_id}", f"passage_{pid}"), 
                                    adapter_name = str(pid))
                # merge
                model.add_weighted_adapter(
                    adapters = [str(i) for i in range(len(passages))], 
                    weights = [1 for _ in range(len(passages))],
                    adapter_name = "merge", 
                    combination_type = "cat",
                )
                model.set_adapter("merge")

                ret.append(get_choice_losses(model,input_ids, input_atten_mask, loss_mask, labels))

                model.delete_adapter("merge")
                model = model.unload()
                torch.cuda.empty_cache()
                gc.collect()
                continue

            else:
                model = PeftModel.from_pretrained(
                    model, 
                    os.path.join(load_adapter_path, filename, f"data_{test_id}", "passage_0"),
                    adapter_name = "0", 
                    is_trainable = False
                )
                for pid in range(1, len(passages)):
                    model.load_adapter(os.path.join(load_adapter_path, filename, f"data_{test_id}", f"passage_{pid}"), 
                                    adapter_name = str(pid))
                # get weights
                # my question : 这里的 weight 是怎么来的
                if args.weight_type == "average":
                    weights = [1.0] * len(passages) if "tie" in args.inference_method else \
                            [1.0 / len(passages)] * len(passages)
                else:
                    raise ValueError(f"Invalid weight type {args.weight_type}")

                # merge
                model.add_weighted_adapter(
                    adapters = [str(i) for i in range(len(passages))],
                    weights = weights,
                    adapter_name = "merge", 
                    combination_type = args.inference_method,
                    density = args.density,
                )
                model.set_adapter("merge")

                ret.append(get_choice_losses(model,input_ids, input_atten_mask, loss_mask, labels))

                model = model.unload()
                torch.cuda.empty_cache()
                gc.collect()

        elif args.peft == "ia3":
            # load total param
            if args.inference_method == "total": 
                model = PeftModel.from_pretrained(
                    model, 
                    os.path.join(load_adapter_path, filename, f"data_{test_id}", "total"),
                    is_trainable = False
                ) 
                ret.append(get_choice_losses(model,input_ids, input_atten_mask, loss_mask, labels))
                model = model.unload()
                torch.cuda.empty_cache()
                gc.collect()
            elif args.inference_method == "merge":
                model = PeftModel.from_pretrained(
                    model, 
                    os.path.join(load_adapter_path, filename, f"data_{test_id}", "passage_0"),
                    adapter_name = "0", 
                    is_trainable = False
                )
                for pid in range(1, len(passages)):
                    model.load_adapter(os.path.join(load_adapter_path, filename, f"data_{test_id}", f"passage_{pid}"), 
                                    adapter_name = str(pid))
                # merge
                model.add_weighted_adapter(
                    adapters = [str(i) for i in range(len(passages))],
                    weights = [1.0 / len(passages)] * len(passages),
                    adapter_name = "merge", 
                )
                model.set_adapter("merge")
                ret.append(get_choice_losses(model,input_ids, input_atten_mask, loss_mask, labels))
                model = model.unload()
                torch.cuda.empty_cache()
                gc.collect()
            else:
                raise ValueError(f"Invalid method {args.inference_method} for IA3")

        else:
            raise ValueError(f"Invalid method {args.peft} {args.inference_method}")

        with open(predict_file, "w") as fout:
            json.dump(ret, fout, indent=4)

    ##### Evaluating #####

    true_labels = [item['labels'] for item in ret]
    predicted_labels = [item['preds'][0] for item in ret]  
    F1 = f1_score(y_true=true_labels, y_pred=predicted_labels, average='micro')
    ret_str = f"f1_score: {F1}"
    print(ret_str)
    with open(os.path.join(output_dir, "result.txt"), "w") as fout:
        fout.write(ret_str)


if __name__ == "__main__":
    main()