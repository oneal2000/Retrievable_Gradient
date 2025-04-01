current_dataset = None
fewshot = None
fewshot_path = "/liuzyai04/thuir/tyc/DAN/src/fewshot/"

SYSTEM_PROMPT = None

USER_PROMPT = "You should answer the question by referring to the knowledge provided below and integrating your own knowledge.\n\
{passages}\n\n\
Question: {question}"

USER_PROMPT_WITH_COT = "You should reference the knowledge provided below and combine it with your own knowledge to answer the question. Please follow the format of the example I provided above.\n\
Here are some examples about how to answer the questions.\n\
{fewshot}\
Here are some reference.\n\
{passages}\n\n\
Let's think step by step. Answer the questions in the same format as above.\n\
Question: {question}"

USER_PROMPT_WITH_QA = "You should answer the question by referring to the knowledge provided below and integrating your own knowledge.\n\
{passages}\n\n\
Here are some questions and answers about the knowledge.\n\
{qa}\n\
You need to answer the question: {question}"

USER_PROMPT_WITH_COT_AND_QA = "You should reference the knowledge provided below and combine it with your own knowledge to answer the question. Please follow the format of the example I provided above.\n\
Here are some examples about how to answer the questions.\n\
{fewshot}\
Here are some reference.\n\
{passages}\n\n\
Here are some questions and answers about the knowledge.\n\
{qa}\n\
Let's think step by step. Answer the questions in the same format as above.\n\
Question: {question}"

ASSISTANT_PROMPT = "The answer is {answer}"
ASSISTANT_PROMPT_WITH_COT = "Answer: {answer}"

def _get_prompt(question, passages=None, answer=None):
    question = question.strip()
    if not question.endswith('?'):
        question = question.strip() + '?'
    elif question.endswith(' ?'):
        question = (question[:-1]).strip() + '?'
     
    if passages and not isinstance(passages, list):
        passages = [passages]
    
    if answer is None:
        answer = ""
    else:
        answer = answer.strip()
        if not answer.endswith('.'):
            answer += "."
    return question, passages, answer


def get_fewshot(dataset):
    import json
    global current_dataset
    global fewshot
    # assert current_dataset is None
    if dataset.endswith("_golden"):
        dataset = dataset.split("_golden")[0]
    current_dataset = dataset
    with open(fewshot_path + dataset + ".json", "r") as fin:
        tmp = json.load(fin)
    fewshot = ""
    for data in tmp:
        q = data["question"]
        a = data["answer"]
        fewshot += f"Question: {q}\nAnswer: {a}\n\n"

# ------------------------------------ Llama 2 Chat Completion ------------------------------------

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]

def llama2_chat_completion(tokenizer, dialog):
    from typing import List
    unsafe_requests = []
    unsafe_requests.append(
        any([tag in msg["content"] for tag in SPECIAL_TAGS for msg in dialog])
    )
    if dialog[0]["role"] == "system":
        dialog = [
            {
                "role": dialog[1]["role"],
                "content": B_SYS
                + dialog[0]["content"]
                + E_SYS
                + dialog[1]["content"],
            }
        ] + dialog[2:]
    assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
        [msg["role"] == "assistant" for msg in dialog[1::2]]
    ), (
        "model only supports 'system', 'user' and 'assistant' roles, "
        "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
    )
    dialog_tokens: List[int] = sum(
        [
            tokenizer.encode(
                f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} ",
            ) + [2]
            for prompt, answer in zip(
                dialog[::2],
                dialog[1::2],
            )
        ],
        [],
    )
    assert (
        dialog[-1]["role"] == "user"
    ), f"Last message must be from user, got {dialog[-1]['role']}"
    dialog_tokens += tokenizer.encode(
        f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}",
    )
    return dialog_tokens

# ------------------------------------ Llama 2 Chat Completion ------------------------------------


def get_prompt(tokenizer, question, passages=None, answer=None, with_cot=False):
    '''
    会返回 tokenizer.encode 的结果，整个 prompt 会对应一个 list[int]
    接下来需要 torch.tensor(return_value).unsqueeze(0).to(model.device)
    '''
    question, passages, answer = _get_prompt(question, passages, answer)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] if SYSTEM_PROMPT else []

    contexts = ""
    if passages:
        for pid, psg in enumerate(passages):
            contexts += f"Passage {pid+1}: {psg}\n"
    if not with_cot:
        user_content = USER_PROMPT.format(question=question, passages=contexts)
        assistant_content = ASSISTANT_PROMPT.format(answer=answer)
    else:
        assert fewshot is not None
        user_content = USER_PROMPT_WITH_COT.format(question=question, passages=contexts, fewshot=fewshot)
        assistant_content = ASSISTANT_PROMPT_WITH_COT.format(answer=answer)

    messages.append({
        "role": "user",
        "content": user_content,
    })

    if 'llama2' in tokenizer.name_or_path:
        inputs = llama2_chat_completion(tokenizer, messages)
        inputs += tokenizer.encode(' ' + assistant_content, add_special_tokens=False)
    else:
        inputs = tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True)
        inputs += tokenizer.encode(assistant_content, add_special_tokens=False)
    return inputs


def get_prompt_with_qa(tokenizer, question, passages, generate_qa, answer=None, with_cot=False):
    '''
    会返回 tokenizer.encode 的结果，整个 prompt 会对应一个 list[int]
    接下来需要 torch.tensor(return_value).unsqueeze(0).to(model.device)
    '''
    qa_str = ""
    for d in generate_qa:
        for qa in d['qa']:
            qq = qa['question']
            aa = qa['answer'] if not with_cot else qa['full_answer']
            if not aa.endswith('.'):
                aa += '.'
            qa_str += f'Question: {qq}\nAnswer: {aa}\n'

    question, passages, answer = _get_prompt(question, passages, answer)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}] if SYSTEM_PROMPT else []

    contexts = ""
    if passages:
        for pid, psg in enumerate(passages):
            contexts += f"Passage {pid+1}: {psg}\n"
    if not with_cot:
        user_content = USER_PROMPT_WITH_QA.format(question=question, passages=contexts, qa=qa_str)
        assistant_content = ASSISTANT_PROMPT.format(answer=answer)
    else:
        assert fewshot is not None
        user_content = USER_PROMPT_WITH_COT_AND_QA.format(question=question, passages=contexts, fewshot=fewshot, qa=qa_str)
        assistant_content = ASSISTANT_PROMPT_WITH_COT.format(answer=answer)

    messages.append({
        "role": "user",
        "content": user_content,
    })

    if 'llama2' in tokenizer.name_or_path:
        inputs = llama2_chat_completion(tokenizer, messages)
        inputs += tokenizer.encode(' ' + assistant_content, add_special_tokens=False)
    else:
        inputs = tokenizer.apply_chat_template(
            messages, 
            add_generation_prompt=True)
        inputs += tokenizer.encode(assistant_content, add_special_tokens=False)

    # inputs = tokenizer.apply_chat_template(
    #     messages, 
    #     add_generation_prompt=True)

    # inputs += tokenizer.encode(assistant_content, add_special_tokens=False)
    return inputs