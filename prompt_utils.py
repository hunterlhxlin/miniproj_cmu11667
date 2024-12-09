# utils.py

def generate_prompt(q, a1, a2, mode='eval', label=None):
    prompt_prefix = (
        "You are given a question and two possible answers. "
        "You need to decide which answer is better. "
        "You can choose one of the following labels: "
        "model_a, model_b, tie, tie(bothbad). \n\n"
        f"Question:\n{q}\n\n"
        f"model_a Answer:\n{a1}\n\n"
        f"model_b Answer:\n{a2}\n\n"
        "Which is better?\n\n"
        f"Label: "
    )
    if mode == 'train':
        prompt = prompt_prefix + label
    elif mode == 'eval':
        prompt = prompt_prefix
    else:
        raise ValueError("Invalid mode. Use 'train' or 'eval'.")
    
    return prompt

def generate_few_shot_prompts(few_shot_examples, q, a1, a2):
    prompt = (
        "You are given a question and two possible answers. "
        "You need to decide which answer is better. "
        "You can choose one of the following labels: "
        "model_a, model_b, tie, tie(bothbad). \n\n"
        "Consider the following examples to understand the task better:\n\n"
    )
    for example in few_shot_examples:
        prompt += (
            f"Example question:\n{example['question']}\n\n"
            f"model_a Answer:\n{example['answer_a']}\n\n"
            f"model_b Answer:\n{example['answer_b']}\n\n"
            f"Label: {example['label']}\n\n"
        )
    prompt += (
        f"Question:\n{q}\n\n"
        f"model_a Answer:\n{a1}\n\n"
        f"model_b Answer:\n{a2}\n\n"
        "Which is better?\n\n"
        "Label: "
    )
    return prompt
