# utils.py

def generate_conversation(conversation):
    conversation_text = ""
    # for entry in conversation:
    #     if entry["role"] == "user":
    #         conversation_text += f"User: {entry['content']}\n"
    #     else:
    #         conversation_text += f"Model: {entry['content']}\n"
    conversation_text += f"User: {conversation['content']}\n"
    conversation_text += f"Model: {conversation['content']}\n"
    return conversation_text

def generate_prompt(a1, a2, mode='eval', label=None):
    prompt_prefix = (
        "You are given two conversations regarding the same question but with different language models. "
        "You need to decide which answer is better. "
        "You can choose one of the following labels: "
        "model_a, model_b, tie, tie (bothbad). \n\n"
        f"model_a conversation:\n{generate_conversation(a1)}\n\n"
        f"model_b conversation:\n{generate_conversation(a2)}\n\n"
        "Which is better?\n\n"
        f"Label: "
    )
    if mode == 'train':
        if label is None:
            raise ValueError("Label is required for training mode.")
        prompt = prompt_prefix + label
    elif mode == 'eval':
        prompt = prompt_prefix
    else:
        raise ValueError("Invalid mode. Use 'train' or 'eval'.")
    
    return prompt

def generate_few_shot_prompts(few_shot_examples, a1, a2):
    prompt = (
        "You will be given two conversations regarding the same question but with different language models. "
        "You need to decide which answer is better. "
        "You can choose one of the following labels: "
        "model_a, model_b, tie, tie (bothbad). \n\n"
        "Consider the following examples to understand the task better:\n\n"
    )
    for example in few_shot_examples:
        prompt += (
            f"model_a conversation:\n{generate_conversation(example['conversation_a'])}\n\n"
            f"model_b conversation:\n{generate_conversation(example['conversation_b'])}\n\n"
            f"Which is better?\n\n"
            f"Label: {example['winner']}\n\n"
        )
    prompt += (
        "Now consider the following pair of conversations:\n\n"
        f"model_a conversation:\n{generate_conversation(a1)}\n\n"
        f"model_b conversation:\n{generate_conversation(a2)}\n\n"
        "Which is better?\n\n"
        "Label: "
    )
    return prompt
