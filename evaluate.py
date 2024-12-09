import argparse
import os
import torch
import random
import numpy as np
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)
from sklearn.metrics import accuracy_score, f1_score
import wandb
from peft import PeftModel
from prompt_utils import generate_prompt, generate_few_shot_prompts

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

label_list = ["model_a", "model_b", "tie", "tie (bothbad)"]
label2id = {l: i for i, l in enumerate(label_list)}
id2label = {i: l for l, i in label2id.items()}

def predict_sft(model, tokenizer, dataset, batch_size=8): 
    model.eval()
    predictions = []
    references = []
    device = model.device
    
    dataset_size = len(dataset)
    num_batches = (dataset_size + batch_size - 1) // batch_size

    for i in range(num_batches):
        batch = dataset[i*batch_size : (i+1)*batch_size]
        prompts = [
            generate_prompt(example["conversation_a"], example["conversation_b"], mode="eval")
            for example in batch
        ]
        
        inputs = tokenizer(prompts, padding=True).to(device)
        
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,              
                temperature=1.0,
                top_k=8,
            )
        generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        
        for j, text in enumerate(generated_texts):
            predicted_label = "tie"
            if "tie (bothbad)" in text:
                predicted_label = "tie (bothbad)"
            elif "tie" in text:
                predicted_label = "tie"
            elif "option a" in text or " a" in text:
                predicted_label = "model_a"
            elif "option b" in text or " b" in text:
                predicted_label = "model_b"
            else:
                print(f"Warning: Unrecognized label in generated text: {text}")
                predicted_label = "tie"
            
            predictions.append(label2id[predicted_label])
            references.append(label2id[batch[j]["winner"]])

    acc = accuracy_score(references, predictions)
    f1 = f1_score(references, predictions, average="weighted")
    return acc, f1

def predict_icl(model, tokenizer, dataset, few_shot_examples, N=4):
    model.eval()
    device = model.device

    predictions = []
    references = []
    for ex in dataset:
        gold_label = label2id[ex["winner"]]
        prompt = generate_few_shot_prompts(few_shot_examples, ex["conversation_a"], ex["conversation_b"])
        inputs = tokenizer(prompt, padding=True).to(device)
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=20)
            output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        chosen_label = "tie"
        output_text_lower = output_text.lower()
        
        if "tie (bothbad)" in output_text_lower:
            chosen_label = "tie (bothbad)"
        elif "tie" in output_text_lower:
            chosen_label = "tie"
        elif "model_a" in output_text_lower:
            chosen_label = "model_a"
        elif "model_b" in output_text_lower:
            chosen_label = "model_b"
        else:
            print(f"Warning: Unrecognized label in generated text: {output_text}")
            chosen_label = "None"

        predictions.append(label2id[chosen_label])
        references.append(gold_label)

    acc = accuracy_score(references, predictions)
    f1 = f1_score(references, predictions, average='weighted')
    return acc, f1


def main(args):
    set_seed(args.seed)

    if args.use_wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name)

    test_dataset = load_from_disk(os.path.join(args.data_dir, "test"))

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.finetune_method == "full":
        model = AutoModelForCausalLM.from_pretrained(args.model_dir, device_map="auto")
    elif args.finetune_method in ["lora", "prefix"]:
        model = PeftModel.from_pretrained(args.model_dir, device_map="auto")
    else:
        raise ValueError("finetune_method should be 'full', 'lora' or 'prefix'")

    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    if args.eval_method == "sft":
        acc, f1 = predict_sft(model, tokenizer, test_dataset, args.batch_size)
    elif args.eval_method == "icl":
        train_dataset = load_from_disk(os.path.join(args.data_dir, "train"))
        few_shot_examples = train_dataset.select(range(args.few_shot))
        acc, f1 = predict_icl(model, tokenizer, test_dataset, few_shot_examples, N=args.few_shot)
    else:
        raise ValueError("eval_method should be either 'sft' or 'icl'")

    print({"accuracy": acc, "weighted_f1": f1})
    if args.use_wandb:
        wandb.log({"test_accuracy": acc, "test_weighted_f1": f1})
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned language models.")

    parser.add_argument("--data_dir", type=str, default="./split_data", help="Path to the split dataset directory.")
    parser.add_argument("--model_dir", type=str, default="meta-llama/Llama-3.2-1B-Instruct", help="Path to the trained model directory.")
    parser.add_argument("--finetune_method", type=str, choices=["full","lora","prefix"], required=True,
                        help="Fine-tuning method used: 'full', 'lora', or 'prefix'.")
    parser.add_argument("--eval_method", type=str, choices=["sft","icl"], required=True,
                        help="Evaluation method: 'sft' for supervised fine-tuning, 'icl' for in-context learning.")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation.")
    parser.add_argument("--few_shot", type=int, default=4, help="Number of few-shot examples for ICL.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--use_wandb", action="store_true", help="Whether to use Weights & Biases for logging.")
    parser.add_argument("--wandb_project", type=str, default="llm_evaluation", help="W&B project name.")
    parser.add_argument("--wandb_run_name", type=str, default="eval_run", help="W&B run name.")

    args = parser.parse_args()
    main(args)
