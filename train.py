import argparse
import os
import random
import numpy as np
import torch
import wandb
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from functools import partial
from peft import LoraConfig, PrefixTuningConfig, get_peft_model
from prompt_utils import generate_prompt

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def preprocess_function(examples, tokenizer):
    inputs = []
    for a1, a2, l in zip(examples["conversation_a"], examples["conversation_b"], examples["winner"]):
        text = generate_prompt(a1, a2, mode='train', label=l)
        inputs.append(text)

    model_inputs = tokenizer(inputs, padding='max_length', max_length=12000)
    model_inputs["labels"] = model_inputs["input_ids"]
    return model_inputs

def main(args):
    set_seed(args.seed)

    # Initialize W&B if enabled
    if args.use_wandb:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name)

    # load dataset
    train_dataset = load_from_disk(os.path.join(args.data_dir, "train"))
    val_dataset = load_from_disk(os.path.join(args.data_dir, "val"))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto")

    if args.finetune_method == "full":
        # Full fine-tuning
        pass
    elif args.finetune_method == "lora":
        # LoRA
        lora_config = LoraConfig(
            r=4,
            lora_alpha=32,
            target_modules=["q_proj","v_proj"], 
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)
    elif args.finetune_method == "prefix":
        # Prefix Tuning
        prefix_config = PrefixTuningConfig(
            num_virtual_tokens=20,
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, prefix_config)
    else:
        raise ValueError("finetune_method must be one of: full, lora, prefix")

    preprocess_fn = partial(preprocess_function, tokenizer=tokenizer)
    train_encoded = train_dataset.map(preprocess_fn, batched=True, remove_columns=train_dataset.column_names)
    val_encoded = val_dataset.map(preprocess_fn, batched=True, remove_columns=val_dataset.column_names)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        evaluation_strategy="epoch",  
        save_strategy="no",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        logging_steps=10,
        fp16=True if torch.cuda.is_available() else False,
        report_to="wandb" if args.use_wandb else None
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_encoded,
        eval_dataset=val_encoded,
        tokenizer=tokenizer,
        data_collator=data_collator
    )

    trainer.train()

    # save model
    trainer.save_model(args.save_model_dir)
    tokenizer.save_pretrained(args.save_model_dir)

    if args.use_wandb:
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, default="./split_data", help="Path to the splitted dataset")
    parser.add_argument("--model_name", type=str, required=True, help="Model name, e.g. meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--finetune_method", type=str, choices=["full","lora","prefix"], default="full", required=True, help="Finetuning method")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    parser.add_argument("--save_model_dir", type=str, default="./trained_model")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="llm_training")
    parser.add_argument("--wandb_run_name", type=str, default="run_name")
    args = parser.parse_args()
    main(args)
