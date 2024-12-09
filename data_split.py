import argparse
import random
from datasets import load_dataset
import os
from langdetect import detect, DetectorFactory, LangDetectException

def set_seed(seed=42):
    random.seed(seed)

def is_english(text):
    try:
        return detect(text) == 'en'
    except LangDetectException:
        return False

def filter_non_english(example):
    for msg in example.get('conversation_a', []):
        content = msg.get('content', '')
        if not is_english(content):
            return False
    for msg in example.get('conversation_b', []):
        content = msg.get('content', '')
        if not is_english(content):
            return False
    return True

def main(args):
    set_seed(args.seed)

    dataset = load_dataset("lmsys/chatbot_arena_conversations")
    full_dataset = dataset["train"]
    original_size = len(full_dataset)
    print(f"Orignal size: {original_size}")

    DetectorFactory.seed = args.seed
    filtered_dataset = full_dataset.filter(filter_non_english, num_proc=args.num_proc)
    filtered_size = len(filtered_dataset)
    filtered_out_size = original_size - filtered_size
    print(f"Filtered size: {filtered_size}")
    print(f"Filtered out size: {filtered_out_size}")

    test_size = int(filtered_size * 0.1)
    val_size = test_size
    train_size = filtered_size - test_size - val_size

    shuffled_indices = list(range(filtered_size))
    random.shuffle(shuffled_indices)

    train_indices = shuffled_indices[:train_size]
    val_indices = shuffled_indices[train_size:train_size+val_size]
    test_indices = shuffled_indices[train_size+val_size:]

    train_dataset = filtered_dataset.select(train_indices)
    val_dataset = filtered_dataset.select(val_indices)
    test_dataset = filtered_dataset.select(test_indices)

    print(f"Train size: {len(train_dataset)}")
    print(f"Val size: {len(val_dataset)}")
    print(f"Test size: {len(test_dataset)}")

    train_dataset.save_to_disk(os.path.join(args.output_dir, "train"))
    val_dataset.save_to_disk(os.path.join(args.output_dir, "val"))
    test_dataset.save_to_disk(os.path.join(args.output_dir, "test"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter non-English data and split dataset into train/val/test.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--output_dir", type=str, default="filtered_split_data", help="Directory to save split datasets.")
    parser.add_argument("--num_proc", type=int, default=4, help="Number of processes for filtering.")
    args = parser.parse_args()
    main(args)
