import argparse
import random
from datasets import load_dataset

def main(args):
    # load dataset
    dataset = load_dataset("lmsys/chatbot_arena_conversations")
    full_dataset = dataset["train"]

    random.seed(args.seed)

    # 80% train, 10% val, 10% test
    test_size = int(len(full_dataset) * 0.1)
    val_size = test_size
    train_size = len(full_dataset) - test_size - val_size

    shuffled_indices = list(range(len(full_dataset)))
    random.shuffle(shuffled_indices)

    train_indices = shuffled_indices[:train_size]
    val_indices = shuffled_indices[train_size:train_size+val_size]
    test_indices = shuffled_indices[train_size+val_size:]

    train_dataset = full_dataset.select(train_indices)
    val_dataset = full_dataset.select(val_indices)
    test_dataset = full_dataset.select(test_indices)

    train_dataset.save_to_disk(args.output_dir + "/train")
    val_dataset.save_to_disk(args.output_dir + "/val")
    test_dataset.save_to_disk(args.output_dir + "/test")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="split_data")
    args = parser.parse_args()
    main(args)
