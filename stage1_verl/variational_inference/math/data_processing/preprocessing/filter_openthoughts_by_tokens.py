"""
Filter OpenThoughts3-1.2M dataset to keep only rows where
total tokens (human + gpt) < 16000 using Qwen3-4B-Base tokenizer
"""
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import os


def count_tokens(example, tokenizer):
    """Count total tokens in human and gpt fields"""    
    human_text = example['conversations'][0].get("value", "")
    gpt_text = example['conversations'][1].get("value", "")

    # Tokenize both fields
    human_tokens = len(tokenizer.encode(human_text, add_special_tokens=False))
    gpt_tokens = len(tokenizer.encode(gpt_text, add_special_tokens=False))

    total_tokens = human_tokens + gpt_tokens
    print(f"Human tokens: {human_tokens}, GPT tokens: {gpt_tokens}, Total tokens: {total_tokens}")
    return total_tokens


def filter_by_token_count(data_path, output_path, max_tokens=16000):
    """Filter dataset to keep only rows with total tokens < max_tokens"""

    print(f"Loading dataset from {data_path}...")
    ds = load_dataset(data_path)

    print(f"Loading tokenizer from Qwen/Qwen3-4B-Base...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Base")

    print(f"Original dataset size: {len(ds['train'])}")

    def filter_fn(example):
        total_tokens = count_tokens(example, tokenizer)        
        return total_tokens < max_tokens

    print(f"Filtering rows with total tokens < {max_tokens}...")
    filtered_ds = ds.filter(filter_fn, num_proc=os.cpu_count())

    print(f"Filtered dataset size: {len(filtered_ds['train'])}")
    print(f"Removed {len(ds['train']) - len(filtered_ds['train'])} rows")

    if output_path:
        print(f"Pushing filtered dataset to {output_path}...")
        filtered_ds.push_to_hub(output_path, private=False)
        print("Done!")

    return filtered_ds


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter OpenThoughts dataset by token count")
    parser.add_argument(
        "--data_path",
        type=str,
        default="open-thoughts/OpenThoughts3-1.2M",
        help="Source dataset path on HuggingFace Hub"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output dataset path on HuggingFace Hub (optional)"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=16000,
        help="Maximum total tokens (human + gpt)"
    )

    args = parser.parse_args()

    filter_by_token_count(args.data_path, args.output_path, args.max_tokens)
