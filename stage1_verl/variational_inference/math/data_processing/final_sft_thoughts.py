"""
Generate SFT thoughts dataset from rewrite and thinking datasets.
"""
import os
import datasets
from datasets import get_dataset_config_names, concatenate_datasets

from jinja2 import Template
from verl.utils.hdfs_io import copy, makedirs
import argparse

from verl.utils.reward_score.math import remove_boxed, last_boxed_only_string
from variational_inference.math.data_processing.utils import load_subsets_parallel


def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))


def get_prompt_template(which_prompt):
    if which_prompt == "basic":
        prompt_template_jinja = """ {{prompt}} """
    else:
        raise ValueError(f"Unsupported prompt template: {which_prompt}")

    prompt_template = Template(prompt_template_jinja)
    return prompt_template

def match_datasets_by_index(rewrite_ds, thinking_ds, split='train'):
    """
    For each index, select min(n_rewrite, n_thinking) responses from both datasets
    and create merged items with both responses.
    
    Args:
        rewrite_ds: Dataset with 'index' field
        thinking_ds: Dataset with 'index' field
        split: Which split to use (default: 'train')
    
    Returns:
        Dataset with merged items containing 'think_response' and 'rewrite_response'
    """
    from collections import defaultdict
    # Group items by index for both datasets
    rewrite_by_index = defaultdict(list)
    thinking_by_index = defaultdict(list)
    
    for example in rewrite_ds[split]:
        rewrite_by_index[example['index']].append(example)
    
    for example in thinking_ds[split]:
        thinking_by_index[example['index']].append(example)
    
    # Find common indices
    common_indices = set(rewrite_by_index.keys()) & set(thinking_by_index.keys())
    
    # Create merged items
    merged_items = []
    total_rewrite = 0
    total_thinking = 0
    
    for idx in sorted(common_indices):
        rewrite_items = rewrite_by_index[idx]
        thinking_items = thinking_by_index[idx]
        
        # Take minimum number from both
        n = min(len(rewrite_items), len(thinking_items))
        total_rewrite += len(rewrite_items)
        total_thinking += len(thinking_items)
        
        for i in range(n):
            rewrite_item = rewrite_items[i]
            thinking_item = thinking_items[i]
            thinking_response = thinking_item['responses']
            if "</think>" not in thinking_response:
                thinking_response = f"<think>{thinking_response}</think>\n\n"
            else:
                thinking_response = thinking_response.split("</think>")[0] + "</think>\n\n"
            
            # Create merged item with common fields and both responses
            merged_item = {
                'data_source': rewrite_item['data_source'],
                # 'prompt': rewrite_item['prompt'],
                'ability': rewrite_item['ability'],
                'reward_model': rewrite_item['reward_model'],
                'extra_info': rewrite_item['extra_info'],
                'response': rewrite_item['responses'],
                'thinking': thinking_response,
                'index': idx
            }
            merged_items.append(merged_item)
    
    print(f"Total common indices: {len(common_indices)}")
    print(f"Total rewrite items with common indices: {total_rewrite}")
    print(f"Total thinking items with common indices: {total_thinking}")
    print(f"Created {len(merged_items)} merged items")
    
    # Create a new dataset from the merged items
    merged_dataset = datasets.Dataset.from_list(merged_items)
    
    return merged_dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--rewrite_path', required=True, help="path to the rewrite dataset")
    parser.add_argument(
        '--thinking_path', default=None, help="path to the thinking dataset")
    parser.add_argument(
        "--local_dir", required=True, help="local directory to save the dataset")
    parser.add_argument("--data_source", required=True, help="data source name")
    parser.add_argument("--which_prompt", default="basic", choices=["basic"])
    parser.add_argument("--mix_human_solution", action="store_true", help="mix human solution with the thinking response")
    args = parser.parse_args()
    def add_index(example):
        example["index"] = example['extra_info']['index']
        return example
        
    # merge thinking and rewrite datasets by index
    rewrite_ds = datasets.load_dataset(args.rewrite_path)
    thinking_ds = datasets.load_dataset(args.thinking_path)
    rewrite_ds = rewrite_ds.map(add_index)
    thinking_ds = thinking_ds.map(add_index)
    merged_ds = match_datasets_by_index(rewrite_ds, thinking_ds)
    # split into train and test
    split_ds = merged_ds.train_test_split(test_size=0.05)
    train_ds = split_ds['train']
    test_ds = split_ds['test']

    prompt_template = get_prompt_template(args.which_prompt)
    def make_map_fn(split):
        def process_fn(example, idx):
            # prompt and full response
            extra_info = example.pop("extra_info")
            question = extra_info.pop("question")
            prompt_message = prompt_template.render(prompt=question)
            prompt = [{"role": "user", "content": prompt_message}]
            thinking = example.pop("thinking")
            response = example.pop("response")
            human_solution = extra_info["solution"]
            if args.mix_human_solution:
                full_response = f"{thinking}{human_solution}"
            else:
                full_response = f"{thinking}{response}"            
            # answer
            reward_model = example.pop("reward_model")
            answer = str(reward_model["ground_truth"])

            data = {
                "data_source": args.data_source,
                "prompt": prompt,
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer
                },
                "extra_info": {
                    'index': idx,
                    'split': split,
                    'solution': full_response,
                    "question": prompt,
                    "answer": answer
                }
            }
            return data

        return process_fn


    local_dir = args.local_dir
    columns_to_keep = ['data_source', 'prompt', 'ability', 'reward_model', 'extra_info']
    
    if train_ds:
        train_ds = train_ds.map(function=make_map_fn(
            split='train'), with_indices=True)        
        # only keep the columns that are needed for the training
        train_ds = train_ds.select_columns(columns_to_keep)
        print("Sample training data: ")
        print("Prompt: ", train_ds[0]['prompt'])
        print("Answer: ", train_ds[0]['extra_info']['answer'])
        print("Solution: ", train_ds[0]['extra_info']['solution'])
        print("Total number of training data: ", len(train_ds))
        train_ds.to_parquet(os.path.join(local_dir, 'train.parquet'))
    
    if test_ds:
        test_ds = test_ds.map(function=make_map_fn(
            split='test'), with_indices=True)        
        # only keep the columns that are needed for the testing
        test_ds = test_ds.select_columns(columns_to_keep)
        print("Total number of test data: ", len(test_ds))
        test_ds.to_parquet(os.path.join(local_dir, 'test.parquet'))
        print("Sample test data: ")
        print("Prompt: ", test_ds[0]['prompt'])
        print("Answer: ", test_ds[0]['extra_info']['answer'])
        print("Solution: ", test_ds[0]['extra_info']['solution'])
        print("Total number of test data: ", len(test_ds))
        test_ds.to_parquet(os.path.join(local_dir, 'test.parquet'))