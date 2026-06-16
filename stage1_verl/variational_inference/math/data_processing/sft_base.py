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
    if which_prompt == "qwen3":   
        # prompt_template_jinja = """\
        #     <|im_start|>user\n{{question}}\nPlease reason step by step, and put your final answer within <answer> \\boxed{} </answer>.<|im_end|>
        #     <|im_start|>assistant\n
        #     """
        prompt_template_jinja = """{{question}}\nPlease reason step by step, and put your final answer within \\boxed{}."""
    else:
        raise ValueError(f"Unsupported prompt template: {which_prompt}")

    prompt_template = Template(prompt_template_jinja)
    return prompt_template

def make_map_fn(split):
    def process_fn(example, idx):
        # prompt and full response
        conversations = example.pop("conversations")
        question = conversations[0]["value"]            
        prompt = prompt_template.render(question=question)
        solution = conversations[1]["value"]            
        domain = example.pop("domain")            

        data = {
            "data_source": args.data_path,
            "prompt": prompt,
            "ability": "math",                
            "extra_info": {
                'index': idx,
                'split': split,
                'solution': solution,
                "question": question,
                "domain": domain
            }
        }
        return data

    return process_fn


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--data_path', required=True, help="path to the dataset")
    parser.add_argument(
        "--local_dir", required=True, help="local directory to save the dataset")
    parser.add_argument("--which_prompt", default="qwen3", choices=["qwen3"])
    args = parser.parse_args()
    dataset = datasets.load_dataset(args.data_path)
    if "test" in dataset and "train" in dataset:
        train_ds = dataset["train"]
        test_ds = dataset["test"]    
    else:    # split into train and test
        dataset = dataset["train"]
        split_ds = dataset.train_test_split(test_size=0.01)
        train_ds = split_ds["train"]
        test_ds = split_ds["test"]

    prompt_template = get_prompt_template(args.which_prompt)
    local_dir = args.local_dir
    columns_to_keep = ['data_source', 'prompt', 'ability', 'extra_info']
    
    if train_ds:
        train_ds = train_ds.map(function=make_map_fn(
            split='train'), with_indices=True)        
        # only keep the columns that are needed for the training
        train_ds = train_ds.select_columns(columns_to_keep)        
        print("Sample training data: ")
        print("Prompt: ", train_ds[0]['prompt'])        
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
        print("Solution: ", test_ds[0]['extra_info']['solution'])
        print("Total number of test data: ", len(test_ds))
        test_ds.to_parquet(os.path.join(local_dir, 'test.parquet'))