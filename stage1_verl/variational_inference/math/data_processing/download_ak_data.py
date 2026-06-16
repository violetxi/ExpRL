# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import datasets
from datasets import get_dataset_config_names, concatenate_datasets

from jinja2 import Template
from verl.utils.hdfs_io import copy, makedirs
import argparse

from verl.utils.reward_score.math import remove_boxed, last_boxed_only_string
from variational_inference.math.data_processing.utils import load_subsets_parallel


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='data/rl_behaviors_verl_stable/data_arc_sft_post_trl_bug/arc-barc-processed-direct-max4k-gpt5.2abstractions-rephrased-0104-sft-fewshot8')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument("--data_source", default="asingh15/arc-direct-gpt5.2abstractions-rephrased-sft")
    args = parser.parse_args()
    
    data_source = args.data_source
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    dataset = datasets.load_dataset(data_source)
    train_dataset = dataset['train']
    test_dataset = dataset['test']
    def make_map_fn(split, is_train):

        def process_fn(example, idx):
            question = example['query']
            solution = example['completion']
            prompt = [{"role": "user", "content": question}]

            data = {
                "data_source": data_source,
                "prompt": prompt,
                "ability": "math",                
                "extra_info": {
                    'index': idx,
                    'split': split,
                    'solution': solution,
                    "question": question
                }
            }
            return data

        return process_fn


    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir
    columns_to_keep = ['data_source', 'prompt', 'ability', 'extra_info']
    
    if train_dataset:
        train_dataset = train_dataset.map(function=make_map_fn(split='train', is_train=True), with_indices=True, num_proc=1)
        # only keep the columns that are needed for the training
        train_dataset = train_dataset.select_columns(columns_to_keep)
        print("Sample training data: ")
        print("Prompt: ", train_dataset[0]['prompt'])
        print("Solution: ", train_dataset[0]['extra_info']['solution'])
        print("Total number of training data: ", len(train_dataset))
        train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    
    if test_dataset:
        test_dataset = test_dataset.map(function=make_map_fn(split='test', is_train=False), with_indices=True)        
        # only keep the columns that are needed for the testing
        test_dataset = test_dataset.select_columns(columns_to_keep)
        print("Total number of test data: ", len(test_dataset))
        test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
        print("Sample test data: ")
        print("Prompt: ", test_dataset[0]['prompt'])
        print("Solution: ", test_dataset[0]['extra_info']['solution'])

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)