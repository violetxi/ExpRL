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

from jinja2 import Template
from verl.utils.hdfs_io import copy, makedirs
import argparse

from verl.utils.reward_score.math import remove_boxed, last_boxed_only_string


def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data/big-math')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument("--data_source", default="RLAIF/math")
    parser.add_argument("--problem_key", default="problem")
    parser.add_argument("--answer_key", default="answer")
    parser.add_argument("--solution_key", default="solution", help="key to the full solution")
    args = parser.parse_args()
    
    data_source = args.data_source
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    if data_source in ["agentica-org/DeepScaleR-Preview-Dataset", "HuggingFaceH4/MATH-500", "HuggingFaceH4/aime_2024"]:
        dataset = datasets.load_dataset(data_source)
    elif data_source in ['Hothan/OlympiadBench']:
        dataset = datasets.load_dataset(data_source, "OE_TO_maths_en_COMP", split='train')
        # remove the list of answers and solutions
        dataset = dataset.map(lambda x: {args.answer_key: x[args.answer_key][0], args.solution_key: x[args.solution_key][0]})
        dataset = datasets.DatasetDict({"test": dataset})
    elif data_source in ["AI-MO/aimo-validation-amc"]:        
        dataset = datasets.load_dataset(data_source, split='train')        
        # add a 'solution' column with null values
        dataset = dataset.map(lambda x: {"solution": "N/A"})
        dataset = datasets.DatasetDict({"test": dataset})
    elif data_source in ["opencompass/AIME2025"]:
        ds1 = datasets.load_dataset(data_source, "AIME2025-I", split='test')
        ds2 = datasets.load_dataset(data_source, "AIME2025-II", split='test')
        ds = datasets.concatenate_datasets([ds1, ds2])
        dataset = datasets.DatasetDict({"test": ds})
    else:
        raise ValueError(f"Unsupported data source: {data_source}")
    
    # filter out data where solution_key or answer_key is for training only
    if data_source == "agentica-org/DeepScaleR-Preview-Dataset":
        dataset = dataset.filter(lambda x: x[args.solution_key] and x[args.answer_key])
    else:
        dataset = dataset.filter(lambda x: x[args.answer_key])

    if 'train' in dataset:
        train_dataset = dataset['train']
    else:
        train_dataset = None
    
    if 'test' in dataset:
        test_dataset = dataset['test']
    else:
        test_dataset = None
    
    # add a row to each data item that represents a unique id    
    prompt_template_jinja = """\
<|im_start|>user\n{{question}}\nThink hard to solve the problem and show your full reasoning process. Then, provide your complete solution inside <solution> </solution> tags, and put only the final answer within \\boxed{}.\n<|im_end|>
<|im_start|>assistant\n
"""
    prompt_template = Template(prompt_template_jinja)

    def make_map_fn(split, is_train):

        def process_fn(example, idx):
            question = example.pop(args.problem_key).lstrip()
            question = prompt_template.render(question=question)
            answer = str(example.pop(args.answer_key))
            if is_train:
                solution = example.pop(args.solution_key)
            else:
                if args.solution_key in example:
                    solution = example.pop(args.solution_key)
                else:
                    solution = "N/A"

            prompt = question

            data = {
                "data_source": data_source,
                "prompt": prompt,
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": answer
                },
                "extra_info": {
                    'index': idx,
                    'split': split,
                    'solution': solution,
                    "question": question,
                    "answer": answer
                }
            }
            return data

        return process_fn


    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir
    columns_to_keep = ['data_source', 'prompt', 'ability', 'reward_model', 'extra_info']
    
    if train_dataset:
        train_dataset = train_dataset.map(function=make_map_fn(split='train', is_train=True), with_indices=True)
        print("Total number of training data: ", len(train_dataset))
        train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
        print("Sample training data: ")
        print("Prompt: ", train_dataset[0]['prompt'])
        print("Answer: ", train_dataset[0]['extra_info']['answer'])
        print("Solution: ", train_dataset[0]['extra_info']['solution'])

    if test_dataset:
        test_dataset = test_dataset.map(function=make_map_fn(split='test', is_train=False), with_indices=True)        
        print("Total number of test data: ", len(test_dataset))
        test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
        print("Sample test data: ")
        print("Prompt: ", test_dataset[0]['prompt'])
        print("Answer: ", test_dataset[0]['extra_info']['answer'])
        print("Solution: ", test_dataset[0]['extra_info']['solution'])

    if hdfs_dir is not None:
        makedirs(hdfs_dir)

        copy(src=local_dir, dst=hdfs_dir)