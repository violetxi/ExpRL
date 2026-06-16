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


def extract_solution(solution_str):
    return remove_boxed(last_boxed_only_string(solution_str))


def get_prompt_template(which_prompt):
    if which_prompt == "basic":
        prompt_template_jinja = """You are an expert mathematician and a meticulous AI reasoning engine. You are capable of deeply understanding complex mathematical problems and formulating the precise, logical steps and background knowledge required to arrive at a given solution.

Your task is to generate the "latent thoughts" or internal monologue that forms a strong, logical bridge from the provided problem to its complete, final solution.

This bridge should be a **single, coherent narrative** that represents the internal reasoning of an ideal problem-solver. As you formulate this reasoning, you should **implicitly weave together** the following mental actions *as needed* in a natural way:

* **Deconstruct the problem**: What is given? What is the goal?
* **Formulate a strategy**: Which concepts, formulas, or theorems are relevant? What is the high-level plan?
* **Justify each step**: Explain *why* a calculation or logical step is performed.
* **Verify and Self-Correct**: Actively check the logic. If a potential mistake, dead end, or assumption is identified (e.g., "Wait, that's not right..." or "I should check this first..."), explicitly state this thought and the subsequent correction.
* **Connect to the solution**: Focus on the implicit knowledge and logical leaps that are *not* explicitly written in the final solution but are necessary to produce it.

Your output under "Underlying Reasoning" should be a single, flowing text. **Do not use markdown headers, bullet points, or explicit labels like "Strategy:", "Deconstruction:", or "Step 1:"** within this reasoning block.

### Problem:
{{problem}}

### Final Solution:
{{solution}}

### Underlying Reasoning
Now, provide the underlying reasoning that connects the problem to the solution. Present this as a clear, step-by-step reasoning trace. Do not simply repeat the solution; explain the *reasoning* that leads to it.
"""    
    else:
        raise ValueError(f"Unsupported prompt template: {which_prompt}")

    prompt_template = Template(prompt_template_jinja)
    return prompt_template

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='data_it/olympiadbench')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument("--data_source", default="Hothan/OlympiadBench")
    parser.add_argument("--problem_key", default="problem")
    parser.add_argument("--answer_key", default="answer")
    parser.add_argument("--solution_key", default=None, help="key to the full solution")
    parser.add_argument("--which_prompt", default="basic", choices=["basic"])
    args = parser.parse_args()
    
    data_source = args.data_source
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    if data_source in [
        "agentica-org/DeepScaleR-Preview-Dataset", "violetxi/omni-math-difficulty-1_2", 
        "violetxi/omni-math-difficulty-2_3"]:
        dataset = datasets.load_dataset(data_source)

    elif data_source in ["CohenQu/AceReason-Math-Qwen3-4B"]:        
        # load all subsets in parallel for much faster downloading
        subset_names = get_dataset_config_names(data_source)
        all_subsets = load_subsets_parallel(data_source, subset_names, max_workers=32)
        dataset = concatenate_datasets(all_subsets)
        dataset = datasets.DatasetDict({"train": dataset})

    elif data_source in ["CohenQu/POPE-hard-dataset-Qwen3-4B-Instruct-32k-128-filtered"]:
        dataset = datasets.load_dataset(data_source)
    
    elif data_source in ["HerrHruby/acemath_rl_4b_inst_hard"]:
        dataset = datasets.load_dataset(data_source)
        dataset = dataset.map(lambda x: {"problem": x["prompt"][0]["content"]})
        dataset = dataset.map(lambda x: {"answer": x["reward_model"]["ground_truth"]})
        # add an "ability" column with the value "math"
        dataset = dataset.map(lambda x: {"ability": "math"})
    else:
        raise ValueError(f"Unsupported data source: {data_source}")
    
    # filter out data where solution_key or answer_key is for training only
    if data_source in ["agentica-org/DeepScaleR-Preview-Dataset", "violetxi/omni-math-difficulty-4_6", "violetxi/omni-math-difficulty-5_7"]:
        dataset = dataset.filter(lambda x: x[args.solution_key] and x[args.answer_key])
    elif data_source in ["HerrHruby/acemath_rl_4b_inst_hard"]:
        dataset = dataset
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
    prompt_template = get_prompt_template(args.which_prompt)
    def make_map_fn(split, is_train):

        def process_fn(example, idx):
            question = example.pop(args.problem_key)
            answer = str(example.pop(args.answer_key))
            if is_train:
                if args.solution_key:
                    solution = example.pop(args.solution_key)
                else:
                    solution = "N/A"
            else:
                if args.solution_key and args.solution_key in example:
                    solution = example.pop(args.solution_key)
                else:
                    solution = "N/A"

            prompt_message = prompt_template.render(problem=question, solution=solution)
            prompt = [{"role": "user", "content": prompt_message}]

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
        train_dataset = train_dataset.map(function=make_map_fn(
            split='train', is_train=True), with_indices=True)
        # only keep the columns that are needed for the training
        train_dataset = train_dataset.select_columns(columns_to_keep)
        print("Sample training data: ")
        print("Prompt: ", train_dataset[0]['prompt'])        
        print("Total number of training data: ", len(train_dataset))
        train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    
    if test_dataset:
        test_dataset = test_dataset.map(function=make_map_fn(
            split='test', is_train=False), with_indices=True)        
        # only keep the columns that are needed for the testing
        test_dataset = test_dataset.select_columns(columns_to_keep)
        print("Total number of test data: ", len(test_dataset))
        test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
        print("Sample test data: ")
        print("Prompt: ", test_dataset[0]['prompt'])        

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)