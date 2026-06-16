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
    if which_prompt == "qwen3":
        prompt_template_jinja = """{{question}}\n"""
    elif which_prompt == "llama":
        # prompt_template_jinja = """Respond to the following user query in a comprehensive and detailed way. But first write down your internal thoughts. This must include your draft response and its evaluation. After this, write your final response after "<R>"."""
        prompt_template_jinja = """{{question}}\n"""    
    else:
        raise ValueError(f"Unsupported prompt template: {which_prompt}")
    prompt_template = Template(prompt_template_jinja)
    return prompt_template

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='data_it/olympiadbench')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument("--data_source", default="Hothan/OlympiadBench")
    parser.add_argument("--prompt_key", default=None)
    parser.add_argument("--problem_key", default=None)
    parser.add_argument("--answer_key", default=None)
    parser.add_argument("--solution_key", default=None, help="key to the full solution")
    parser.add_argument("--which_prompt", default="llama", choices=["qwen3", "llama"])
    args = parser.parse_args()
    
    data_source = args.data_source
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    if data_source in [
        "agentica-org/DeepScaleR-Preview-Dataset", "HuggingFaceH4/MATH-500", 
        "HuggingFaceH4/aime_2024", "violetxi/omni-math-difficulty-2_5", 
        "violetxi/omni-math-difficulty-4_6", "violetxi/omni-math-difficulty-5_7", 
        "violetxi/omni-math-difficulty-2", "violetxi/omni-math-difficulty-5", 
        "violetxi/omni-math-difficulty-8", "violetxi/test-qwen4b-rewrite-omni-l8",
        "violetxi/test-qwen8b-rewrite-omni-l8", "violetxi/test-qwen4b-rewrite-omni-l2",
        "violetxi/test-qwen8b-rewrite-omni-l2", "violetxi/omni-math-difficulty-2_3", 
        "violetxi/omni-math-difficulty-1_2", "violetxi/qwen8b-rewrite-omni-l5", 
        "violetxi/qwen8b-rewrite-omni-l8", 'violetxi/qwen8b-rewrite-omni-l1_2-score',
        "violetxi/omni-math-difficulty-4", "violetxi/qwen4b-thinking-omni-l1_2",         
        "violetxi/qwen4b-thinking-omni-l1_2-score", "violetxi/omni-filtered-by-cohen", 
        "violetxi/qwen4b-thinking-omni-l2_3-score", "violetxi/qwen4b-thinking-omni-l4-score",
        "violetxi/test-qwen8b-rewrite-omni-l1_2-score", "violetxi/test-qwen8b-rewrite-omni-l2_3-score", 
        "violetxi/omni-math-above-l6", "violetxi/omni-math-above-l7", 
        "violetxi/omni-math-above-l7-rule-gemini-pro-filtered",
        "d1shs0ap/unified-hard-set-with-student-solutions-guided-rl",
        "MathArena/aime_2026", "MathArena/hmmt_nov_2025", 
        "CohenQu/POPE-hard-dataset-Qwen3-4B-Instruct-32k-128-filtered-iter3-gemini-success",
        "d1shs0ap/unified-hard-set-with-student-solutions-guided-rl"]:
        dataset = datasets.load_dataset(data_source)

    elif data_source in ["Hwilner/imo-answerbench"]:
        dataset = datasets.load_dataset(data_source)
        # change 'Problem' to 'problem', 'Short Answer' to 'answer'
        dataset = dataset.map(lambda x: {args.problem_key: x["Problem"]})
        dataset = dataset.map(lambda x: {args.answer_key: x["Short Answer"]})        

    elif data_source in ["violetxi/pope-hard-w-guide-gemini-solution-hint"]:
        dataset = datasets.load_dataset(data_source)

    elif data_source in ["CohenQu/AceReason-Math-Qwen3-4B"]:        
        # load all subsets in parallel for much faster downloading
        subset_names = get_dataset_config_names(data_source)
        all_subsets = load_subsets_parallel(data_source, subset_names, max_workers=32)
        dataset = concatenate_datasets(all_subsets)
        dataset = datasets.DatasetDict({"train": dataset})

    elif data_source in ["CohenQu/POPE-hard-dataset-Qwen3-4B-Instruct-32k-128-filtered"]:
        dataset = datasets.load_dataset(data_source)

    elif data_source in ["violetxi/pope-hard-w-guide-gemini-solution"]:
        dataset = datasets.load_dataset(data_source, split='test')
        # change `gemini_solution` to `solution`
        dataset = dataset.map(lambda x: {"solution": x["gemini_solution"]})
        dataset = datasets.DatasetDict({"test": dataset})

    elif data_source in ['Hothan/OlympiadBench']:
        dataset = datasets.load_dataset(data_source, "OE_TO_maths_en_COMP", split='train')
        # remove the list of answers and solutions
        dataset = dataset.map(lambda x: {args.answer_key: x[args.answer_key][0]})
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
    elif data_source in ["HerrHruby/acemath_rl_4b_inst_hard"]:
        dataset = datasets.load_dataset(data_source)
        dataset = dataset.map(lambda x: {"problem": x["prompt"][0]["content"]})
        dataset = dataset.map(lambda x: {"answer": x["reward_model"]["ground_truth"]})
        # add an "ability" column with the value "math"
        dataset = dataset.map(lambda x: {"ability": "math"})
        # # drop the 'reward_model' column
        # dataset = dataset.remove_columns("reward_model")
        # # drop the 'prompt' column
        # dataset = dataset.remove_columns("prompt")
    elif data_source in [
        "violetxi/sciknoweval_oe_v1_gemini-3-pro",
        "violetxi/sciknoweval_oe_v1_physics_gemini-3-pro",
        "violetxi/sciknoweval_mcq_l3_loose_gemini-3-pro",
    ]:
        dataset = datasets.load_dataset(data_source)
    else:
        raise ValueError(f"Unsupported data source: {data_source}")
    
    # filter out data where solution_key or answer_key is for training only
    if data_source in ["agentica-org/DeepScaleR-Preview-Dataset", "violetxi/omni-math-difficulty-4_6", "violetxi/omni-math-difficulty-5_7"]:
        dataset = dataset.filter(lambda x: x[args.solution_key] and x[args.answer_key])
    else:
        dataset = dataset

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
            if args.prompt_key:
                prompt = example.pop(args.prompt_key)
                question = prompt[0]["content"]
            else:
                if args.problem_key:
                    question = example.pop(args.problem_key)
                else:
                    question = example['extra_info']['question']

                if isinstance(question, dict):
                    question = question["question"]
                else:
                    question = question
                question = prompt_template.render(question=question)            

                prompt = [
                    {"role": "system", "content": "Please reason step by step, and put your final answer within \boxed{}."},
                    {"role": "user", "content": question},
                    ]
            
            if args.answer_key in example:
                answer = str(example.pop(args.answer_key))
            else:
                answer = example['reward_model']['ground_truth']

            if args.solution_key:
                solution = example.pop(args.solution_key)
            else:
                if 'extra_info' in example:
                    solution = example['extra_info']['solution']
                else:
                    solution = "N/A"

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

    # Detect oracle/verl-schema datasets (downloaded HF parquets that are already
    # in the trainer's expected shape).  Skip make_map_fn — it would overwrite the
    # dataset-specific system prompt (math \\boxed{}-style) and clobber `ability`.
    already_verl_schema = False
    schema_probe = train_dataset or test_dataset
    if schema_probe is not None:
        already_verl_schema = all(c in schema_probe.column_names for c in columns_to_keep)
        if already_verl_schema:
            print(f"[instruct.py] {data_source} is already in verl schema — passthrough mode")

    def _peek(row, key):
        ei = row.get("extra_info") or {}
        return ei.get(key) or row.get("reward_model", {}).get("ground_truth", "")

    if train_dataset:
        if not already_verl_schema:
            train_dataset = train_dataset.map(function=make_map_fn(split='train', is_train=True), with_indices=True, num_proc=1)
        # only keep the columns that are needed for the training
        train_dataset = train_dataset.select_columns(columns_to_keep)
        print("Sample training data: ")
        print("Prompt: ", train_dataset[0]['prompt'])
        print("Answer: ", _peek(train_dataset[0], "answer"))
        print("Solution: ", str(_peek(train_dataset[0], "solution")))
        print("Total number of training data: ", len(train_dataset))
        train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))

    if test_dataset:
        if not already_verl_schema:
            test_dataset = test_dataset.map(function=make_map_fn(split='test', is_train=False), with_indices=True)
        # only keep the columns that are needed for the testing
        test_dataset = test_dataset.select_columns(columns_to_keep)
        print("Total number of test data: ", len(test_dataset))
        test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))
        print("Sample test data: ")
        print("Prompt: ", test_dataset[0]['prompt'])
        print("Answer: ", _peek(test_dataset[0], "answer"))
        print("Solution: ", str(_peek(test_dataset[0], "solution")))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir)