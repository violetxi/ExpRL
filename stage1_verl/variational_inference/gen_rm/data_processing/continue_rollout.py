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
    if which_prompt == "rollout":
        system_prompt = None
        prompt_template_jinja = """\
You are given a problem and a partial solution.

Your task is to carefully study the partial response, identify what reasoning or steps are already provided, and then complete the solution from where it left off. Ensure your continuation is logically consistent and leads to a complete and correct final answer.

**Important**: Show your reasoning step-by-step, and clearly present the final answer using LaTeX-style `\\boxed{}` notation.

Problem:
{{question}}

Partial Response:
{{partial_response}}

Continue solving the problem, starting from where the partial response ends. Make sure your final answer is written as: \\boxed{your_answer_here}
"""
    elif which_prompt == "judge":
        system_prompt = """You are an expert mathematician and a meticulous AI reasoning evaluator. Your task is to assess the quality of a "Generated Reasoning" trace.
This trace may be **partial or incomplete**.

Specifically, you must judge how likely it is that a model, *after* thinking through the "Generated Reasoning" (even if it's just a starting point), would *continue* on a correct path to eventually produce the *exact* "Reference Solution" provided.

You are evaluating the **logical and causal link** between this (potentially partial) reasoning and the final answer. Is this a correct and promising *prefix* of a full, correct reasoning trace?

### Instructions

First, provide a step-by-step analysis of the connection between the "Generated Reasoning" and the "Reference Solution." In your analysis, consider the following:

* **Correctness & Alignment:** Is the reasoning *so far* mathematically sound? Does it align with the known facts of the problem and the logical path required to reach the "Reference Solution"?
* **Progression:** Does this partial trace represent a *correct and logical step* (or steps) towards the solution? Is it on the right path, or has it already made a mistake, taken an unproductive turn, or stopped at a point that isn't a clear step forward?
* **Contradiction:** Is there anything *in this partial trace* that already contradicts the "Reference Solution" or makes it impossible to arrive at it logically? Does it set up a line of thinking that would lead to a *different* answer?

After your analysis, provide a single numerical score on the 5-point Likert scale defined below.

**Likert Scale:**
* **1 (Very Unlikely):** The partial reasoning is incorrect, unrelated, or already points toward a completely different solution.
* **2 (Unlikely):** The partial reasoning has significant flaws, is a "dead end," or is too vague, making it highly improbable that a correct continuation would follow from it.
* **3 (Neutral/Possible):** The partial reasoning is generally on the right track (or at least not wrong) but is very incomplete, trivial, or doesn't represent a significant step forward. It's plausible but not guaranteed to lead to the solution.
* **4 (Likely):** The partial reasoning is correct, logical, and represents a clear and significant step on the path to the "Reference Solution."
* **5 (Very Likely):** The partial reasoning is a sound, strong, and unambiguous *prefix* of a correct path to the "Reference Solution." The next logical step is clearly in the direction of the final answer.

---

Please follow this output format:
Reasoning: [Your detailed analysis goes here.]

Score: [Provide the single numerical score: 1, 2, 3, 4, or 5.]
"""
        prompt_template_jinja = """### Math Problem
{{question}}

### Generated Reasoning
{{generated_reasoning}}

### Reference Solution
{{reference_solution}}
"""
    else:
        raise ValueError(f"Unsupported prompt template: {which_prompt}")
    prompt_template = Template(prompt_template_jinja)
    return system_prompt, prompt_template

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_source", default="Hothan/OlympiadBench")
    parser.add_argument('--local_dir', default='data/process_reward/olympiadbench')
    parser.add_argument("--which_prompt", default="rollout", choices=["rollout", "judge"])
    args = parser.parse_args()
    
    data_source = args.data_source
    print(f"Loading the {data_source} dataset from huggingface...", flush=True)
    if data_source in ["violetxi/qwen4b-no-thinking-omni-l5-score-steps", 
        "violetxi/qwen4b-no-thinking-omni-l2-score-steps", 
        "violetxi/qwen4b-instruct-2507-omni-l7-score-steps",
        "violetxi/omni-rule-l7-above-gemini-pro-filtered_qwen3-4b-instruct-2507-steps",
        "violetxi/judge-calibration_set1_16k_qwen3-4b-score-steps"]:
        dataset = datasets.load_dataset(data_source)    
    else:
        raise ValueError(f"Unsupported data source: {data_source}")
        
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
    system_prompt, prompt_template = get_prompt_template(args.which_prompt)
    def make_map_fn(split, which_prompt):
        def process_fn(example, idx):
            question = example['problem']
            partial_response = example['step_solution']
            reference_solution = example['extra_info']['solution']
            if which_prompt == "rollout":
                question = prompt_template.render(question=question, partial_response=partial_response)
                prompt = [{"role": "user", "content": question}]
            elif which_prompt == "judge":
                question = prompt_template.render(question=question, generated_reasoning=partial_response, reference_solution=reference_solution)
                prompt = [{"role": "system", "content": system_prompt}, {"role": "user", "content": question}]
            else:
                raise ValueError(f"Unsupported prompt template: {which_prompt}")            

            reward_model = example['reward_model']
            answer = reward_model['ground_truth']
            solution = example['extra_info']['solution']
            data_source = example['data_source']
            data = {
                "data_source": data_source,
                "prompt": prompt,
                "ability": "math",
                'problem_index': example['problem_index'],
                'step_index': example['step_index'],
                'original_score': example['original_score'],
                'original_response': example['original_response'],                
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
    
    if train_dataset:
        train_dataset = train_dataset.map(
            function=make_map_fn(split='train', which_prompt=args.which_prompt), 
            with_indices=True, num_proc=1)
        print("Sample training data: ")
        print("Prompt: ", train_dataset[0]['prompt'])
        print("Answer: ", train_dataset[0]['extra_info']['answer'])
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
        print("Answer: ", test_dataset[0]['extra_info']['answer'])
        print("Solution: ", test_dataset[0]['extra_info']['solution'])

    # print(train_dataset[0]['prompt'][0]['content'])