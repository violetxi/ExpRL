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
"""
Offline evaluate the performance of a generated file using reward model and ground truth verifier.
The input is a parquet file that contains N generated sequences and (optional) the ground truth.

"""

from collections import defaultdict
import os
import socket
import hydra
import numpy as np
import pandas as pd
import ray
from tqdm import tqdm
import datasets

from verl.trainer.ppo.reward import get_custom_reward_fn
from verl.utils.fs import copy_to_local
from verl.utils.reward_score import default_compute_score


@ray.remote
def process_item(idx, reward_fn, data_source, response, reward_data):
    ground_truth = reward_data["ground_truth"]
    score = reward_fn(data_source, response, ground_truth)
    return idx, data_source, score


@hydra.main(config_path="config", config_name="evaluation", version_base=None)
def main(config):
    # local_path = copy_to_local(config.data.path, use_shm=config.data.get("use_shm", False))
    # dataset = pd.read_parquet(local_path)
    dataset = datasets.load_dataset(config.data.path, split='train')
    dataset = dataset.to_pandas()
    responses = dataset[config.data.response_key]
    if config.data.data_source_key not in dataset.columns:
        # this is a sign the dataset is not standard verl processed
        data_sources = ["prime_test"] * len(dataset)
        reward_model_data = [{"ground_truth": gt} for gt in dataset[config.data.ground_truth_key]]
    else:
        data_sources = dataset[config.data.data_source_key]
        reward_model_data = dataset[config.data.reward_model_key]

    total = len(dataset)
    # evaluate test_score based on data source
    data_source_reward = defaultdict(list)
    # compute_score = get_custom_reward_fn(config)
    compute_score = default_compute_score    

    # Initialize Ray
    if not ray.is_initialized():
        detected_ip = socket.gethostbyname(socket.gethostname())
        os.environ["RAY_NODE_IP_ADDRESS"] = detected_ip  # Forces consistent IP for Raylet
        ray.init(
            runtime_env={"env_vars": {
                "TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", 
                "RAY_DEBUG": "legacy", "RAY_DEBUGGER_EXTERNAL": "1"    # Enable legacy debugger and external exposure in docker
                }},
            num_cpus=config.ray_init.num_cpus,
        )
    
    # Create remote tasks
    if "base" in config.data.path:
        remote_tasks = [
            process_item.remote(i, compute_score, data_sources[i], responses[i][:50], reward_model_data[i]) for i in range(total)
        ]
    else:
            remote_tasks = [
                process_item.remote(i, compute_score, data_sources[i], responses[i], reward_model_data[i]) for i in range(total)
            ]    

    # Process results as they come in
    with tqdm(total=total) as pbar:
        while len(remote_tasks) > 0:
            # Use ray.wait to get completed tasks
            done_ids, remote_tasks = ray.wait(remote_tasks)
            for result_id in done_ids:
                idx, data_source, score = ray.get(result_id)
                # add score to dataset
                dataset.at[idx, 'score'] = score
                data_source_reward[data_source].append(score)
                pbar.update(1)

    metric_dict = {}
    for data_source, rewards in data_source_reward.items():
        metric_dict[f"test_score/{data_source}"] = np.mean(rewards)

    print(metric_dict)    
    output_path = f"{config.data.path}-score"
    # convert to HF dataset
    dataset = datasets.Dataset.from_pandas(dataset)
    dataset.push_to_hub(output_path, private=False)
    print(f"Pushed final ds to huggingface: {output_path}")


if __name__ == "__main__":
    main()
