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
Generate responses given a dataset of prompts
"""

import os
import copy
import socket
import hydra
import numpy as np
import ray
from tqdm import tqdm

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
# os.environ['TORCH_COMPILE_DISABLE'] = '1'

from pprint import pprint

import pandas as pd
from omegaconf import OmegaConf
import datasets
from datasets import Dataset

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.hdfs_io import makedirs
from verl.utils.model import compute_position_id_with_mask
from verl.workers.fsdp_workers import ActorRolloutRefWorker


@hydra.main(config_path="config", config_name="generation", version_base=None)
def main(config):
    run_generation(config)


def run_generation(config) -> None:
    if not ray.is_initialized():
        # expose the ip address of the node to raylet
        detected_ip = socket.gethostbyname(socket.gethostname())
        os.environ["RAY_NODE_IP_ADDRESS"] = detected_ip  # Forces consistent IP for Raylet
        ray.init(
            runtime_env={"env_vars": {
                "TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN", 
                "RAY_DEBUG": "legacy", "RAY_DEBUGGER_EXTERNAL": "1"    # Enable legacy debugger and external exposure in docker
                }},
            num_cpus=config.ray_init.num_cpus,
        )

    ray.get(main_task.remote(config))


def load_data(config):    
    def _expand_dataset(dataset: datasets.Dataset, n: int) -> datasets.Dataset:
        """
        Expands a Hugging Face dataset by repeating each row n_repeats times
        with different memory references.
        
        Args:
            dataset: A Hugging Face dataset
            n_repeats: Number of times to repeat each row
        
        Returns:
            Expanded dataset
        """        
        expanded_data = {feature: [] for feature in dataset.features}
        for example in dataset:
            for _ in range(n):
                example_copy = copy.deepcopy(example)                
                for feature in example:
                    expanded_data[feature].append(example_copy[feature])

        return datasets.Dataset.from_dict(expanded_data)

    df = pd.read_parquet(config.data.path)
    ds = datasets.Dataset.from_pandas(df)
    n_samples = config.data.n_samples
    new_ds = _expand_dataset(ds, n_samples)
    print(f"A total of {len(new_ds)} samples")
    print(f"Sample prompt: {new_ds[0]['prompt']}")
    return new_ds


@ray.remote(num_cpus=1)
def main_task(config):
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    revision = config.model.revision
    if revision is not None:
        revision = str(revision)
    local_path = copy_to_local(config.model.path)
    trust_remote_code = config.data.get("trust_remote_code", False)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code, revision=revision)

    if config.rollout.temperature == 0.0:
        assert config.data.n_samples == 1, "When temperature=0, n_samples must be 1."
    assert config.data.n_samples >= 1, "n_samples should always >= 1"

    # read dataset. Note that the dataset should directly contain chat template format (e.g., a list of dictionary)
    ds = load_data(config)
    chat_lst = ds[config.data.prompt_key]

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=config, role="rollout")
    resource_pool = RayResourcePool(process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes)
    wg = RayWorkerGroup(
        resource_pool=resource_pool,
        ray_cls_with_init=ray_cls_with_init,
        device_name=config.trainer.device,
    )
    wg.init_model()

    total_samples = len(ds)
    config_batch_size = config.data.batch_size
    num_batch = -(-total_samples // config_batch_size)
    output_lst = []    

    # enable thinking for hybrid models
    enable_thinking = config.data.get("enable_thinking", False)
    for batch_idx in tqdm(range(num_batch)):
        print(f"[{batch_idx + 1}/{num_batch}] Start to process.")
        batch_chat_lst = chat_lst[batch_idx * config_batch_size : (batch_idx + 1) * config_batch_size]
        if config.data.use_chat_template:
            inputs = tokenizer.apply_chat_template(
                batch_chat_lst,
                add_generation_prompt=True,
                padding=True,
                truncation=True,
                max_length=config.rollout.prompt_length,
                return_tensors="pt",
                return_dict=True,
                tokenize=True,
                enable_thinking=enable_thinking
            )
        else:
            inputs = tokenizer(batch_chat_lst, return_tensors="pt", padding=True, truncation=True, max_length=config.rollout.prompt_length)
        
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        position_ids = compute_position_id_with_mask(attention_mask)
        batch_dict = {"input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids}

        data = DataProto.from_dict(batch_dict)
        data_padded, pad_size = pad_dataproto_to_divisor(data, wg.world_size)
        # START TO GENERATE FOR n_samples TIMES
        print(f"[{batch_idx + 1}/{num_batch}] Start to generate.")        
        output_padded = wg.generate_sequences(data_padded)
        output = unpad_dataproto(output_padded, pad_size=pad_size)

        for i in range(len(output.batch)):
            data_item = output[i]
            prompt_length = data_item.batch["prompts"].shape[-1]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = data_item.batch["responses"][:valid_response_length]
            response_str = tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            output_lst.append(response_str)
    
    # add to the data frame
    ds = ds.add_column(config.data.new_column_name, output_lst)
    ds.push_to_hub(config.data.output_path, private=False)

if __name__ == "__main__":
    main()
