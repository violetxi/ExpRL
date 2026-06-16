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
Standalone vLLM generation script - lightweight alternative to Ray-based generation.
Saves outputs incrementally to avoid OOM issues.
"""

import os
import copy
import shutil
from pathlib import Path
from typing import List, Optional

import hydra
from omegaconf import OmegaConf, DictConfig
from pprint import pprint
from tqdm import tqdm

import pandas as pd
import datasets
from datasets import Dataset, concatenate_datasets

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


def load_data(config: DictConfig) -> datasets.Dataset:
    """Load and expand dataset by n_samples."""

    def _expand_dataset(dataset: datasets.Dataset, n: int) -> datasets.Dataset:
        """
        Expands a Hugging Face dataset by repeating each row n times.
        """
        if n == 1:
            return dataset

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
    print(f"Total samples after expansion: {len(new_ds)} ({len(ds)} prompts × {n_samples} samples)")
    print(f"Sample prompt: {new_ds[0][config.data.prompt_key][:200]}...")
    return new_ds


def prepare_prompts(
    tokenizer: AutoTokenizer,
    chat_lst: List,
    config: DictConfig,
) -> List[str]:
    """Apply chat template to prompts if needed."""
    enable_thinking = config.data.get("enable_thinking", False)

    if config.data.use_chat_template:
        # Apply chat template and decode back to strings for vLLM
        prompts = []
        for chat in chat_lst:
            prompt = tokenizer.apply_chat_template(
                chat,
                add_generation_prompt=True,
                tokenize=False,
                enable_thinking=enable_thinking,
            )
            prompts.append(prompt)
        return prompts
    else:
        # Use raw prompts
        return chat_lst


def save_chunk(
    chunk_data: List[dict],
    chunk_idx: int,
    output_dir: Path,
    config: DictConfig,
) -> Path:
    """Save a chunk of results to a local parquet file."""
    chunk_path = output_dir / f"chunk_{chunk_idx:05d}.parquet"
    df = pd.DataFrame(chunk_data)
    df.to_parquet(chunk_path, index=False)
    print(f"Saved chunk {chunk_idx} with {len(chunk_data)} samples to {chunk_path}")
    return chunk_path


def merge_and_push(
    output_dir: Path,
    config: DictConfig,
) -> None:
    """Merge all chunks and push to HuggingFace Hub."""
    chunk_files = sorted(output_dir.glob("chunk_*.parquet"))
    print(f"Merging {len(chunk_files)} chunks...")

    # Load and concatenate all chunks
    dfs = []
    for chunk_file in tqdm(chunk_files, desc="Loading chunks"):
        df = pd.read_parquet(chunk_file)
        dfs.append(df)

    merged_df = pd.concat(dfs, ignore_index=True)
    print(f"Total merged samples: {len(merged_df)}")

    # Convert to dataset and push
    ds = Dataset.from_pandas(merged_df)

    output_path = config.data.output_path
    print(f"Pushing to HuggingFace Hub: {output_path}")
    ds.push_to_hub(output_path, private=False)
    print("Push complete!")

    # Optionally save merged file locally
    merged_path = output_dir / "merged_output.parquet"
    merged_df.to_parquet(merged_path, index=False)
    print(f"Also saved merged file to {merged_path}")


def run_generation(config: DictConfig) -> None:
    """Main generation function using standalone vLLM."""
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)

    # Setup output directory for chunks
    output_dir = Path(config.data.get("chunk_output_dir", "./generation_chunks"))
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Chunk output directory: {output_dir}")

    # Load tokenizer
    revision = config.model.revision
    if revision is not None:
        revision = str(revision)

    trust_remote_code = config.data.get("trust_remote_code", False)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.path,
        trust_remote_code=trust_remote_code,
        revision=revision,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Validation
    if config.rollout.temperature == 0.0:
        assert config.data.n_samples == 1, "When temperature=0, n_samples must be 1."
    assert config.data.n_samples >= 1, "n_samples should always >= 1"

    # Load dataset
    ds = load_data(config)
    chat_lst = ds[config.data.prompt_key]

    # Initialize vLLM
    print("Initializing vLLM...")
    llm = LLM(
        model=config.model.path,
        revision=revision,
        tokenizer=config.model.path,
        tokenizer_revision=revision,
        trust_remote_code=trust_remote_code,
        tensor_parallel_size=config.rollout.tensor_model_parallel_size,
        dtype=config.rollout.dtype,
        gpu_memory_utilization=config.rollout.gpu_memory_utilization,
        max_model_len=config.rollout.get("max_model_len", None),
        max_num_batched_tokens=config.rollout.max_num_batched_tokens,
        max_num_seqs=config.rollout.max_num_seqs,
        enforce_eager=config.rollout.get("enforce_eager", False),
        enable_chunked_prefill=config.rollout.get("enable_chunked_prefill", True),
        disable_log_stats=config.rollout.get("disable_log_stats", True),
    )
    print("vLLM initialized successfully!")

    # Setup sampling parameters
    sampling_params = SamplingParams(
        temperature=config.rollout.temperature,
        top_p=config.rollout.get("top_p", 1.0),
        top_k=config.rollout.get("top_k", -1),
        max_tokens=config.rollout.response_length,
        ignore_eos=config.rollout.get("ignore_eos", False),
        n=1,  # We handle n_samples via dataset expansion
    )
    print(f"Sampling params: {sampling_params}")

    # Process in batches
    total_samples = len(ds)
    batch_size = config.data.batch_size
    num_batches = -(-total_samples // batch_size)  # Ceiling division

    print(f"Processing {total_samples} samples in {num_batches} batches (batch_size={batch_size})")

    new_column_name = config.data.get("new_column_name", "response")

    for batch_idx in tqdm(range(num_batches), desc="Generating"):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, total_samples)

        # Get batch data
        batch_ds = ds.select(range(start_idx, end_idx))
        batch_chat_lst = chat_lst[start_idx:end_idx]

        # Prepare prompts
        prompts = prepare_prompts(tokenizer, batch_chat_lst, config)

        print(f"[{batch_idx + 1}/{num_batches}] Generating {len(prompts)} samples...")

        # Generate
        outputs = llm.generate(prompts, sampling_params)

        # Collect results for this chunk
        chunk_data = []
        for i, output in enumerate(outputs):
            # Get the original data for this sample
            original_data = {k: batch_ds[i][k] for k in batch_ds.features}

            # Add generated response
            generated_text = output.outputs[0].text
            original_data[new_column_name] = generated_text

            chunk_data.append(original_data)

        # Save chunk immediately to free memory
        save_chunk(chunk_data, batch_idx, output_dir, config)

        # Clear chunk_data to free memory
        del chunk_data
        del outputs

    print("\nGeneration complete! Merging chunks...")

    # Merge all chunks and push to hub
    merge_and_push(output_dir, config)

    # Cleanup chunks if requested
    if config.data.get("cleanup_chunks", False):
        print(f"Cleaning up chunk directory: {output_dir}")
        shutil.rmtree(output_dir)

    print("All done!")


@hydra.main(config_path="config", config_name="vllm_generation", version_base=None)
def main(config: DictConfig) -> None:
    run_generation(config)


if __name__ == "__main__":
    main()
