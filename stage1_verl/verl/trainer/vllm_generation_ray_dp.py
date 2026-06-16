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
Ray-based data parallel vLLM generation script.
Spawns multiple vLLM instances across GPUs for parallel processing.
"""

import json
import os
import shutil
from pathlib import Path
from typing import List, Optional

import hydra
from omegaconf import OmegaConf, DictConfig
from pprint import pprint
from tqdm import tqdm

import pandas as pd
import pyarrow.parquet as pq
import datasets
from datasets import Dataset

import ray


def _read_parquet_safe(path) -> pd.DataFrame:
    """Read parquet via per-batch iteration. Avoids pyarrow's dataset
    Scanner.to_table() path, which raises ArrowNotImplementedError
    ("Nested data conversions not implemented for chunked array outputs")
    when coalescing multi-batch scans over list<struct> / struct columns.
    Hits us on LCB v6 chunks (multi-GB, 32K rows, nested prompt+extra_info)
    where the scanner produces multiple RecordBatches and tries to combine
    them. iter_batches bypasses Scanner.to_table entirely."""
    pf = pq.ParquetFile(str(path))
    return pd.concat(
        [b.to_pandas() for b in pf.iter_batches(batch_size=64)],
        ignore_index=True,
    )
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


def load_data(config: DictConfig) -> datasets.Dataset:
    """Load the prompt dataset. n_samples is delegated to vLLM SamplingParams.n
    rather than expanded in Python — for LCB v6 (1055 prompts × ~MB GT blob)
    a 16x Python-level expansion previously blew up host RAM to ~250 GB.

    Uses `datasets.Dataset.from_parquet` (Arrow-native) rather than pandas.
    Going through pandas crashes on parquet files that combine nested types
    (list<struct>, struct) with multiple row groups —
    `ArrowNotImplementedError: Nested data conversions not implemented for
    chunked array outputs` — which is the case for LCB v6 (46 row groups,
    list<struct> prompt + struct reward_model)."""
    ds = datasets.Dataset.from_parquet(config.data.path)
    max_rows = config.data.get("max_rows", None)
    if max_rows is not None and max_rows > 0 and max_rows < len(ds):
        ds = ds.select(range(int(max_rows)))
        print(f"max_rows={max_rows} — truncating dataset")
    print(f"Total prompts: {len(ds)} (each will generate n_samples={config.data.n_samples} via vLLM)")
    sample = ds[0][config.data.prompt_key]
    if isinstance(sample, list) and sample and isinstance(sample[0], dict) and "content" in sample[0]:
        print(f"Sample prompt[0].content: {sample[0]['content'][:200]}...")
    else:
        print(f"Sample prompt: {str(sample)[:200]}...")
    return ds


def prepare_prompts(
    tokenizer: AutoTokenizer,
    chat_lst: List,
    config: DictConfig,
) -> List[str]:
    """Apply chat template to prompts if needed."""
    enable_thinking = config.data.get("enable_thinking", None)

    if config.data.use_chat_template:
        # Apply chat template and decode back to strings for vLLM
        prompts = []
        for i, chat in enumerate(chat_lst):
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


@ray.remote(num_cpus=1)
def score_row(data_source: str, response: str, ground_truth, extra_info=None) -> float:
    """Reward for one (response, ground_truth) pair. Runs in a dedicated CPU
    process so scoring never competes with the GPU workers and many rows can
    be scored in parallel (critical for LiveCodeBench: each call may spawn a
    subprocess with a 6s timeout). `extra_info` is forwarded to scorers that
    need contextual fields beyond ground_truth (e.g. SciKnowEval's LLM judge
    needs the question text)."""
    from verl.utils.reward_score import default_compute_score

    try:
        res = default_compute_score(data_source, response, ground_truth, extra_info=extra_info)
    except Exception as e:
        print(f"[score_row] error for data_source={data_source}: {e}")
        return 0.0
    if isinstance(res, dict):
        return float(res.get("score", res.get("reward", 0.0)))
    return float(res)


@ray.remote(num_gpus=1)
class VLLMWorker:
    """Ray actor that runs vLLM generation on a single GPU."""

    def __init__(self, worker_id: int, config: DictConfig):
        """Initialize the vLLM worker.

        Args:
            worker_id: Unique ID for this worker (used for logging and file naming)
            config: Full configuration object
        """
        self.worker_id = worker_id
        self.config = config

        # Load tokenizer
        revision = config.model.revision
        if revision is not None:
            revision = str(revision)

        trust_remote_code = config.data.get("trust_remote_code", False)
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model.path,
            trust_remote_code=trust_remote_code,
            revision=revision,
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Initialize vLLM - each worker gets its own instance
        print(f"[Worker {worker_id}] Initializing vLLM on GPU...")
        self.llm = LLM(
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
        print(f"[Worker {worker_id}] vLLM initialized successfully!")

        # Setup sampling parameters. vLLM generates `n` completions per prompt
        # with KV-cache sharing across the n branches — no need to expand the
        # dataset in Python.
        self.sampling_params = SamplingParams(
            temperature=config.rollout.temperature,
            top_p=config.rollout.get("top_p", 1.0),
            top_k=config.rollout.get("top_k", -1),
            max_tokens=config.rollout.response_length,
            ignore_eos=config.rollout.get("ignore_eos", False),
            skip_special_tokens=False,
            n=int(config.data.n_samples),
        )
        print(f"[Worker {worker_id}] Sampling params: {self.sampling_params}")

    def generate_batch(
        self,
        batch_data: dict,
        batch_idx: int,
    ) -> List[dict]:
        """Generate responses for a batch of prompts.

        Args:
            batch_data: Dictionary containing batch of data with features
            batch_idx: Index of this batch (for logging)

        Returns:
            List of dictionaries with original data + generated responses
        """
        # Extract prompts
        batch_size = len(batch_data[self.config.data.prompt_key])
        chat_lst = batch_data[self.config.data.prompt_key]

        # Prepare prompts
        prompts = prepare_prompts(self.tokenizer, chat_lst, self.config)

        n_per_prompt = self.sampling_params.n
        print(
            f"[Worker {self.worker_id}] Batch {batch_idx}: "
            f"Generating {len(prompts)} prompts × {n_per_prompt} samples..."
        )

        # Generate
        outputs = self.llm.generate(prompts, self.sampling_params)

        # Collect results — each prompt produces sampling_params.n completions,
        # emit one row per (prompt, sample). Order matches the previous
        # _expand_dataset behavior (consecutive samples per prompt).
        results = []
        new_column_name = self.config.data.get("new_column_name", "response")

        for i, output in enumerate(outputs):
            base_row = {k: batch_data[k][i] for k in batch_data.keys()}
            for completion in output.outputs:
                row = dict(base_row)  # shallow copy — heavy values shared
                row[new_column_name] = completion.text
                row["has_thinking"] = "</think>" in completion.text
                results.append(row)

        print(f"[Worker {self.worker_id}] Batch {batch_idx}: Completed {len(results)} samples")
        return results


def shard_dataset(ds: datasets.Dataset, num_workers: int) -> List[dict]:
    """Shard dataset across workers.

    Args:
        ds: Full dataset
        num_workers: Number of workers to shard across

    Returns:
        List of dataset shards (as dictionaries) for each worker
    """
    total_samples = len(ds)
    shard_size = (total_samples + num_workers - 1) // num_workers  # Ceiling division

    shards = []
    for worker_id in range(num_workers):
        start_idx = worker_id * shard_size
        end_idx = min((worker_id + 1) * shard_size, total_samples)

        if start_idx >= total_samples:
            # This worker gets an empty shard
            shard = {feature: [] for feature in ds.features}
        else:
            # Select the shard and convert to dict
            shard_ds = ds.select(range(start_idx, end_idx))
            shard = {feature: shard_ds[feature] for feature in shard_ds.features}

        print(f"Worker {worker_id}: {len(shard[list(shard.keys())[0]])} samples (indices {start_idx}:{end_idx})")
        shards.append(shard)

    return shards


def run_generation(config: DictConfig) -> None:
    """Main generation function using Ray data parallelism."""
    pprint(OmegaConf.to_container(config, resolve=True))
    OmegaConf.resolve(config)
    
    output_dir = Path(config.data.get("chunk_output_dir", "./generation_chunks"))
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    if config.rollout.temperature == 0.0:
        assert config.data.n_samples == 1, "When temperature=0, n_samples must be 1."
    assert config.data.n_samples >= 1, "n_samples should always >= 1"

    scoring_cfg = config.get("scoring", None)
    scoring_enabled = bool(scoring_cfg and scoring_cfg.get("enabled", False))
    ds_key = "data_source"
    gt_key = "reward_model"
    gt_field = "ground_truth"
    if scoring_enabled:
        ds_key = scoring_cfg.get("data_source_key", ds_key)
        gt_key = scoring_cfg.get("ground_truth_key", gt_key)
        gt_field = scoring_cfg.get("ground_truth_field", gt_field)
        print(
            f"Scoring enabled: ds={ds_key}, ground_truth=row[{gt_key}][{gt_field}]"
        )
    
    ds = load_data(config)
    num_workers = config.rollout.get("data_parallel_size", 1)
    print(f"\nUsing {num_workers} data parallel workers")
    if not ray.is_initialized():
        ray.init()
        print("Ray initialized")

    # Create workers
    print(f"Creating {num_workers} vLLM workers...")
    workers = [VLLMWorker.remote(worker_id=i, config=config) for i in range(num_workers)]
    print(f"All {num_workers} workers created")

    # Shard dataset across workers
    print(f"\nSharding dataset across {num_workers} workers...")
    shards = shard_dataset(ds, num_workers)

    # Process each shard with its worker
    batch_size = config.data.batch_size
    all_futures = []

    for worker_id, (worker, shard) in enumerate(zip(workers, shards)):
        shard_size = len(shard[list(shard.keys())[0]])
        if shard_size == 0:
            print(f"Worker {worker_id}: Empty shard, skipping")
            continue

        num_batches = (shard_size + batch_size - 1) // batch_size
        print(f"Worker {worker_id}: Processing {shard_size} samples in {num_batches} batches")

        # Submit all batches for this worker
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, shard_size)

            # Extract batch from shard
            batch_data = {k: v[start_idx:end_idx] for k, v in shard.items()}

            # Submit batch generation task
            future = worker.generate_batch.remote(batch_data, batch_idx)
            all_futures.append((worker_id, batch_idx, future))

    print(f"\nSubmitted {len(all_futures)} batches across {num_workers} workers. Waiting for results...")

    response_col = config.data.get("new_column_name", "response")
    score_jobs: List[tuple] = []  # list of (chunk_path, [score futures in row order])

    # Collect results as they complete
    completed = 0
    with tqdm(total=len(all_futures), desc="Generating") as pbar:
        while all_futures:
            # Wait for any batch to complete
            ready_futures = [f[2] for f in all_futures]
            ready, _ = ray.wait(ready_futures, num_returns=1, timeout=None)

            if ready:
                ready_ref = ready[0]
                # Find which batch completed
                for i, (worker_id, batch_idx, future_ref) in enumerate(all_futures):
                    if future_ref == ready_ref:
                        # Get result
                        batch_results = ray.get(ready_ref)

                        # Save chunk immediately. Keep `df` (in memory) full so
                        # the scoring fan-out below can still pull GT — but drop
                        # heavy/nested columns from what we write to disk. For
                        # LCB v6, `reward_model.ground_truth` carries MB-scale
                        # base64+zlib test cases for some problems, replicated
                        # n_samples times per prompt; chunks balloon to 10-22 GB
                        # each and trip pyarrow's "Nested data conversions not
                        # implemented for chunked array outputs" on read-back.
                        chunk_path = output_dir / f"worker{worker_id}_batch{batch_idx:05d}.parquet"
                        df = pd.DataFrame(batch_results)
                        drop_cols = list(config.data.get("chunk_drop_cols", ["reward_model"]))
                        df_disk = df.drop(columns=[c for c in drop_cols if c in df.columns])
                        df_disk.to_parquet(chunk_path, index=False)
                        del df_disk

                        # Fan out per-row scoring as soon as the chunk lands;
                        # tasks run in parallel with subsequent generation.
                        if scoring_enabled:
                            row_futures = []
                            for _, row in df.iterrows():
                                gt_obj = row[gt_key]
                                gt_val = gt_obj[gt_field] if gt_obj is not None else None
                                ei = row.get("extra_info") if "extra_info" in df.columns else None
                                # numpy/pandas may surface dict-likes as objects;
                                # cast to plain dict for clean Ray serialization.
                                if ei is not None and not isinstance(ei, dict):
                                    ei = dict(ei)
                                row_futures.append(
                                    score_row.remote(row[ds_key], row[response_col], gt_val, ei)
                                )
                            score_jobs.append((chunk_path, row_futures))

                        # Remove from pending list
                        all_futures.pop(i)
                        completed += 1
                        pbar.update(1)
                        break

    print(f"\nAll {completed} batches completed!")

    # Drain scoring tasks (most should already be done since they ran during gen)
    if scoring_enabled and score_jobs:
        total_score = sum(len(fs) for _, fs in score_jobs)
        print(f"\nDraining {total_score} scoring tasks...")

        flat_futures = []
        locations = {}
        results_by_chunk = {}
        for cpath, futs in score_jobs:
            results_by_chunk[cpath] = [None] * len(futs)
            for ridx, f in enumerate(futs):
                flat_futures.append(f)
                locations[f] = (cpath, ridx)

        with tqdm(total=total_score, desc="Scoring") as spbar:
            while flat_futures:
                n_get = min(64, len(flat_futures))
                ready, flat_futures = ray.wait(flat_futures, num_returns=n_get, timeout=None)
                for fref in ready:
                    cp, ridx = locations[fref]
                    results_by_chunk[cp][ridx] = ray.get(fref)
                    spbar.update(1)

        print("Writing scores back to chunk files...")
        for cpath, _ in score_jobs:
            df = _read_parquet_safe(cpath)
            df["score"] = results_by_chunk[cpath]
            df.to_parquet(cpath, index=False)

    # Merge all chunks and push to hub
    print("\nMerging chunks...")
    chunk_files = sorted(output_dir.glob("worker*_batch*.parquet"))
    print(f"Found {len(chunk_files)} chunk files")

    dfs = []
    for chunk_file in tqdm(chunk_files, desc="Loading chunks"):
        df = _read_parquet_safe(chunk_file)
        dfs.append(df)

    merged_df = pd.concat(dfs, ignore_index=True)
    print(f"Total merged samples: {len(merged_df)}")

    if scoring_enabled and "score" in merged_df.columns:
        n_per_prompt = config.data.n_samples
        metrics = {}
        print("\n=== Eval metrics ===")
        for ds_val, group in merged_df.groupby(ds_key):
            pass_at_1 = float(group["score"].mean())
            pass_at_n: Optional[float] = None
            n_prompts: Optional[int] = None
            if "extra_info" in group.columns:
                def _idx(x):
                    return x.get("index") if isinstance(x, dict) else None
                prompt_idx = group["extra_info"].apply(_idx)
                valid = prompt_idx.notna()
                if valid.any():
                    per_prompt_max = group[valid].groupby(prompt_idx[valid])["score"].max()
                    pass_at_n = float(per_prompt_max.mean())
                    n_prompts = int(len(per_prompt_max))
            metrics[str(ds_val)] = {
                "pass@1": pass_at_1,
                f"pass@{n_per_prompt}": pass_at_n,
                "n_prompts": n_prompts,
                "n_total": int(len(group)),
            }
            print(
                f"  {ds_val}: pass@1={pass_at_1:.4f}"
                f"{f', pass@{n_per_prompt}={pass_at_n:.4f}' if pass_at_n is not None else ''}"
                f", prompts={n_prompts}, total={len(group)}"
            )
        metrics_filename = (scoring_cfg or {}).get("metrics_filename", "metrics.json")
        metrics_path = output_dir / metrics_filename
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved metrics to {metrics_path}")

    # Convert to dataset and push
    ds_output = Dataset.from_pandas(merged_df)

    output_path = config.data.output_path
    print(f"Pushing to HuggingFace Hub: {output_path}")
    ds_output.push_to_hub(output_path, private=False)
    print("Push complete!")

    # Save merged file locally
    merged_path = output_dir / "merged_output.parquet"
    merged_df.to_parquet(merged_path, index=False)
    print(f"Saved merged file to {merged_path}")

    # Cleanup chunks if requested
    if config.data.get("cleanup_chunks", False):
        print(f"Cleaning up chunk directory: {output_dir}")
        shutil.rmtree(output_dir)

    # Shutdown Ray
    ray.shutdown()
    print("\nAll done!")


@hydra.main(config_path="config", config_name="vllm_generation", version_base=None)
def main(config: DictConfig) -> None:
    run_generation(config)


if __name__ == "__main__":
    main()
