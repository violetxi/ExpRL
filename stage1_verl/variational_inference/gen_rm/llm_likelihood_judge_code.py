#!/usr/bin/env python3
"""
llm_likelihood_judge_code.py – Batch evaluator for coding reasoning traces.

Coding analog of llm_likelihood_judge.py. The math version pulls the reference
solution from `extra_info["solution"]` in the same row as the response; LCB v6
generations don't carry that field (the slim hub dataset has prompt_idx +
response + score only), so we join against a separate oracle parquet that maps
prompt_idx -> (prompt, oracle solution_code_only).

Local inference runs through the offline DP+Ray vLLM runner
(_judge_runner.py), the same pattern as verl/trainer/vllm_generation_ray_dp.py.

Example:
    python variational_inference/gen_rm/llm_likelihood_judge_code.py \\
        --data violetxi/judge_calibration_lcb_v5_qwen8b \\
        --oracle data/instruct/livecodebench_v6_with_oracle/train.parquet \\
        --model qwen/qwen3_4b_instruct \\
        --use_local_model --multi_ref \\
        --output violetxi/Qwen4b-Instruct-Judge-lcb-v6-calibration-multi-ref
"""

import argparse
import asyncio
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pyarrow.parquet as pq
from datasets import Dataset, load_dataset

from verl.utils.prompt_templates.code_likelihood_judge import (
    SYSTEM_PROMPT as SYSTEM_PROMPT_REF,
    USER_PROMPT as USER_PROMPT_REF,
)
from verl.utils.prompt_templates.code_likelihood_judge_no_ref import (
    SYSTEM_PROMPT as SYSTEM_PROMPT_NOREF,
    USER_PROMPT as USER_PROMPT_NOREF,
)

from variational_inference.gen_rm._judge_runner import (
    parse_score_likert,
    run_judge_dp,
)

# Back-compat aliases.
SYSTEM_PROMPT = SYSTEM_PROMPT_REF
USER_PROMPT = USER_PROMPT_REF

CONDITIONS = ("ref", "noref", "wrongref")
WRONG_REF_SEED = 0xC0DE


def build_messages(
    condition: str,
    problem: str,
    generated_response: str,
    reference_solution: str,
    wrong_reference_solution: str,
) -> List[Dict[str, str]]:
    if condition == "ref":
        sys_p = SYSTEM_PROMPT_REF
        user_p = USER_PROMPT_REF.format(
            problem=problem,
            generated_response=generated_response,
            reference_solution=reference_solution,
        )
    elif condition == "noref":
        sys_p = SYSTEM_PROMPT_NOREF
        user_p = USER_PROMPT_NOREF.format(
            problem=problem,
            generated_response=generated_response,
        )
    elif condition == "wrongref":
        sys_p = SYSTEM_PROMPT_REF
        user_p = USER_PROMPT_REF.format(
            problem=problem,
            generated_response=generated_response,
            reference_solution=wrong_reference_solution,
        )
    else:
        raise ValueError(f"unknown condition: {condition!r}")
    return [{"role": "system", "content": sys_p},
            {"role": "user",   "content": user_p}]


def build_wrong_ref_map(oracle_keys: List[int], seed: int = WRONG_REF_SEED) -> Dict[int, int]:
    keys = sorted(int(k) for k in oracle_keys)
    if len(keys) < 2:
        raise ValueError("wrong-reference requires at least 2 oracle prompts")
    rng = random.Random(seed)
    shuffled = keys.copy()
    rng.shuffle(shuffled)
    pos = {k: i for i, k in enumerate(shuffled)}
    n = len(shuffled)
    out: Dict[int, int] = {}
    for k in keys:
        i = pos[k]
        twin = shuffled[(i + 1) % n]
        if twin == k:
            twin = shuffled[(i + 2) % n]
        out[k] = twin
    return out


def _read_parquet_safe(path) -> pd.DataFrame:
    """Per-batch read sidesteps pyarrow Scanner.to_table failures on the LCB
    v6 oracle (46 row groups, nested list<struct>/struct columns)."""
    pf = pq.ParquetFile(str(path))
    return pd.concat(
        [b.to_pandas() for b in pf.iter_batches(batch_size=64)],
        ignore_index=True,
    )


def load_oracle(oracle_path: Path, reference_field: str = "solution_code_only") -> Dict[int, Tuple[str, str]]:
    assert reference_field in ("solution_code_only", "solution"), \
        f"unsupported reference_field={reference_field}"
    df = _read_parquet_safe(oracle_path)
    lookup: Dict[int, Tuple[str, str]] = {}
    for _, row in df.iterrows():
        ei = row["extra_info"]
        idx = int(ei["index"])
        problem = row["prompt"][0]["content"]
        ref = ei.get(reference_field) or ""
        lookup[idx] = (problem, str(ref))
    print(f"Loaded oracle: {len(lookup)} prompt_idx -> (problem, reference_solution) "
          f"[reference_field={reference_field}]")
    return lookup


def load_jobs(data_path: Path, oracle_path: Path,
              max_prompts: int = 0, max_samples_per_prompt: int = 0,
              reference_field: str = "solution_code_only",
              multi_ref: bool = False):
    oracle = load_oracle(oracle_path, reference_field=reference_field)
    dataset = load_dataset(data_path, split="train")
    print(f"Loaded generations: {len(dataset)} rows from {data_path}")

    wrong_ref_map: Dict[int, int] = {}
    if multi_ref:
        wrong_ref_map = build_wrong_ref_map(list(oracle.keys()))
        print(f"Built wrong-ref map for {len(wrong_ref_map)} prompts "
              f"(seed={hex(WRONG_REF_SEED)})")

    jobs: List[Tuple[str, str, str, str]] = []
    output_data = {
        "prompt_idx": [],
        "problem": [],
        "generated_response": [],
        "reference_solution": [],
        "reward": [],
    }
    if multi_ref:
        output_data["wrong_reference_prompt_idx"] = []
        output_data["wrong_reference_solution"] = []

    all_responses = dataset["responses"] if "responses" in dataset.column_names else dataset["response"]
    # Resolve prompt index: prefer top-level `prompt_idx` (older slim-hub format),
    # fall back to `extra_info["index"]` (the schema vllm_generation_ray_dp writes).
    if "prompt_idx" in dataset.column_names:
        prompt_indices = dataset["prompt_idx"]
    else:
        prompt_indices = [int(ei["index"]) for ei in dataset["extra_info"]]
        print("No top-level 'prompt_idx' column; using extra_info['index'] instead.")
    skipped = 0
    prompt_order: List[int] = []
    seen_prompts: set = set()
    per_prompt_count: Dict[int, int] = {}
    for prompt_idx, response, reward in zip(
        prompt_indices, all_responses, dataset["score"]
    ):
        idx = int(prompt_idx)
        if idx not in oracle:
            skipped += 1
            continue
        problem, reference_solution = oracle[idx]
        if not reference_solution:
            skipped += 1
            continue

        if idx not in seen_prompts:
            if max_prompts and len(prompt_order) >= max_prompts:
                continue
            seen_prompts.add(idx)
            prompt_order.append(idx)
        if max_samples_per_prompt and per_prompt_count.get(idx, 0) >= max_samples_per_prompt:
            continue
        per_prompt_count[idx] = per_prompt_count.get(idx, 0) + 1

        if multi_ref:
            wrong_idx = wrong_ref_map[idx]
            _, wrong_reference_solution = oracle[wrong_idx]
            jobs.append((problem, response, reference_solution, wrong_reference_solution))
            output_data["wrong_reference_prompt_idx"].append(wrong_idx)
            output_data["wrong_reference_solution"].append(wrong_reference_solution)
        else:
            jobs.append((problem, response, reference_solution, ""))

        output_data["prompt_idx"].append(idx)
        output_data["problem"].append(problem)
        output_data["generated_response"].append(response)
        output_data["reference_solution"].append(reference_solution)
        output_data["reward"].append(reward)

    if skipped:
        print(f"Skipped {skipped} rows lacking an oracle reference solution")
    print(f"Will judge {len(jobs)} rows (across {len(prompt_order)} unique prompts)"
          + (f" x {len(CONDITIONS)} conditions = {len(jobs)*len(CONDITIONS)} LLM calls" if multi_ref else ""))
    return jobs, Dataset.from_dict(output_data)


# ---------------- hosted async fallback (OpenRouter / OpenAI) ----------------

async def _hosted_call(client, system_prompt, user_msg, model, sem, temperature, max_tokens):
    async with sem:
        try:
            r = await client.chat.completions.create(
                model=model, temperature=temperature, n=1, max_tokens=max_tokens,
                messages=[{"role": "system", "content": system_prompt},
                          {"role": "user",   "content": user_msg}],
            )
            return r.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error: {e}")
            return ""


async def _hosted_batch(messages_list, model, temperature, max_tokens, max_concurrency, request_timeout):
    from openai import AsyncOpenAI
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENROUTER_API_KEY env var is not set.")
    client = AsyncOpenAI(api_key=api_key, base_url="https://openrouter.ai/api/v1",
                        timeout=request_timeout, max_retries=2)
    sem = asyncio.Semaphore(max_concurrency)
    coros = [
        _hosted_call(client, m[0]["content"], m[1]["content"], model, sem, temperature, max_tokens)
        for m in messages_list
    ]
    return await asyncio.gather(*coros)


def run_inference(args, messages_list: List[List[Dict[str, str]]]) -> List[str]:
    if args.use_local_model:
        return run_judge_dp(
            model_name=args.model,
            messages_list=messages_list,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            data_parallel_size=args.data_parallel_size,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
            max_num_batched_tokens=args.max_num_batched_tokens,
            max_num_seqs=args.max_num_seqs,
        )
    return asyncio.run(_hosted_batch(
        messages_list, args.model, args.temperature, args.max_tokens,
        args.max_concurrency, args.request_timeout,
    ))


def main(args):
    jobs, output_ds = load_jobs(
        args.data, args.oracle,
        max_prompts=args.max_prompts,
        max_samples_per_prompt=args.max_samples_per_prompt,
        reference_field=args.reference_field,
        multi_ref=args.multi_ref,
    )

    if args.multi_ref:
        all_messages: List[List[Dict[str, str]]] = []
        for problem, generated_response, reference, wrong_reference in jobs:
            for cond in CONDITIONS:
                all_messages.append(build_messages(
                    cond, problem, generated_response, reference, wrong_reference,
                ))
        responses = run_inference(args, all_messages)
        scores = [parse_score_likert(r) for r in responses]

        per_cond_responses = {c: [] for c in CONDITIONS}
        per_cond_scores = {c: [] for c in CONDITIONS}
        for k, (r, s) in enumerate(zip(responses, scores)):
            cond = CONDITIONS[k % len(CONDITIONS)]
            per_cond_responses[cond].append(r)
            per_cond_scores[cond].append(s)

        for cond in CONDITIONS:
            output_ds = output_ds.add_column(f"llm_score_{cond}", per_cond_scores[cond])
            output_ds = output_ds.add_column(f"llm_response_{cond}", per_cond_responses[cond])
    else:
        all_messages = [
            build_messages("ref", problem, generated_response, reference, "")
            for problem, generated_response, reference, _ in jobs
        ]
        responses = run_inference(args, all_messages)
        scores = [parse_score_likert(r) for r in responses]
        output_ds = output_ds.add_column("llm_score", scores)
        output_ds = output_ds.add_column("llm_response", responses)

    print(f"Pushing to HuggingFace Hub: {args.output}")
    output_ds.push_to_hub(args.output, private=False)
    print("Done.")


def cli():
    p = argparse.ArgumentParser(description="Batch LLM likelihood evaluator for coding reasoning traces.")
    p.add_argument("--data", required=True, help="HF dataset of generations")
    p.add_argument("--oracle", required=True, type=Path,
                   help="Local parquet with prompt + extra_info.solution_code_only")
    p.add_argument("--model", default="qwen/qwen3_4b_instruct")
    p.add_argument("--use_local_model", action="store_true",
                   help="Use offline DP+Ray vLLM (recommended)")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_tokens", type=int, default=4096)
    p.add_argument("--max_prompts", type=int, default=0)
    p.add_argument("--max_samples_per_prompt", type=int, default=0)
    p.add_argument("--output", required=True)
    p.add_argument("--reference_field", choices=["solution_code_only", "solution"],
                   default="solution_code_only")
    p.add_argument("--multi_ref", action="store_true",
                   help="Run ref/noref/wrongref per row")
    # Local DP+Ray knobs
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--data_parallel_size", type=int, default=None,
                   help="Default: len(CUDA_VISIBLE_DEVICES)//TP")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--max_model_len", type=int, default=None)
    p.add_argument("--max_num_batched_tokens", type=int, default=32768)
    p.add_argument("--max_num_seqs", type=int, default=2048)
    # Hosted API fallback knobs
    p.add_argument("--max-concurrency", type=int, default=256, dest="max_concurrency")
    p.add_argument("--request_timeout", type=float, default=1200.0)
    # Legacy flags (accepted but ignored)
    p.add_argument("--launch_model", action="store_true",
                   help="(deprecated) ignored — offline runner self-launches")
    return p.parse_args()


if __name__ == "__main__":
    main(cli())
