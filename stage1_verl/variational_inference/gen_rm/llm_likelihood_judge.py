#!/usr/bin/env python3
"""
llm_likelihood_judge.py – Batch evaluator for math reasoning trace likelihood.

Each input row is expected to follow the verl rollout schema:
    prompt / extra_info / response / score
where prompt[0]["content"] is the problem text and extra_info["solution"] is
the reference solution.

Local model inference goes through the offline DP+Ray vLLM runner
(_judge_runner.py), mirroring verl/trainer/vllm_generation_ray_dp.py — this
sidesteps the V1-engine compile path that trips a triton import in this
conda env.

Example (with-reference only, local 4-GPU node):
    python llm_likelihood_judge.py \\
        --data violetxi/judge_calibration_int_qwen8b \\
        --model qwen/qwen3_4b_instruct \\
        --use_local_model \\
        --tensor_parallel_size 1 \\
        --data_parallel_size 4 \\
        --output violetxi/Qwen4b-Instruct-Judge-int-qwen8b

Example (all 3 conditions, local 4-GPU node):
    python llm_likelihood_judge.py \\
        --data violetxi/judge_calibration_int_qwen8b \\
        --model qwen/qwen3_4b_instruct \\
        --use_local_model \\
        --multi_ref \\
        --output violetxi/Qwen4b-Instruct-Judge-int-qwen8b-multi-ref
"""

import argparse
import asyncio
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from datasets import Dataset, load_dataset
from huggingface_hub import snapshot_download

from verl.utils.prompt_templates.math_likelihood_judge import (
    SYSTEM_PROMPT as SYSTEM_PROMPT_REF,
    USER_PROMPT as USER_PROMPT_REF,
)
from verl.utils.prompt_templates.math_likelihood_judge_no_ref import (
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
    reasoning_trace: str,
    reference_solution: str,
    wrong_reference_solution: str,
) -> List[Dict[str, str]]:
    """Return chat-format [system, user] messages for the given condition."""
    if condition == "ref":
        sys_p = SYSTEM_PROMPT_REF
        user_p = USER_PROMPT_REF.format(
            problem=problem,
            reasoning_trace=reasoning_trace,
            reference_solution=reference_solution,
        )
    elif condition == "noref":
        sys_p = SYSTEM_PROMPT_NOREF
        user_p = USER_PROMPT_NOREF.format(
            problem=problem,
            reasoning_trace=reasoning_trace,
        )
    elif condition == "wrongref":
        sys_p = SYSTEM_PROMPT_REF
        user_p = USER_PROMPT_REF.format(
            problem=problem,
            reasoning_trace=reasoning_trace,
            reference_solution=wrong_reference_solution,
        )
    else:
        raise ValueError(f"unknown condition: {condition!r}")
    return [{"role": "system", "content": sys_p},
            {"role": "user",   "content": user_p}]


def build_wrong_ref_map(unique_keys: List[str], seed: int = WRONG_REF_SEED) -> Dict[str, str]:
    keys = sorted(set(unique_keys))
    if len(keys) < 2:
        raise ValueError("wrong-reference requires at least 2 unique problems")
    rng = random.Random(seed)
    shuffled = keys.copy()
    rng.shuffle(shuffled)
    pos = {k: i for i, k in enumerate(shuffled)}
    n = len(shuffled)
    out: Dict[str, str] = {}
    for k in keys:
        i = pos[k]
        twin = shuffled[(i + 1) % n]
        if twin == k:
            twin = shuffled[(i + 2) % n]
        out[k] = twin
    return out


def _extract_problem(prompt, extra_info: Dict) -> str:
    """Real user question, regardless of whether the dataset puts it in
    extra_info, in a user-role chat message, or directly at prompt[0]."""
    if isinstance(extra_info, dict) and extra_info.get("question"):
        return str(extra_info["question"])
    if isinstance(prompt, list):
        for msg in reversed(prompt):
            if isinstance(msg, dict) and msg.get("role") == "user":
                content = msg.get("content")
                if content:
                    return str(content)
        # Legacy fallback: single-message prompts where prompt[0] *is* the question.
        if prompt and isinstance(prompt[0], dict) and prompt[0].get("content"):
            return str(prompt[0]["content"])
    return ""


def _extract_reference(extra_info: Dict) -> str:
    if isinstance(extra_info, dict):
        sol = extra_info.get("solution")
        if sol is not None and str(sol) != "":
            return str(sol)
        ans = extra_info.get("answer")
        if ans is not None and str(ans) != "":
            return str(ans)
    return ""


def _load_dataset_with_parquet_fallback(data_path: str):
    try:
        return load_dataset(data_path, split="train")
    except Exception as exc:
        print(
            f"load_dataset({data_path!r}) failed; retrying from parquet files "
            f"so scored rollout columns are preserved. Error: {type(exc).__name__}: {exc}"
        )

    local_path = Path(data_path)
    if local_path.exists():
        root = local_path
    else:
        root = Path(
            snapshot_download(
                repo_id=data_path,
                repo_type="dataset",
                allow_patterns=["data/*.parquet", "*.parquet"],
            )
        )

    parquet_files = sorted((root / "data").glob("*.parquet")) if (root / "data").exists() else []
    if not parquet_files:
        parquet_files = sorted(root.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found for {data_path!r} under {root}")
    return load_dataset("parquet", data_files=[str(p) for p in parquet_files], split="train")


def load_jobs(data_path: Path, multi_ref: bool = False):
    dataset = _load_dataset_with_parquet_fallback(str(data_path))
    has_score = "score" in dataset.column_names

    problem_to_ref: Dict[str, str] = {}
    for prompt_info, extra_info in zip(dataset["prompt"], dataset["extra_info"]):
        problem = _extract_problem(prompt_info, extra_info)
        reference_solution = _extract_reference(extra_info)
        if problem and reference_solution and problem not in problem_to_ref:
            problem_to_ref[problem] = reference_solution

    wrong_q_map: Dict[str, str] = {}
    if multi_ref:
        wrong_q_map = build_wrong_ref_map(list(problem_to_ref.keys()))
        print(f"Built wrong-ref map for {len(wrong_q_map)} unique problems "
              f"(seed={hex(WRONG_REF_SEED)})")

    jobs: List[Tuple[str, str, str, str]] = []
    output_data = {
        "problem": [],
        "reasoning_trace": [],
        "reference_solution": [],
        "finish_thinking": [],
        "reward": [],
        "score": [],
    }
    if multi_ref:
        output_data["wrong_reference_problem"] = []
        output_data["wrong_reference_solution"] = []

    all_responses = dataset["responses"] if "responses" in dataset.column_names else dataset["response"]
    rewards = dataset["score"] if has_score else [None] * len(dataset)
    skipped = 0
    for prompt_info, extra_info, response, reward in zip(
        dataset["prompt"], dataset["extra_info"], all_responses, rewards
    ):
        problem = _extract_problem(prompt_info, extra_info)
        reference_solution = _extract_reference(extra_info)
        if not problem or not reference_solution:
            skipped += 1
            continue

        if response is None:
            response = ""
        if "</think>" in response:
            finish_thinking = True
            reasoning_trace = response.split("</think>")[0] + "</think>"
        else:
            finish_thinking = False
            reasoning_trace = response

        if multi_ref:
            wrong_problem = wrong_q_map[problem]
            wrong_reference_solution = problem_to_ref[wrong_problem]
            jobs.append((problem, reasoning_trace, reference_solution, wrong_reference_solution))
            output_data["wrong_reference_problem"].append(wrong_problem)
            output_data["wrong_reference_solution"].append(wrong_reference_solution)
        else:
            jobs.append((problem, reasoning_trace, reference_solution, ""))

        output_data["problem"].append(problem)
        output_data["reasoning_trace"].append(reasoning_trace)
        output_data["reference_solution"].append(reference_solution)
        output_data["finish_thinking"].append(finish_thinking)
        output_data["reward"].append(reward)
        output_data["score"].append(reward)

    if skipped:
        print(f"Skipped {skipped} rows missing problem or reference solution")
    print(f"Loaded {len(jobs)} math rows from {data_path}"
          + (f" x {len(CONDITIONS)} conditions = {len(jobs)*len(CONDITIONS)} LLM calls" if multi_ref else ""))
    return jobs, Dataset.from_dict(output_data)


# ---------------- hosted (OpenRouter / OpenAI) async fallback ----------------
# Preserved for callers that pass --model openai/* or use OPENROUTER_API_KEY.

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
            enable_thinking=args.enable_thinking,
        )
    return asyncio.run(_hosted_batch(
        messages_list, args.model, args.temperature, args.max_tokens,
        args.max_concurrency, args.request_timeout,
    ))


def main(args):
    jobs, output_ds = load_jobs(args.data, multi_ref=args.multi_ref)

    if args.multi_ref:
        all_messages: List[List[Dict[str, str]]] = []
        for problem, reasoning_trace, reference, wrong_reference in jobs:
            for cond in CONDITIONS:
                all_messages.append(build_messages(
                    cond, problem, reasoning_trace, reference, wrong_reference,
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
            build_messages("ref", problem, reasoning_trace, reference, "")
            for problem, reasoning_trace, reference, _ in jobs
        ]
        responses = run_inference(args, all_messages)
        scores = [parse_score_likert(r) for r in responses]
        output_ds = output_ds.add_column("llm_score", scores)
        output_ds = output_ds.add_column("llm_response", responses)

    print(f"Pushing to HuggingFace Hub: {args.output}")
    output_ds.push_to_hub(args.output, private=False)
    print("Done.")


def cli():
    p = argparse.ArgumentParser(description="Batch LLM judge for math reasoning traces.")
    p.add_argument("--data", required=True, help="HF dataset path")
    p.add_argument("--model", default="qwen/qwen3_4b_instruct",
                   help="Model name (short qwen/* for local, or hosted id like openai/gpt-4.1)")
    p.add_argument("--use_local_model", action="store_true",
                   help="Use offline DP+Ray vLLM (recommended; bypasses the V1 server triton bug)")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_tokens", type=int, default=4096)
    p.add_argument("--output", required=True, help="HF Hub dataset path")
    p.add_argument("--multi_ref", action="store_true",
                   help="Run ref/noref/wrongref per row; emit llm_score_{cond} columns")
    # Local (DP+Ray) inference knobs
    p.add_argument("--tensor_parallel_size", type=int, default=1,
                   help="TP per vLLM worker (default 1 — Qwen3-4B fits on one H100/GH200)")
    p.add_argument("--data_parallel_size", type=int, default=None,
                   help="Number of DP workers (default: len(CUDA_VISIBLE_DEVICES)//TP)")
    p.add_argument("--gpu_memory_utilization", type=float, default=0.85)
    p.add_argument("--max_model_len", type=int, default=None)
    p.add_argument("--max_num_batched_tokens", type=int, default=32768)
    p.add_argument("--max_num_seqs", type=int, default=2048)
    # Qwen3 chat-template toggle: pass --no_thinking to disable the reasoning block.
    thinking = p.add_mutually_exclusive_group()
    thinking.add_argument("--enable_thinking", dest="enable_thinking",
                          action="store_true", default=None,
                          help="Force enable_thinking=True in the Qwen3 chat template")
    thinking.add_argument("--no_thinking", dest="enable_thinking",
                          action="store_false",
                          help="Force enable_thinking=False in the Qwen3 chat template")
    # Hosted (AsyncOpenAI) fallback knobs — only used when --use_local_model is omitted
    p.add_argument("--max-concurrency", type=int, default=256, dest="max_concurrency",
                   help="Concurrent requests for hosted API path only")
    p.add_argument("--request_timeout", type=float, default=1200.0,
                   help="Per-request timeout for hosted API path only")
    # Accepted-but-ignored legacy flags (so old scripts keep working)
    p.add_argument("--launch_model", action="store_true",
                   help="(deprecated) ignored — offline runner always self-launches")
    return p.parse_args()


if __name__ == "__main__":
    main(cli())
