#!/usr/bin/env python3
"""
Batch LLM judge for SciKnowEval generated responses (OE & MCQ).

Each input row follows the vllm_generation/SciKnowEval schema:
    prompt / extra_info / response / score
question is read from extra_info["question"] when available; reference answer
is read from extra_info["solution"] (or reward_model.ground_truth).

Local inference uses the offline DP+Ray vLLM runner (_judge_runner.py).
"""

import argparse
import asyncio
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from datasets import Dataset, load_dataset

from verl.utils.prompt_templates.sciknow_likelihood_judge import (
    SYSTEM_PROMPT as SYSTEM_PROMPT_REF,
    USER_PROMPT as USER_PROMPT_REF,
)
from verl.utils.prompt_templates.sciknow_likelihood_judge_no_ref import (
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


def extract_question(item: Dict) -> str:
    extra_info = item.get("extra_info") or {}
    if extra_info.get("question"):
        return str(extra_info["question"])
    prompt = item.get("prompt") or []
    if isinstance(prompt, list):
        for msg in reversed(prompt):
            if isinstance(msg, dict) and msg.get("role") == "user":
                return str(msg.get("content") or "")
        if prompt and isinstance(prompt[-1], dict):
            return str(prompt[-1].get("content") or "")
    return str(prompt)


def extract_reference_solution(item: Dict) -> str:
    extra_info = item.get("extra_info") or {}
    if extra_info.get("solution"):
        return str(extra_info["solution"])
    reward_model = item.get("reward_model") or {}
    if isinstance(reward_model, dict) and reward_model.get("ground_truth"):
        return str(reward_model["ground_truth"])
    return ""


def build_messages(
    condition: str,
    question: str,
    generated_response: str,
    reference_solution: str,
    wrong_reference_solution: str,
    metadata: Dict,
) -> List[Dict[str, str]]:
    domain = metadata.get("domain") or "Science"
    task = metadata.get("task") or ""
    subtask = metadata.get("subtask") or ""
    if condition == "ref":
        sys_p = SYSTEM_PROMPT_REF
        user_p = USER_PROMPT_REF.format(
            domain=domain, task=task, subtask=subtask,
            question=question,
            generated_response=generated_response,
            reference_solution=reference_solution,
        )
    elif condition == "noref":
        sys_p = SYSTEM_PROMPT_NOREF
        user_p = USER_PROMPT_NOREF.format(
            domain=domain, task=task, subtask=subtask,
            question=question,
            generated_response=generated_response,
        )
    elif condition == "wrongref":
        sys_p = SYSTEM_PROMPT_REF
        user_p = USER_PROMPT_REF.format(
            domain=domain, task=task, subtask=subtask,
            question=question,
            generated_response=generated_response,
            reference_solution=wrong_reference_solution,
        )
    else:
        raise ValueError(f"unknown condition: {condition!r}")
    return [{"role": "system", "content": sys_p},
            {"role": "user",   "content": user_p}]


def build_wrong_ref_map(unique_keys: List[str], seed: int = WRONG_REF_SEED) -> Dict[str, str]:
    keys = sorted(set(unique_keys))
    if len(keys) < 2:
        raise ValueError("wrong-reference requires at least 2 unique questions")
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


def load_jobs(data_path: str, split: str, max_rows: int = 0, multi_ref: bool = False):
    dataset = load_dataset(data_path, split=split)
    if max_rows:
        dataset = dataset.select(range(min(max_rows, len(dataset))))

    question_to_ref: Dict[str, str] = {}
    for item in dataset:
        q = extract_question(item)
        r = extract_reference_solution(item)
        if q and r and q not in question_to_ref:
            question_to_ref[q] = r

    wrong_q_map: Dict[str, str] = {}
    if multi_ref:
        wrong_q_map = build_wrong_ref_map(list(question_to_ref.keys()))
        print(f"Built wrong-ref map for {len(wrong_q_map)} unique questions "
              f"(seed={hex(WRONG_REF_SEED)})")

    jobs: List[Tuple[str, str, str, str, Dict]] = []
    output_data = {
        "data_source": [],
        "prompt": [],
        "ability": [],
        "extra_info": [],
        "question": [],
        "generated_response": [],
        "reasoning_trace": [],
        "reference_solution": [],
        "finish_thinking": [],
        "has_thinking": [],
        "original_score": [],
    }
    if multi_ref:
        output_data["wrong_reference_question"] = []
        output_data["wrong_reference_solution"] = []

    skipped = 0
    for item in dataset:
        extra_info = item.get("extra_info") or {}
        question = extract_question(item)
        reference_solution = extract_reference_solution(item)
        generated_response = item.get("response")
        if generated_response is None:
            generated_response = item.get("responses")
        generated_response = "" if generated_response is None else str(generated_response)

        if not question or not reference_solution:
            skipped += 1
            continue

        finish_thinking = "</think>" in generated_response
        has_thinking = bool(item.get("has_thinking", finish_thinking))

        if multi_ref:
            wrong_question = wrong_q_map[question]
            wrong_reference_solution = question_to_ref[wrong_question]
            jobs.append((question, generated_response, reference_solution,
                         wrong_reference_solution, extra_info))
            output_data["wrong_reference_question"].append(wrong_question)
            output_data["wrong_reference_solution"].append(wrong_reference_solution)
        else:
            jobs.append((question, generated_response, reference_solution, "", extra_info))

        output_data["data_source"].append(item.get("data_source", ""))
        output_data["prompt"].append(item.get("prompt", []))
        output_data["ability"].append(item.get("ability", "Science"))
        output_data["extra_info"].append(extra_info)
        output_data["question"].append(question)
        output_data["generated_response"].append(generated_response)
        output_data["reasoning_trace"].append(generated_response)
        output_data["reference_solution"].append(reference_solution)
        output_data["finish_thinking"].append(finish_thinking)
        output_data["has_thinking"].append(has_thinking)
        output_data["original_score"].append(item.get("score"))

    if skipped:
        print(f"Skipped {skipped} rows without question or reference solution")
    print(f"Loaded {len(jobs)} SciKnowEval rows from {data_path} [{split}]"
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
            enable_thinking=args.enable_thinking,
        )
    return asyncio.run(_hosted_batch(
        messages_list, args.model, args.temperature, args.max_tokens,
        args.max_concurrency, args.request_timeout,
    ))


def main(args):
    jobs, output_ds = load_jobs(args.data, args.split, args.max_rows, multi_ref=args.multi_ref)

    if args.multi_ref:
        all_messages: List[List[Dict[str, str]]] = []
        for question, generated_response, reference, wrong_reference, metadata in jobs:
            for cond in CONDITIONS:
                all_messages.append(build_messages(
                    cond, question, generated_response, reference, wrong_reference, metadata,
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
            build_messages("ref", q, gr, ref, "", md)
            for q, gr, ref, _, md in jobs
        ]
        responses = run_inference(args, all_messages)
        scores = [parse_score_likert(r) for r in responses]
        output_ds = output_ds.add_column("llm_score", scores)
        output_ds = output_ds.add_column("llm_response", responses)

    print(f"Pushing to HuggingFace Hub: {args.output}")
    output_ds.push_to_hub(args.output, private=False)
    print("Done.")


def cli():
    p = argparse.ArgumentParser(description="Batch LLM judge for SciKnowEval generated responses.")
    p.add_argument("--data", required=True)
    p.add_argument("--split", default="train")
    p.add_argument("--model", default="qwen/qwen3_4b_instruct")
    p.add_argument("--use_local_model", action="store_true")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--max_tokens", type=int, default=4096)
    p.add_argument("--max_rows", type=int, default=0)
    p.add_argument("--output", required=True)
    p.add_argument("--multi_ref", action="store_true")
    # Local DP+Ray knobs
    p.add_argument("--tensor_parallel_size", type=int, default=1)
    p.add_argument("--data_parallel_size", type=int, default=None)
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
    # Hosted API fallback knobs
    p.add_argument("--max-concurrency", type=int, default=256, dest="max_concurrency")
    p.add_argument("--request_timeout", type=float, default=1200.0)
    # Legacy flags
    p.add_argument("--launch_model", action="store_true",
                   help="(deprecated) ignored")
    return p.parse_args()


if __name__ == "__main__":
    main(cli())
