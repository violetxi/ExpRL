#!/usr/bin/env python3
"""
llm_likelihood_judge_science.py – Async batch evaluator for science reasoning traces.

Science (open-ended) analog of llm_likelihood_judge.py. The math judge expects a
single `prompt[0]` containing the problem; SciKnowEval OE rows are system+user
chats, so the problem is taken from `extra_info["question"]` (set by
scripts/prepare_sciknoweval_*.py / build_oracle_dataset.py).

Each row must contain:
    { "prompt": [{role, content}, ...],
      "extra_info": {"question": ..., "solution": ...},
      "response": str,
      "score": float }

Prompts come from verl/utils/prompt_templates/science_likelihood_judge.py.

Example:
    python variational_inference/gen_rm/llm_likelihood_judge_science.py \\
        --data violetxi/sciknoweval_oe_v1_smoketest100_qwen8b \\
        --model qwen/qwen3_4b_instruct \\
        --use_local_model \\
        --output violetxi/Qwen4b-Instruct-Judge-sciknow-oe-v1-smoketest100-outcome \\
        --max-concurrency 256 \\
        --temperature 0.0
"""

import os
import asyncio
import argparse
from datasets import Dataset, load_dataset
from pathlib import Path
from typing import Tuple, List

from openai import AsyncOpenAI

from verl.utils.prompt_templates.science_likelihood_judge import SYSTEM_PROMPT, USER_PROMPT


async def single_judgement(
    client: AsyncOpenAI,
    problem: str,
    reasoning_trace: str,
    reference_solution: str,
    model: str,
    sem: asyncio.Semaphore,
    temperature: float = 0.0,
) -> Tuple[str, int]:
    """One LLM call. Returns (response_str, score)."""
    user_msg = USER_PROMPT.format(
        problem=problem,
        reasoning_trace=reasoning_trace,
        reference_solution=reference_solution,
    )

    try:
        async with sem:
            response = await client.chat.completions.create(
                model=model,
                temperature=temperature,
                n=1,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
            )
        response_str = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error: {e}")
        response_str = ""

    try:
        if "Score:" in response_str:
            score_str = response_str.split("Score:")[1].lstrip().split("\n")[0]
        else:
            score_str = response_str.split("Score")[1].lstrip().split("\n")[0]
        score_str = score_str.replace("**", "").replace("*", "")[0]
        score = int(score_str[0])
        if score not in [1, 2, 3, 4, 5]:
            score = None
    except Exception:
        score = None

    return response_str, score


async def evaluate_reasoning(
    client: AsyncOpenAI,
    problem: str,
    reasoning_trace: str,
    reference_solution: str,
    model: str,
    sem: asyncio.Semaphore,
    temperature: float = 0.,
) -> Tuple[str, int]:
    response, score = await single_judgement(
        client, problem, reasoning_trace, reference_solution, model, sem, temperature
    )
    return response, score


def _extract_problem(prompt_field, extra_info) -> str:
    """Prefer extra_info['question'] (preserved verbatim by the SciKnowEval prep
    scripts). Fall back to the last user message in the chat-formatted prompt."""
    q = (extra_info or {}).get("question")
    if q:
        return str(q)
    for msg in reversed(prompt_field or []):
        if msg.get("role") == "user":
            return msg.get("content", "")
    return prompt_field[-1]["content"] if prompt_field else ""


def load_jobs(data_path: Path) -> Tuple[List[Tuple[str, str, str]], Dataset]:
    """Load generations dataset and extract (problem, reasoning, reference)."""
    jobs = []
    dataset = load_dataset(data_path, split="train")

    output_data = {
        "problem": [],
        "reasoning_trace": [],
        "reference_solution": [],
        "finish_thinking": [],
        "reward": [],
    }

    all_responses = dataset["responses"] if "responses" in dataset.column_names else dataset["response"]
    skipped = 0
    for prompt_info, extra_info, response, reward in zip(
        dataset["prompt"], dataset["extra_info"], all_responses, dataset["score"]
    ):
        problem = _extract_problem(prompt_info, extra_info)
        reference_solution = str((extra_info or {}).get("solution") or "")
        if not reference_solution:
            skipped += 1
            continue

        if "</think>" in response:
            finish_thinking = True
            reasoning_trace = response.split("</think>")[0] + "</think>"
        else:
            finish_thinking = False
            reasoning_trace = response

        jobs.append((problem, reasoning_trace, reference_solution))
        output_data["problem"].append(problem)
        output_data["reasoning_trace"].append(reasoning_trace)
        output_data["reference_solution"].append(reference_solution)
        output_data["finish_thinking"].append(finish_thinking)
        output_data["reward"].append(reward)

    if skipped:
        print(f"Skipped {skipped} rows lacking a reference solution in extra_info")
    print(f"Will judge {len(jobs)} rows")
    return jobs, Dataset.from_dict(output_data)


async def main_async(args):
    if args.use_local_model:
        assert args.model in [
            "qwen/qwen3_4b", "qwen/qwen3_8b", "qwen/qwen3_14b",
            "qwen/qwen3_4b_instruct", "qwen/qwen3_30b_instruct",
        ], f"{args.model} is not supported for local model"
        client = AsyncOpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
    else:
        OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
        if not OPENROUTER_API_KEY:
            raise EnvironmentError("OPENROUTER_API_KEY env var is not set.")
        client = AsyncOpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")
    sem = asyncio.Semaphore(args.max_concurrency)

    jobs, output_ds = load_jobs(args.data)

    coros = [
        evaluate_reasoning(
            client, problem, reasoning, reference, args.model, sem, args.temperature,
        )
        for problem, reasoning, reference in jobs
    ]
    results = await asyncio.gather(*coros)
    scores = [s for _, s in results]
    responses = [r for r, _ in results]

    output_ds = output_ds.add_column("llm_score", scores)
    output_ds = output_ds.add_column("llm_response", responses)

    output_ds.push_to_hub(args.output, private=False)


def cli():
    parser = argparse.ArgumentParser(description="Async LLM likelihood evaluator for science reasoning traces.")
    parser.add_argument("--data", required=True, help="path to HF dataset of generations")
    parser.add_argument("--model", default="qwen/qwen3_4b_instruct", help="Judge model name")
    parser.add_argument("--use_local_model", action="store_true", help="Use local vLLM server at :8000/v1")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-concurrency", type=int, default=256, help="Simultaneous LLM calls (default 256).")
    parser.add_argument("--output", required=True, help="HF Hub dataset path to push results to")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main_async(cli()))
