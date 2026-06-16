#!/usr/bin/env python3
"""
llm_likelihood_judge.py – Async batch evaluator for reasoning trace likelihood.

Each input must contain:
    { "problem": "...", "reasoning_trace": "...", "reference_solution": "..." }

Example:
    python llm_likelihood_judge.py --data dataset_path --model gpt-4o-mini
"""

import os
import asyncio
import argparse
import numpy as np
from datasets import Dataset, load_dataset
from pathlib import Path
from typing import Tuple, List

from openai import AsyncOpenAI   # pip install --upgrade openai>=1.2.4



USER_PROMPT = """{problem}\n\nPlease solve the problem step by step and put your final answer within \\boxed{{}}."""


async def single_generation(
    client: AsyncOpenAI,
    problem: str,
    model: str,
    sem: asyncio.Semaphore,
    temperature: float = 0.0,
    max_tokens: int = 16384,
) -> str:
    """One LLM call. Returns (response_str, score)."""
    user_msg = USER_PROMPT.format(problem=problem)
    try:
        async with sem:   # concurrency throttle
            response = await client.chat.completions.create(
                model=model,
                temperature=temperature,
                n=1,
                max_tokens=max_tokens,
                messages=[
                    {"role": "user",   "content": user_msg},
                ],
                # disable thinking
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
            )        
        response_str = response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error: {e}")
        response_str = ""        
    
    return response_str

async def generate_llm_solutions(
    client: AsyncOpenAI,
    problem: str,
    model: str,
    sem: asyncio.Semaphore,
    temperature: float = 0.,
    max_tokens: int = 16384,
) -> str:
    """
    Evaluate a single reasoning trace.
    Returns response_str.
    """
    response = await single_generation(client, problem, model, sem, temperature, max_tokens)
    
    return response


def load_jobs(data_path: Path) -> Tuple[List[str], Dataset]:
    """Load dataset and extract problems."""
    dataset = load_dataset(data_path, split="train")
    # if the dataset does not have the 'problem_index' column, add the index of row as the problem_index
    if 'problem_index' not in dataset.column_names:
        dataset = dataset.map(lambda x, idx: {"problem_index": idx}, with_indices=True)
    jobs = dataset["problem"]  # extract problem column as list
    return jobs, dataset



async def main_async(args):
    if args.use_local_model:
        assert args.model in ["qwen/qwen3_4b", "qwen/qwen3_8b", "qwen/qwen3_14b", "qwen/qwen3_4b_instruct_2507"], \
            f"{args.model} is not supported for local model"
        client = AsyncOpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
    else:
        OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
        if not OPENROUTER_API_KEY:
            raise EnvironmentError("OPENROUTER_API_KEY env var is not set.")
        client = AsyncOpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")
    sem = asyncio.Semaphore(args.max_concurrency)

    jobs, output_ds = load_jobs(args.data)
    all_results = []
    total_jobs = len(jobs)
    for i in range(0, total_jobs, args.chunk_size):
        chunk = jobs[i:i + args.chunk_size]
        chunk_end = min(i + args.chunk_size, total_jobs)
        print(f"Processing chunk {i // args.chunk_size + 1}: jobs {i + 1}-{chunk_end} of {total_jobs}")

        coros = [
            generate_llm_solutions(
                client, problem, args.model, sem, args.temperature, args.max_tokens
            ) for problem in chunk
        ]
        chunk_results = await asyncio.gather(*coros)
        all_results.extend(chunk_results)
    
    output_ds = output_ds.add_column("llm_solutions", all_results)
    output_ds.push_to_hub(args.output, private=False)

def cli():
    parser = argparse.ArgumentParser(description="Async LLM likelihood evaluator for reasoning traces.")
    parser.add_argument("--data", required=True, help="path to dataset")
    parser.add_argument("--model", default="x-ai/grok-3-mini", help="Model name")
    parser.add_argument("--use_local_model", action="store_true", help="Use local model")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=32768, help="Max tokens to generate (default 32768).")
    parser.add_argument("--max-concurrency", type=int, default=256, help="Simultaneous LLM calls (default 256).")
    parser.add_argument("--chunk-size", type=int, default=4096, help="Process jobs in chunks of this size (default 10000).")
    parser.add_argument("--output", required=True, help="path to output (HuggingFace Hub)")
    return parser.parse_args()

if __name__ == "__main__":
    asyncio.run(main_async(cli()))
