#!/usr/bin/env python3
"""
llm_likelihood_judge_no_ref.py – Async batch evaluator for reasoning trace quality WITHOUT reference solution.

Each input must contain:
    { "problem": "...", "reasoning_trace": "..." }

This judge evaluates the reasoning WITHOUT seeing the ground truth solution,
assessing the mathematical correctness, logical soundness, and completeness of the reasoning.

Example:
    python llm_likelihood_judge_no_ref.py --data dataset_path --model gpt-4o-mini
"""

import os
import asyncio
import argparse
import numpy as np
from datasets import Dataset, load_dataset
from pathlib import Path
from typing import Tuple, List

from openai import AsyncOpenAI   # pip install --upgrade openai>=1.2.4


SYSTEM_PROMPT = """You are an expert mathematician and a meticulous AI reasoning evaluator. Your task is to assess the quality and correctness of a "Generated Reasoning" trace for a math problem.

**Important:** You will NOT be given a reference solution. You must independently evaluate whether the reasoning is mathematically sound, logically coherent, and arrives at a correct answer.

You are evaluating the **absolute quality** of the reasoning, not comparing it to anything."""


USER_PROMPT = """### Math Problem
{problem}

### Generated Reasoning
{generated_reasoning}

### Instructions
Provide a thorough step-by-step analysis of the reasoning quality. In your analysis, consider the following:

* **Mathematical Correctness:** Are all mathematical operations, formulas, and logical steps correct? Are there any errors in calculations or logical deductions?
* **Logical Coherence:** Does the reasoning flow logically from one step to the next? Are there any logical jumps, contradictions, or non-sequiturs?
* **Completeness:** Does the reasoning provide a complete solution to the problem? Does it arrive at a final answer, or does it stop prematurely?
* **Clarity:** Is the reasoning clear and well-explained? Can you follow the logic easily?
* **Answer Validity:** If a final answer is provided, does it make sense given the problem? Is it in the correct format and units (if applicable)?

After your analysis, provide a single numerical score on the 5-point Likert scale defined below.

**Likert Scale:**
* **1 (Very Poor):** The reasoning contains major mathematical errors, is logically incoherent, or completely fails to address the problem.
* **2 (Poor):** The reasoning has significant flaws, gaps, or errors that would likely lead to an incorrect answer or no answer at all.
* **3 (Acceptable):** The reasoning is generally on the right track with minor errors or incompleteness. It shows understanding but may not reach a fully correct or complete solution.
* **4 (Good):** The reasoning is mostly correct and complete with only trivial issues. It demonstrates strong mathematical reasoning and likely arrives at the correct answer.
* **5 (Excellent):** The reasoning is mathematically sound, logically coherent, complete, and clearly leads to a correct solution. The work is exemplary.

---

Please follow this output format:
Reasoning: [Your detailed analysis goes here.]

Score: [Provide the single numerical score: 1, 2, 3, 4, or 5.]
"""


async def single_judgement(
    client: AsyncOpenAI,
    problem: str,
    reasoning_trace: str,
    model: str,
    sem: asyncio.Semaphore,
    temperature: float = 0.0,
) -> Tuple[str, int]:
    """One LLM call. Returns (response_str, score)."""
    user_msg = USER_PROMPT.format(
        problem=problem,
        generated_reasoning=reasoning_trace
    )

    try:
        async with sem:   # concurrency throttle
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
        # Extract the numerical score (1-5)
        score = int(score_str[0])  # Get first character and convert to int
        if score not in [1, 2, 3, 4, 5]:
            score = None
    except Exception as e:
        score = None

    return response_str, score

async def evaluate_reasoning(
    client: AsyncOpenAI,
    problem: str,
    reasoning_trace: str,
    model: str,
    sem: asyncio.Semaphore,
    temperature: float = 0.,
) -> Tuple[str, int]:
    """
    Evaluate a single reasoning trace.
    Returns (response_str, score) where score is 1-5.
    """
    response, score = await single_judgement(
        client, problem, reasoning_trace, model, sem, temperature
    )

    return response, score


def load_jobs(data_path: Path) -> Tuple[List[Tuple[str, str]], Dataset]:
    """Load dataset and extract problem and reasoning (no reference solution)."""
    jobs = []
    dataset = load_dataset(data_path, split="train")

    output_data = {
        "problem": [],
        "reasoning_trace": [],
        "finish_thinking": [],
        "reward": []
    }

    all_responses = dataset["responses"] if "responses" in dataset else dataset["response"]
    for prompt_info, extra_info, response, reward in zip(
        dataset["prompt"], dataset["extra_info"], all_responses, dataset["score"]):
        prompt = prompt_info[0]["content"]

        if  "</think>" in response:
            finish_thinking = True
            reasoning_trace = response.split("</think>")[0] + "</think>"
        else:
            finish_thinking = False
            reasoning_trace = response

        jobs.append((prompt, reasoning_trace))
        output_data["problem"].append(prompt)
        output_data["reasoning_trace"].append(reasoning_trace)
        output_data["finish_thinking"].append(finish_thinking)
        output_data["reward"].append(reward)
    return jobs, Dataset.from_dict(output_data)



async def main_async(args):
    if args.use_local_model:
        assert args.model in ["qwen/qwen3_4b", "qwen/qwen3_8b", "qwen/qwen3_14b", "qwen/qwen3_4b_instruct", "qwen/qwen3_30b_instruct"], \
            f"{args.model} is not supported for local model"
        client = AsyncOpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
    else:
        OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
        if not OPENROUTER_API_KEY:
            raise EnvironmentError("OPENROUTER_API_KEY env var is not set.")
        client = AsyncOpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")
    sem = asyncio.Semaphore(args.max_concurrency)

    jobs, output_ds = load_jobs(args.data)

    # Evaluate each reasoning trace
    coros = [
        evaluate_reasoning(
            client, problem, reasoning, args.model, sem, args.temperature
        ) for problem, reasoning in jobs
    ]
    results = await asyncio.gather(*coros)
    scores = [s for _, s in results]
    responses = [r for r, _ in results]

    # Add results to output dataset
    output_ds = output_ds.add_column("llm_score", scores)
    output_ds = output_ds.add_column("llm_response", responses)

    output_ds.push_to_hub(args.output, private=False)

def cli():
    parser = argparse.ArgumentParser(description="Async LLM evaluator for reasoning traces WITHOUT reference solution.")
    parser.add_argument("--data", required=True, help="path to dataset")
    parser.add_argument("--model", default="x-ai/grok-3-mini", help="Model name")
    parser.add_argument("--use_local_model", action="store_true", help="Use local model")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-concurrency", type=int, default=256, help="Simultaneous LLM calls (default 256).")
    parser.add_argument("--output", required=True, help="path to output (HuggingFace Hub)")
    return parser.parse_args()

if __name__ == "__main__":
    asyncio.run(main_async(cli()))
