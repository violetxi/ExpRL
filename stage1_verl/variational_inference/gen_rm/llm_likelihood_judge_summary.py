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


system_prompt = """You are an expert mathematician and a meticulous AI reasoning evaluator. Your task is to assess the substantive equivalence between a "Generated Reasoning" trace and a "Reference Solution."

You will first interpret and synthesize the {reasoning_trace} into its implied, coherent solution. Then, you will compare this synthesized solution to the {reference_solution}, focusing *only* on the mathematical and logical substance, not on the writing style or formatting.
"""

user_prompt_template = """
### Math Problem
{problem}


### Generated Reasoning
{reasoning_trace}


### Reference Solution
{reference_solution}


### Instructions
Your goal is to score the substantive equivalence between the solution implied by the "Generated Reasoning" and the "Reference Solution."
1.  Synthesize Solution: First, read the *Generated Reasoning* and write a clear, step-by-step summary of the solution it implies. This summary should be written as if it's the final answer derived from the thoughts.
2.  Analyze Comparison: Next, provide a step-by-step analysis comparing your *Implied Solution from Reasoning* (from Step 1) to the provided *Reference Solution.* You must ignore all differences in style, formatting, verbosity, or phrasing. Focus *only* on:
    * Logical Equivalence: Do both solutions use the same core method, logic, and justification?
    * Numerical Equivalence: Are the key intermediate values, formulas, and the final result identical?
    * Substantive Gaps: Does the *Implied Solution* *omit* any critical logical steps or calculations that are present in the *Reference Solution*?
    * Contradictions: Does the *Implied Solution* contain any logic or results that *contradict* the *Reference Solution*?
3.  Score: After your analysis, provide a single numerical score on the 5-point "Equivalence Scale" defined below.

### Output Format

Implied Solution from Reasoning: [Your summary of the solution implied by the *Generated Reasoning* goes here.]

Comparison Analysis: [Your detailed analysis comparing the "Implied Solution" to the "Reference Solution" goes here.]

Score: [Provide only a single numerical score: 1, 2, 3, 4, or 5.]
"""


async def single_judgement(
    client: AsyncOpenAI,
    problem: str,
    reasoning_trace: str,
    reference_solution: str,
    model: str,
    sem: asyncio.Semaphore,
    temperature: float = 0.0,
    use_local_model: bool = False,
) -> Tuple[str, int]:
    """One LLM call. Returns (response_str, score)."""
    user_msg = user_prompt_template.format(
        problem=problem,
        reasoning_trace=reasoning_trace,
        reference_solution=reference_solution
    )
    
    try:
        async with sem:   # concurrency throttle
            if use_local_model:
                response = await client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    max_tokens=4096,
                    n=1,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user",   "content": user_msg},
                    ],
                    extra_body={"chat_template_kwargs": {"enable_thinking": False}},                
                )    
            else:
                response = await client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    max_tokens=4096,
                    n=1,
                    messages=[
                        {"role": "system", "content": system_prompt},
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
        score = int(score_str)
        if score not in [1, 2, 3, 4, 5]:
            score = None
    except Exception as e:
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
    use_local_model: bool = False,
) -> Tuple[str, int]:
    """
    Evaluate a single reasoning trace.
    Returns (response_str, score) where score is 1-5.
    """
    response, score = await single_judgement(
        client, problem, reasoning_trace, reference_solution, model, sem, temperature, use_local_model
    )
    
    return response, score


def load_jobs(data_path: Path) -> Tuple[List[Tuple[str, str, str]], Dataset]:
    """Load dataset and extract problem, reasoning, and reference solution."""
    jobs = []
    dataset = load_dataset(data_path, split="train")

    output_data = {
        "problem": [],
        "reasoning_trace": [],
        "reference_solution": [],
        "finish_thinking": [],
        "reward": []
    }
    
    for prompt_info, extra_info, response, reward in zip(
        dataset["prompt"], dataset["extra_info"], dataset["responses"], dataset["score"]):
        prompt = prompt_info[0]["content"]
        reference_solution = str(extra_info["solution"])
        
        if  "</think>" in response:
            finish_thinking = True
            reasoning_trace = response.split("</think>")[0] + "</think>"
        else:
            finish_thinking = False
            reasoning_trace = response

        jobs.append((prompt, reasoning_trace, reference_solution))
        output_data["problem"].append(prompt)
        output_data["reasoning_trace"].append(reasoning_trace)
        output_data["reference_solution"].append(reference_solution)
        output_data["finish_thinking"].append(finish_thinking)
        output_data["reward"].append(reward)
    return jobs, Dataset.from_dict(output_data)



async def main_async(args):
    OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
    if not OPENROUTER_API_KEY:
        raise EnvironmentError("OPENROUTER_API_KEY env var is not set.")
        
    if args.use_local_model:
        assert args.model in ["qwen/qwen3_4b", "qwen/qwen3_8b", "qwen/qwen3_14b"], \
            f"{args.model} is not supported for local model"
        client = AsyncOpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
    else:        
        client = AsyncOpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")
    sem = asyncio.Semaphore(args.max_concurrency)

    jobs, output_ds = load_jobs(args.data)
    
    # Evaluate each reasoning trace
    coros = [
        evaluate_reasoning(
            client, problem, reasoning, answer, args.model, sem, args.temperature, args.use_local_model
        ) for problem, reasoning, answer in jobs
    ]
    results = await asyncio.gather(*coros)    
    scores = [s for _, s in results]
    responses = [r for r, _ in results]

    # Add results to output dataset
    output_ds = output_ds.add_column("llm_score", scores)
    output_ds = output_ds.add_column("llm_response", responses)

    output_ds.push_to_hub(args.output, private=False)

def cli():
    parser = argparse.ArgumentParser(description="Async LLM likelihood evaluator for reasoning traces.")
    parser.add_argument("--data", required=True, help="path to dataset")
    parser.add_argument("--model", default="x-ai/grok-3-mini", help="Model name")
    parser.add_argument("--use_local_model", action="store_true", help="Use local model")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-concurrency", type=int, default=256, help="Simultaneous LLM calls (default 256).")
    parser.add_argument("--enable_thinking", action="store_true", help="Enable thinking")
    parser.add_argument("--output", required=True, help="path to output (HuggingFace Hub)")
    return parser.parse_args()

if __name__ == "__main__":
    asyncio.run(main_async(cli()))
