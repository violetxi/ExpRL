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
import subprocess
import time
import requests

from openai import AsyncOpenAI   # pip install --upgrade openai>=1.2.4


SYSTEM_PROMPT = """You are an expert mathematician and a meticulous AI reasoning evaluator. Your task is to assess the quality of a "Generated Reasoning" trace.
This trace may be **partial or incomplete**.

Specifically, you must judge how likely it is that a model, *after* thinking through the "Generated Reasoning" (even if it's just a starting point), would *continue* on a correct path to eventually produce the *exact* "Reference Solution" provided.

You are evaluating the **logical and causal link** between this (potentially partial) reasoning and the final answer. Is this a correct and promising *prefix* of a full, correct reasoning trace?

### Instructions

First, provide a step-by-step analysis of the connection between the "Generated Reasoning" and the "Reference Solution." In your analysis, consider the following:

* **Correctness & Alignment:** Is the reasoning *so far* mathematically sound? Does it align with the known facts of the problem and the logical path required to reach the "Reference Solution"?
* **Progression:** Does this partial trace represent a *correct and logical step* (or steps) towards the solution? Is it on the right path, or has it already made a mistake, taken an unproductive turn, or stopped at a point that isn't a clear step forward?
* **Contradiction:** Is there anything *in this partial trace* that already contradicts the "Reference Solution" or makes it impossible to arrive at it logically? Does it set up a line of thinking that would lead to a *different* answer?

After your analysis, provide a single numerical score on the 5-point Likert scale defined below.

**Likert Scale:**
* **1 (Very Unlikely):** The partial reasoning is incorrect, unrelated, or already points toward a completely different solution.
* **2 (Unlikely):** The partial reasoning has significant flaws, is a "dead end," or is too vague, making it highly improbable that a correct continuation would follow from it.
* **3 (Neutral/Possible):** The partial reasoning is generally on the right track (or at least not wrong) but is very incomplete, trivial, or doesn't represent a significant step forward. It's plausible but not guaranteed to lead to the solution.
* **4 (Likely):** The partial reasoning is correct, logical, and represents a clear and significant step on the path to the "Reference Solution."
* **5 (Very Likely):** The partial reasoning is a sound, strong, and unambiguous *prefix* of a correct path to the "Reference Solution." The next logical step is clearly in the direction of the final answer.

---

Please follow this output format:
Reasoning: [Your detailed analysis goes here.]

Score: [Provide the single numerical score: 1, 2, 3, 4, or 5.]
"""


USER_PROMPT = """### Math Problem
{problem}

### Generated Reasoning
{generated_reasoning}

### Reference Solution
{reference_solution}
"""


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
        generated_reasoning=reasoning_trace,
        reference_solution=reference_solution
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
                # disable thinking
                extra_body={"chat_template_kwargs": {"enable_thinking": False}},
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
    reference_solution: str,
    model: str,
    sem: asyncio.Semaphore,
    temperature: float = 0.,
) -> Tuple[str, int]:
    """
    Evaluate a single reasoning trace.
    Returns (response_str, score) where score is 1-5.
    """
    response, score = await single_judgement(
        client, problem, reasoning_trace, reference_solution, model, sem, temperature
    )
    
    return response, score


def load_jobs(data_path: Path) -> Tuple[List[Tuple[str, str, str]], Dataset]:
    """Load dataset and extract problem, reasoning, and Reference Solution."""
    jobs = []
    dataset = load_dataset(data_path, split="train")

    output_data = {
        "problem": [],
        "problem_index": [],
        "original_response": [],
        "reasoning_trace": [],
        "step_index": [],
        "ability": [],
        "data_source": [],
        "extra_info": [],
        "reward_model": [],
        "reference_solution": [],
        "original_score": []
    }
    for item in dataset:
        problem = item["problem"]
        problem_index = item["problem_index"]
        original_response = item["original_response"]
        reasoning_trace = item["step_only"]
        step_index = item["step_index"]
        ability = item["ability"]
        data_source = item["data_source"]
        extra_info = item["extra_info"]
        reward_model = item["reward_model"]
        reference_solution = str(item['extra_info']["solution"])
        original_score = item["original_score"]

        jobs.append((problem, reasoning_trace, reference_solution))
        output_data["problem"].append(problem)
        output_data["problem_index"].append(problem_index)
        output_data["original_response"].append(original_response)
        output_data["reasoning_trace"].append(reasoning_trace)
        output_data["step_index"].append(step_index)
        output_data["reference_solution"].append(reference_solution)
        output_data["original_score"].append(original_score)
        output_data["ability"].append(ability)
        output_data["data_source"].append(data_source)
        output_data["extra_info"].append(extra_info)
        output_data["reward_model"].append(reward_model)
    return jobs, Dataset.from_dict(output_data)


def get_vllm_launch_command(model_name: str, tensor_parallel_size: int = None) -> list:
    """Get the vLLM launch command based on the model name."""
    # Auto-detect tensor parallel size from CUDA_VISIBLE_DEVICES if not specified
    if tensor_parallel_size is None:
        cuda_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        if cuda_devices:
            tensor_parallel_size = len(cuda_devices.split(','))
        else:
            tensor_parallel_size = 8  # default

    tp_size = str(tensor_parallel_size)

    model_configs = {
        "qwen/qwen3_4b": [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", "Qwen/Qwen3-4B",
            "--served-model-name", "qwen/qwen3_4b",
            "--tensor-parallel-size", tp_size,
            "--max-num-batched-tokens", "32768",
            "--max-num-seqs", "2048"
        "--enforce-eager",
        ],
        "qwen/qwen3_8b": [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", "Qwen/Qwen3-8B",
            "--served-model-name", "qwen/qwen3_8b",
            "--tensor-parallel-size", tp_size,
            "--gpu-memory-utilization", "0.85",
            "--max-num-batched-tokens", "32768",
            "--max-num-seqs", "2048",
            "--max-model-len", "131072",
            "--rope-scaling", '{"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 40960}'
        "--enforce-eager",
        ],
        "qwen/qwen3_14b": [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", "Qwen/Qwen3-14B",
            "--served-model-name", "qwen/qwen3_14b",
            "--tensor-parallel-size", tp_size,
            "--max-num-batched-tokens", "32768",
            "--max-num-seqs", "2048"
        "--enforce-eager",
        ],
        "qwen/qwen3_4b_instruct": [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", "Qwen/Qwen3-4B-Instruct-2507",
            "--served-model-name", "qwen/qwen3_4b_instruct",
            "--tensor-parallel-size", tp_size,
            "--max-num-batched-tokens", "32768",
            "--max-num-seqs", "2048"
        "--enforce-eager",
        ],
        "qwen/qwen3_30b_instruct": [
            "python", "-m", "vllm.entrypoints.openai.api_server",
            "--model", "Qwen/Qwen3-30B-A3B-Instruct-2507",
            "--served-model-name", "qwen/qwen3_30b_instruct",
            "--tensor-parallel-size", tp_size,
            "--max-num-batched-tokens", "32768",
            "--max-num-seqs", "2048"
        ]

    }

    if model_name not in model_configs:
        raise ValueError(f"Model {model_name} is not supported. Supported models: {list(model_configs.keys())}")

    return model_configs[model_name]


def wait_for_server_ready(base_url: str = "http://localhost:8000", timeout: int = 600, check_interval: int = 5):
    """Wait for the vLLM server to be ready by polling the health endpoint."""
    health_url = f"{base_url}/health"
    models_url = f"{base_url}/v1/models"

    print(f"Waiting for vLLM server to be ready at {base_url}...")
    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            # Try to hit the health endpoint
            response = requests.get(health_url, timeout=2)
            if response.status_code == 200:
                try:
                    models_response = requests.get(models_url, timeout=2)
                    if models_response.status_code == 200:
                        print(f"Server is ready! (took {time.time() - start_time:.1f}s)")
                        return True
                except:
                    pass
        except requests.exceptions.RequestException:
            pass

        time.sleep(check_interval)
        elapsed = time.time() - start_time
        print(f"Still waiting... ({elapsed:.1f}s elapsed)")

    raise TimeoutError(f"Server did not become ready within {timeout} seconds")


def launch_vllm_server(model_name: str, tensor_parallel_size: int = None) -> subprocess.Popen:
    """Launch the vLLM server in a subprocess."""
    cmd = get_vllm_launch_command(model_name, tensor_parallel_size)
    print(f"Launching vLLM server with command: {' '.join(cmd)}")

    # Launch the server in a subprocess
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1
    )

    print(f"vLLM server process started with PID: {process.pid}")
    return process



async def main_async(args):
    vllm_process = None

    # Validate arguments
    if args.launch_model and not args.use_local_model:
        raise ValueError("--launch_model requires --use_local_model to be set")

    try:
        if args.use_local_model:
            assert args.model in ["qwen/qwen3_4b", "qwen/qwen3_8b", "qwen/qwen3_14b", "qwen/qwen3_4b_instruct", "qwen/qwen3_30b_instruct"], \
                f"{args.model} is not supported for local model"

            # Launch the vLLM server if requested
            if args.launch_model:
                vllm_process = launch_vllm_server(args.model, args.tensor_parallel_size)
                # Wait for the server to be ready
                wait_for_server_ready()

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
                client, problem, reasoning, answer, args.model, sem, args.temperature
            ) for problem, reasoning, answer in jobs
        ]
        results = await asyncio.gather(*coros)
        scores = [s for _, s in results]
        responses = [r for r, _ in results]

        # Add results to output dataset
        output_ds = output_ds.add_column("llm_score", scores)
        output_ds = output_ds.add_column("llm_response", responses)

        output_ds.push_to_hub(args.output, private=False)

    finally:
        # Clean up the vLLM server process if we launched it
        if vllm_process is not None:
            print(f"Shutting down vLLM server (PID: {vllm_process.pid})...")
            vllm_process.terminate()
            try:
                vllm_process.wait(timeout=10)
                print("vLLM server shut down gracefully")
            except subprocess.TimeoutExpired:
                print("vLLM server did not terminate gracefully, killing it...")
                vllm_process.kill()
                vllm_process.wait()
                print("vLLM server killed")

def cli():
    parser = argparse.ArgumentParser(description="Async LLM likelihood evaluator for reasoning traces.")
    parser.add_argument("--data", required=True, help="path to dataset")
    parser.add_argument("--model", default="x-ai/grok-3-mini", help="Model name")
    parser.add_argument("--use_local_model", action="store_true", help="Use local model")
    parser.add_argument("--launch_model", action="store_true", help="Launch local vLLM server (requires --use_local_model)")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size (default 8).")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-concurrency", type=int, default=256, help="Simultaneous LLM calls (default 256).")
    parser.add_argument("--output", required=True, help="path to output (HuggingFace Hub)")
    return parser.parse_args()

if __name__ == "__main__":
    asyncio.run(main_async(cli()))
