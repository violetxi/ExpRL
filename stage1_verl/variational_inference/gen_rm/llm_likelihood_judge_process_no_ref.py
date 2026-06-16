#!/usr/bin/env python3
"""
llm_likelihood_judge_process_no_ref.py – Async batch evaluator for PARTIAL reasoning trace quality WITHOUT reference solution.

Each input must contain:
    { "problem": "...", "reasoning_trace": "..." }

This judge evaluates PARTIAL/INCOMPLETE reasoning WITHOUT seeing the ground truth solution.
The reasoning_trace may be just the first few steps, not a complete solution.

Example:
    python llm_likelihood_judge_process_no_ref.py --data dataset_path --model gpt-4o-mini
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


SYSTEM_PROMPT = """You are an expert mathematician and a meticulous AI reasoning evaluator. Your task is to assess the quality of a "Generated Reasoning" trace that may be **partial or incomplete**.

**Important:** You will NOT be given a reference solution. You must independently evaluate whether this partial reasoning is mathematically sound and represents good progress on the problem.

You are evaluating the **quality and correctness of the reasoning steps provided so far**, even if they don't constitute a complete solution."""


USER_PROMPT = """### Math Problem
{problem}

### Generated Reasoning (Partial/Incomplete)
{generated_reasoning}

### Instructions

First, provide a step-by-step analysis of the reasoning quality. In your analysis, consider the following:

* **Mathematical Correctness:** Are the steps provided so far mathematically correct? Are there any errors in calculations, formulas, or logical deductions up to this point?
* **Logical Soundness:** Do the reasoning steps follow logically from the problem statement and from each other? Are there any contradictions or illogical jumps?
* **Progress & Direction:** Do these partial steps represent meaningful progress on the problem? Are they moving in a productive direction, or do they seem like a "dead end" or tangential to solving the problem?
* **Clarity & Coherence:** Are the reasoning steps clear and well-explained? Can you follow the logic?
* **Potential for Continuation:** Even though incomplete, do these steps set up a reasonable foundation for continuing toward a correct solution? Or have they already made errors that would prevent arriving at a correct answer?

After your analysis, provide a single numerical score on the 5-point Likert scale defined below.

**Likert Scale:**
* **1 (Very Poor):** The partial reasoning contains major mathematical errors, is logically unsound, or heads in a completely wrong direction.
* **2 (Poor):** The partial reasoning has significant flaws or errors that make it unlikely to lead to a correct solution if continued. It may be a "dead end" or too vague to be useful.
* **3 (Acceptable):** The partial reasoning is generally sound but may have minor issues, be somewhat incomplete/trivial, or not represent substantial progress. It's not wrong, but not particularly strong either.
* **4 (Good):** The partial reasoning is mathematically correct, logically sound, and represents clear progress on the problem. It sets up a good foundation for reaching a solution.
* **5 (Excellent):** The partial reasoning is exemplary—mathematically rigorous, logically coherent, and represents significant, correct progress toward solving the problem. The next steps toward a complete solution are clear.

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
    """Load dataset and extract problem and partial reasoning (no reference solution)."""
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
        original_score = item["original_score"]

        jobs.append((problem, reasoning_trace))
        output_data["problem"].append(problem)
        output_data["problem_index"].append(problem_index)
        output_data["original_response"].append(original_response)
        output_data["reasoning_trace"].append(reasoning_trace)
        output_data["step_index"].append(step_index)
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
    parser = argparse.ArgumentParser(description="Async LLM evaluator for PARTIAL reasoning traces WITHOUT reference solution.")
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
