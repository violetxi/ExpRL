"""Build LiveCodeBench (code_generation_lite) test parquet for the verl eval pipeline.

Ports recipe/r1/data_process.py::build_livecodebench_dataset() into the
`data/instruct/<bench>/test.parquet` layout the rest of the inference scripts
expect (columns: data_source, prompt, ability, reward_model, extra_info).

Default contest window matches DeepSeek-R1's reported eval (2024-08 -> 2025-01).
"""

import argparse
import base64
import json
import os
import pickle
import zlib
from functools import partial

import datasets


DEFAULT_DATA_SOURCE = "livecodebench/code_generation_lite"
DEFAULT_LOCAL_DIR = "data/instruct/livecodebench"
DEFAULT_START_DATE = "2024-08-00T00:00:00"
DEFAULT_END_DATE = "2025-01-00T00:00:00"


def _build_query_prompt(example):
    query_prompt = (
        "You will be given a question (problem specification) and will generate a correct Python program "
        f"that matches the specification and passes all tests.\n\nQuestion: {example['question_content']}\n\n"
    )
    if example["starter_code"]:
        query_prompt += (
            "You will use the following starter code to write the solution to the problem and enclose your "
            f"code within delimiters.\n```python\n{example['starter_code']}\n```"
        )
    else:
        query_prompt += (
            "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test "
            "on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python "
            "program runs, it reads the inputs, runs the algorithm and writes output to STDOUT."
            "```python\n# YOUR CODE HERE\n```"
        )
    return query_prompt


def _encode_test_cases(example):
    public_test_cases = json.loads(example["public_test_cases"])
    try:
        private_test_cases = json.loads(example["private_test_cases"])
    except Exception:
        private_test_cases = json.loads(
            pickle.loads(zlib.decompress(base64.b64decode(example["private_test_cases"].encode("utf-8"))))
        )
    full_test_cases = public_test_cases + private_test_cases

    metadata = json.loads(example["metadata"])
    test_cases = {
        "inputs": [t["input"] for t in full_test_cases],
        "outputs": [t["output"] for t in full_test_cases],
        "fn_name": metadata.get("func_name", None),
    }
    return base64.b64encode(zlib.compress(pickle.dumps(json.dumps(test_cases)))).decode("utf-8")


def make_map_fn(data_source, split):
    def process_fn(example, idx):
        question = _build_query_prompt(example)
        ground_truth = _encode_test_cases(example)

        prompt = [
            {
                "role": "system",
                "content": "You are a helpful programming assistant. Solve the problem and return a complete Python program.",
            },
            {"role": "user", "content": question},
        ]

        return {
            "data_source": data_source,
            "prompt": prompt,
            "ability": "code",
            "reward_model": {"style": "rule", "ground_truth": ground_truth},
            "extra_info": {
                "index": idx,
                "split": split,
                "question": question,
                "question_id": example.get("question_id", ""),
                "contest_date": example.get("contest_date", ""),
            },
        }

    return process_fn


def main():
    parser = argparse.ArgumentParser(description="Build LiveCodeBench test parquet for the verl eval pipeline.")
    parser.add_argument("--data_source", default=DEFAULT_DATA_SOURCE)
    parser.add_argument("--local_dir", default=DEFAULT_LOCAL_DIR)
    parser.add_argument("--start_date", default=DEFAULT_START_DATE, help="Inclusive lower bound on contest_date.")
    parser.add_argument("--end_date", default=DEFAULT_END_DATE, help="Exclusive upper bound on contest_date.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--num_proc", type=int, default=8)
    args = parser.parse_args()

    print(f"Loading {args.data_source} (split={args.split}) from huggingface...", flush=True)
    dataset = datasets.load_dataset(args.data_source, split=args.split, trust_remote_code=True)

    print(f"Filtering contest_date in [{args.start_date}, {args.end_date})...", flush=True)
    dataset = dataset.filter(lambda line: args.start_date <= line["contest_date"] < args.end_date)
    print(f"Filtered dataset size: {len(dataset)}", flush=True)

    columns_to_keep = ["data_source", "prompt", "ability", "reward_model", "extra_info"]
    dataset = dataset.map(
        function=make_map_fn(args.data_source, args.split),
        with_indices=True,
        num_proc=args.num_proc,
        remove_columns=dataset.column_names,
    )
    dataset = dataset.select_columns(columns_to_keep)

    os.makedirs(args.local_dir, exist_ok=True)
    out_path = os.path.join(args.local_dir, "test.parquet")
    dataset.to_parquet(out_path)

    print(f"Wrote {len(dataset)} rows to {out_path}")
    print("Sample row prompt[user]:")
    print(dataset[0]["prompt"][1]["content"][:500])


if __name__ == "__main__":
    main()
