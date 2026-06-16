"""Build LiveCodeBench (v5 / v6) and GPQA Diamond test parquets for vllm_generation eval.

Writes one parquet per task under data/instruct/<task>/test.parquet
in the schema expected by verl.trainer.vllm_generation_ray_dp
(data_source / prompt / ability / reward_model / extra_info).

LiveCodeBench
=============
`livecodebench/code_generation_lite` is a script-based dataset and
datasets>=4 dropped script loading, so we pull the underlying jsonl shards
directly from HF Hub. Each `testN.jsonl` is the *delta* added in release_vN
(non-overlapping across releases), so the cumulative union of shards 1..N is
release_vN.

We expose:
- livecodebench_v5 : shards 1..5  -> 880 problems  (the standard "LCB v5" set)
- livecodebench_v6 : shards 1..6  -> 1055 problems (the standard "LCB v6" set)

No date filter is applied — `LCB vN` in the literature normally refers to the
*full* release set. If you want a contamination-window slice (e.g. R1's
2024-08 -> 2025-01), edit R1_WINDOW_LO/HI and call _filter_dates() below.

GPQA Diamond
============
Reuses recipe.r1.data_process.build_gpqa_dimond_dataset unchanged.

Usage:
    python -m scripts.prepare_livecodebench_gpqa \\
        --base_dir data/instruct \\
        --tasks livecodebench_v5,livecodebench_v6,gpqa_diamond
"""

import argparse
import base64
import json
import os
import pickle
import zlib
from datetime import datetime
from typing import List, Optional, Tuple

from datasets import load_dataset
from huggingface_hub import hf_hub_download

LCB_REPO = "livecodebench/code_generation_lite"
LCB_DATA_SOURCE = "livecodebench/code_generation_lite"  # matches verl reward-score routing
LCB_RELEASE_SHARDS = {
    "v1": ["test.jsonl"],
    "v2": ["test.jsonl", "test2.jsonl"],
    "v3": ["test.jsonl", "test2.jsonl", "test3.jsonl"],
    "v4": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl"],
    "v5": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl"],
    "v6": ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl", "test5.jsonl", "test6.jsonl"],
}

# Optional contamination-control filter (off by default). The R1 paper used these bounds.
R1_WINDOW_LO = datetime(2024, 8, 1)
R1_WINDOW_HI = datetime(2025, 1, 1)


def _process_lcb_row(example):
    """Build (prompt, packed_test_cases) for a LiveCodeBench problem (R1 prompt template)."""
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
    packed = base64.b64encode(zlib.compress(pickle.dumps(json.dumps(test_cases)))).decode("utf-8")
    return query_prompt, packed


def _lcb_example_map(example, idx):
    question, solution = _process_lcb_row(example)
    return {
        "data_source": LCB_DATA_SOURCE,
        "prompt": [{"role": "user", "content": question}],
        "ability": "Code",
        "reward_model": {"style": "rule", "ground_truth": solution},
        "extra_info": {"split": "test", "index": idx},
    }


def _filter_dates(ds, lo: datetime, hi: datetime):
    """Optional helper: keep only rows whose contest_date is in [lo, hi)."""
    return ds.filter(lambda x: lo <= x["contest_date"] < hi)


def build_livecodebench(release: str, window: Optional[Tuple[datetime, datetime]] = None):
    """Build a LiveCodeBench dataset for the given release ('v1'..'v6').

    Set `window=(lo, hi)` to additionally filter by contest_date.
    """
    if release not in LCB_RELEASE_SHARDS:
        raise ValueError(f"Unknown release '{release}'. Choices: {list(LCB_RELEASE_SHARDS)}")
    shards = LCB_RELEASE_SHARDS[release]
    print(f"LiveCodeBench {release}: downloading {len(shards)} shard(s) from {LCB_REPO} ...", flush=True)
    paths = [hf_hub_download(repo_id=LCB_REPO, filename=f, repo_type="dataset") for f in shards]
    ds = load_dataset("json", data_files=paths, split="train")
    # Defensive dedupe by question_id (shards should be non-overlapping but stay safe).
    seen, keep_idx = {}, []
    for i, qid in enumerate(ds["question_id"]):
        seen[qid] = i
    keep_idx = sorted(seen.values())
    if len(keep_idx) != len(ds):
        ds = ds.select(keep_idx)
    print(f"  release_{release} unique problems: {len(ds)}", flush=True)

    if window is not None:
        lo, hi = window
        ds = _filter_dates(ds, lo, hi)
        print(f"  after date filter [{lo.date()}, {hi.date()}): {len(ds)}", flush=True)

    ds = ds.map(_lcb_example_map, with_indices=True, remove_columns=ds.column_names, num_proc=8)
    return ds


def build_gpqa_diamond():
    from recipe.r1.data_process import build_gpqa_dimond_dataset as _builder

    return _builder()


TASK2BUILDER = {
    "livecodebench_v5": ("livecodebench_v5", lambda: build_livecodebench("v5")),
    "livecodebench_v6": ("livecodebench_v6", lambda: build_livecodebench("v6")),
    "gpqa_diamond": ("gpqa_diamond", build_gpqa_diamond),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default="data/instruct")
    parser.add_argument(
        "--tasks",
        default="livecodebench_v5,livecodebench_v6,gpqa_diamond",
        help="Comma-separated subset of " + ",".join(TASK2BUILDER),
    )
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    for task in tasks:
        if task not in TASK2BUILDER:
            raise ValueError(f"Unknown task '{task}'. Choices: {list(TASK2BUILDER)}")
        subdir, builder = TASK2BUILDER[task]
        out_dir = os.path.join(args.base_dir, subdir)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "test.parquet")
        print(f"\n=== {task} -> {out_path} ===", flush=True)
        ds = builder()
        ds.to_parquet(out_path)
        print(f"Wrote {len(ds)} rows to {out_path}", flush=True)


if __name__ == "__main__":
    main()
