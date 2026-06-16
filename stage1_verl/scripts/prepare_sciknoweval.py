"""Build SciKnowEval-OE (open-ended, difficult) test parquets for vllm_generation eval.

Filters the `hicai-zju/SciKnowEval` HF dataset to the same slice that's pushed
to `violetxi/sciKnowEval-OE` (see scripts/build_and_push_sciknoweval_oe.py):

  Step 1: type == "open-ended-qa"           (drops MC / T-F / fill / extract)
  Step 2: answer.strip() is non-empty       (defensive)
  Step 3: level in {"L4", "L5"}             (only difficult cognitive levels)
  Step 4: subtask NOT in DROP_SUBTASKS_NON_NL
          (drops subtasks whose INPUT is dominated by non-natural-language
           tokens — AA sequences, SMILES — since the bottleneck there is
           encoding fluency rather than science reasoning / knowledge)

Writes one parquet per source version under data/instruct/<task>/test.parquet
in the schema expected by verl.trainer.vllm_generation_ray_dp
(data_source / prompt / ability / reward_model / extra_info).

Usage:
    python -m scripts.prepare_sciknoweval \\
        --base_dir data/instruct \\
        --tasks sciknoweval_oe_v1
"""

import argparse
import json
import os

from datasets import Dataset
from huggingface_hub import hf_hub_download

REPO = "hicai-zju/SciKnowEval"
DATA_SOURCE = "hicai-zju/SciKnowEval"  # routed to the Gemini-judge scorer in verl
SHARDS = {
    "v1": "data/v1/sciknoweval_test_v1.jsonl",
    "v2": "data/v2/sciknoweval_test_v2.jsonl",
}

KEEP_LEVELS = {"L4", "L5"}
DROP_SUBTASKS_NON_NL = {
    "protein_description_generation",  # input is an amino-acid sequence
    "molecule_captioning",              # input is a SMILES string
}


def _to_verl_row(example, idx: int):
    """Map a raw SciKnowEval row to the verl prompt-eval schema."""
    sys_prompt = (example.get("prompt") or {}).get("default", "") or ""
    question = example["question"]
    answer = example["answer"]
    details = example.get("details") or {}

    chat = []
    if sys_prompt:
        chat.append({"role": "system", "content": sys_prompt})
    chat.append({"role": "user", "content": question})

    return {
        "data_source": DATA_SOURCE,
        "prompt": chat,
        "ability": "Science",
        "reward_model": {"style": "rule", "ground_truth": answer},
        "extra_info": {
            "split": "test",
            "index": idx,
            # `question` is duplicated here so the LLM judge can see it at score
            # time without needing to plumb the prompt column into the scorer.
            "question": question,
            "domain": example.get("domain"),
            "level": details.get("level"),
            "task": details.get("task"),
            "subtask": details.get("subtask"),
            "source": details.get("source"),
            "type": example.get("type"),
        },
    }


def build_sciknoweval(version: str) -> Dataset:
    if version not in SHARDS:
        raise ValueError(f"Unknown version '{version}'. Choices: {list(SHARDS)}")
    print(f"Downloading SciKnowEval {version} from {REPO}/{SHARDS[version]} ...", flush=True)
    path = hf_hub_download(repo_id=REPO, filename=SHARDS[version], repo_type="dataset")

    rows = []
    n_total = 0
    n_step1 = 0
    n_step2 = 0
    n_step3 = 0
    n_step4 = 0
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            n_total += 1
            # Step 1: open-ended only
            if r.get("type") != "open-ended-qa":
                continue
            n_step1 += 1
            # Step 2: non-empty answer
            if not ((r.get("answer") or "").strip()):
                continue
            n_step2 += 1
            # Step 3: difficult cognitive levels
            details = r.get("details") or {}
            if details.get("level") not in KEEP_LEVELS:
                continue
            n_step3 += 1
            # Step 4: drop subtasks with non-NL inputs
            if details.get("subtask") in DROP_SUBTASKS_NON_NL:
                continue
            n_step4 += 1
            rows.append(r)

    print(
        f"  raw rows: {n_total}\n"
        f"  step 1 (open-ended-qa):                       -> {n_step1}\n"
        f"  step 2 (non-empty answer):                    -> {n_step2}\n"
        f"  step 3 (level in {sorted(KEEP_LEVELS)}):                  -> {n_step3}\n"
        f"  step 4 (drop {sorted(DROP_SUBTASKS_NON_NL)}): -> {n_step4}",
        flush=True,
    )

    mapped = [_to_verl_row(r, i) for i, r in enumerate(rows)]
    return Dataset.from_list(mapped)


TASK2BUILDER = {
    "sciknoweval_oe_v1": ("sciknoweval_oe_v1", lambda: build_sciknoweval("v1")),
    "sciknoweval_oe_v2": ("sciknoweval_oe_v2", lambda: build_sciknoweval("v2")),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", default="data/instruct")
    parser.add_argument(
        "--tasks",
        default="sciknoweval_oe_v1",
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
