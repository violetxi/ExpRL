"""Build the open-ended SciKnowEval (sciKnowEval-OE) dataset and push to HF.

Starts from `hicai-zju/SciKnowEval` v1 and applies four filtering steps in order:

  Step 1: type == "open-ended-qa"              (drops MC / T-F / fill / extract)
  Step 2: answer.strip() is non-empty          (defensive)
  Step 3: level in {"L4", "L5"}                (only difficult cognitive levels)
  Step 4: subtask NOT in DROP_SUBTASKS_NON_NL  (drop subtasks whose INPUT is
          dominated by non-natural-language tokens — AA sequences, SMILES)
  Step 5: rewrite to verl-compatible schema
          (data_source / prompt / ability / reward_model / extra_info)

The rationale: we want open-ended L4/L5 problems with broad cross-domain
coverage, including knowledge-retrieval tasks, but excluding those where
solving the problem requires interpreting a non-natural-language input
encoding (protein AA sequences, SMILES). Solving those is more about
encoding/decoding fluency than scientific reasoning or knowledge.

Usage:
    python -m scripts.build_and_push_sciknoweval_oe \\
        --repo violetxi/sciKnowEval-OE --private
"""

import argparse
import json
from collections import Counter

from datasets import Dataset
from huggingface_hub import hf_hub_download

REPO_SRC = "hicai-zju/SciKnowEval"
SHARD_SRC = "data/v1/sciknoweval_test_v1.jsonl"
DATA_SOURCE_TAG = "hicai-zju/SciKnowEval"  # what the verl scorer routes on

# Step 3 — keep only difficult cognitive levels.
KEEP_LEVELS = {"L4", "L5"}

# Step 4 — drop subtasks whose INPUT is dominated by non-natural-language tokens.
# Verified by random sampling: protein_description_generation has an AA
# sequence in 100% of inputs; molecule_captioning has SMILES in essentially
# every input. Solving these is about decoding the encoding, not science
# reasoning / knowledge, so they don't fit a general "open-ended science" eval.
DROP_SUBTASKS_NON_NL = {
    "protein_description_generation",
    "molecule_captioning",
}


def _to_verl_row(example, idx: int):
    sys_prompt = (example.get("prompt") or {}).get("default", "") or ""
    question = example["question"]
    answer = example["answer"]
    details = example.get("details") or {}
    chat = []
    if sys_prompt:
        chat.append({"role": "system", "content": sys_prompt})
    chat.append({"role": "user", "content": question})
    return {
        "data_source": DATA_SOURCE_TAG,
        "prompt": chat,
        "ability": "Science",
        "reward_model": {"style": "rule", "ground_truth": answer},
        "extra_info": {
            "split": "test",
            "index": idx,
            "question": question,  # for the LLM-judge scorer
            "domain": example.get("domain"),
            "level": details.get("level"),
            "task": details.get("task"),
            "subtask": details.get("subtask"),
            "source": details.get("source"),
            "type": example.get("type"),
        },
    }


def build_strict():
    print(f"--- Source: {REPO_SRC}/{SHARD_SRC} ---")
    path = hf_hub_download(repo_id=REPO_SRC, filename=SHARD_SRC, repo_type="dataset")

    n_raw = n_step1 = n_step2 = n_step3 = n_step4 = 0
    kept = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            n_raw += 1
            # Step 1: open-ended only
            if r.get("type") != "open-ended-qa":
                continue
            n_step1 += 1
            # Step 2: non-empty answer
            if not ((r.get("answer") or "").strip()):
                continue
            n_step2 += 1
            # Step 3: difficult levels only
            details = r.get("details") or {}
            lvl = details.get("level")
            if lvl not in KEEP_LEVELS:
                continue
            n_step3 += 1
            # Step 4: drop non-NL-input subtasks
            subtask = details.get("subtask")
            if subtask in DROP_SUBTASKS_NON_NL:
                continue
            n_step4 += 1
            kept.append(r)

    print(f"\nStep 1 — keep type == 'open-ended-qa':       {n_raw:>6}  ->  {n_step1:>6}")
    print(f"Step 2 — keep non-empty answer:              {n_step1:>6}  ->  {n_step2:>6}")
    print(f"Step 3 — keep level in {sorted(KEEP_LEVELS)}:          {n_step2:>6}  ->  {n_step3:>6}")
    print(f"Step 4 — drop non-NL-input subtasks {sorted(DROP_SUBTASKS_NON_NL)}: {n_step3:>6}  ->  {n_step4:>6}")
    print(f"Step 5 — rewrite to verl schema:             {n_step4} rows")

    rows = [_to_verl_row(r, i) for i, r in enumerate(kept)]

    # Composition summary
    print("\n=== Final composition (sciKnowEval-OE) ===")
    by_level = Counter()
    by_domain = Counter()
    by_lvl_dom = Counter()
    by_subtask = Counter()
    for r in rows:
        ei = r["extra_info"]
        by_level[ei["level"]] += 1
        by_domain[ei["domain"]] += 1
        by_lvl_dom[(ei["level"], ei["domain"])] += 1
        by_subtask[(ei["level"], ei["domain"], ei["subtask"])] += 1

    print(f"Total rows: {len(rows)}")
    print("\nBy level:")
    for k in sorted(by_level): print(f"  {k}: {by_level[k]}")
    print("\nBy domain:")
    for k, v in sorted(by_domain.items(), key=lambda x: -x[1]): print(f"  {k}: {v}")
    print("\nBy (level, domain):")
    for k, v in sorted(by_lvl_dom.items(), key=lambda x: (x[0][0], -x[1])):
        print(f"  {k[0]} / {k[1]}: {v}")
    print("\nBy (level, domain, subtask):")
    for k, v in sorted(by_subtask.items(), key=lambda x: (x[0][0], x[0][1], -x[1])):
        print(f"  {k[0]} / {k[1]} / {k[2]}: {v}")

    return Dataset.from_list(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="violetxi/sciKnowEval-OE")
    parser.add_argument("--private", action="store_true", default=True,
                        help="Push as a private dataset (default).")
    parser.add_argument("--public", action="store_true",
                        help="Push as a public dataset (overrides --private).")
    parser.add_argument("--no_push", action="store_true",
                        help="Build the dataset locally without pushing.")
    args = parser.parse_args()
    private = not args.public

    ds = build_strict()
    if args.no_push:
        print("\n--no_push set; skipping HF push.")
        return
    print(f"\nPushing to https://huggingface.co/datasets/{args.repo} (private={private}) ...")
    ds.push_to_hub(args.repo, split="test", private=private)
    print("Push complete.")


if __name__ == "__main__":
    main()
