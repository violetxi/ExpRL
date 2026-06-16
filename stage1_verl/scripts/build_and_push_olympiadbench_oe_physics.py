"""Build the OlympiadBench open-ended (OE) physics dataset in verl schema.

Source: Hothan/OlympiadBench, config `OE_TO_physics_en_COMP`
        = Open-Ended / Text-Only / Physics / English / Competition (236 rows).

This is the ENGLISH-ONLY, TEXT-ONLY, OPEN-ENDED physics slice — no multimodal
rows, no MCQ. Every row is graded at rollout time by the Gemini Flash Lite LLM
judge (verl/utils/reward_score/olympiadbench_oe_judge.py), routed on the
`data_source` tag below.

Each row is rewritten to the verl-compatible schema
(data_source / prompt / ability / reward_model / extra_info):
  - prompt    : system instruction + (context constants + question)
  - reward_model.ground_truth : the official final_answer (multi-part joined)
  - extra_info: carries `question` (full text, for the judge), plus the
                physics-grading hints `unit`, `error`, `answer_type`,
                `is_multiple_answer`, `subfield`.

Usage:
    # local only (writes data/instruct/olympiadbench_oe_physics/train.parquet)
    python -m scripts.build_and_push_olympiadbench_oe_physics --no_push

    # also push to HF
    python -m scripts.build_and_push_olympiadbench_oe_physics --repo violetxi/olympiadbench_oe_physics
"""

import argparse
import os
from collections import Counter

from datasets import Dataset, load_dataset

SRC_REPO = "Hothan/OlympiadBench"
SRC_CONFIG = "OE_TO_physics_en_COMP"
# Distinct tag (NOT "Hothan/OlympiadBench", which already routes to prime_math).
# This one routes to the Gemini OE physics judge in verl reward_score/__init__.py.
DATA_SOURCE_TAG = "violetxi/olympiadbench_oe_physics"

LOCAL_DIR = "data/instruct/olympiadbench_oe_physics"

SYSTEM_PROMPT = (
    "You are an expert physicist solving a competition physics problem. "
    "Work through the problem and give the final answer clearly at the end, "
    "prefixed with 'Final Answer:'. If the answer has a unit, include it."
)


def _join_answer(final_answer) -> str:
    """final_answer is a list[str]; join multi-part answers with ' ; '."""
    if isinstance(final_answer, (list, tuple)):
        parts = [str(a).strip() for a in final_answer if str(a).strip()]
        return " ; ".join(parts)
    return str(final_answer or "").strip()


def _to_verl_row(ex, idx: int):
    context = (ex.get("context") or "").strip()
    question = (ex.get("question") or "").strip()
    full_q = f"{context}\n\n{question}".strip() if context else question

    chat = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": full_q},
    ]
    return {
        "data_source": DATA_SOURCE_TAG,
        "prompt": chat,
        "ability": "Physics",
        "reward_model": {"style": "rule", "ground_truth": _join_answer(ex.get("final_answer"))},
        "extra_info": {
            "split": "test",
            "index": idx,
            "id": ex.get("id"),
            "question": full_q,  # full text (context + question) for the LLM judge
            "unit": ex.get("unit") or "",
            "error": ex.get("error") or "",
            "answer_type": ex.get("answer_type") or "",
            "is_multiple_answer": bool(ex.get("is_multiple_answer")),
            "subfield": ex.get("subfield") or "",
            "subject": ex.get("subject") or "Physics",
            "type": "open-ended-qa",
        },
    }


def build():
    print(f"--- Source: {SRC_REPO} :: {SRC_CONFIG} ---")
    ds = load_dataset(SRC_REPO, SRC_CONFIG, split="train")
    print(f"Loaded {len(ds)} rows.")

    rows = []
    n_dropped_empty = n_dropped_image = 0
    for i, ex in enumerate(ds):
        # Defensive: this config is text-only, but drop anything with an image.
        if any(ex.get(f"image_{k}") is not None for k in range(1, 10)):
            n_dropped_image += 1
            continue
        row = _to_verl_row(ex, i)
        if not row["reward_model"]["ground_truth"]:
            n_dropped_empty += 1
            continue
        rows.append(row)

    print(f"Dropped {n_dropped_image} rows with images (should be 0 for *_TO_*).")
    print(f"Dropped {n_dropped_empty} rows with empty final_answer.")
    print(f"Kept {len(rows)} rows.")

    by_type = Counter(r["extra_info"]["answer_type"] for r in rows)
    by_field = Counter(r["extra_info"]["subfield"] for r in rows)
    n_multi = sum(r["extra_info"]["is_multiple_answer"] for r in rows)
    print("\n=== Composition ===")
    print("answer_type:", dict(by_type))
    print("multi-part answers:", n_multi)
    print("subfields:", dict(by_field.most_common()))

    return Dataset.from_list(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="violetxi/olympiadbench_oe_physics")
    parser.add_argument("--public", action="store_true", help="Push as public (default private).")
    parser.add_argument("--no_push", action="store_true", help="Build/write locally without pushing to HF.")
    parser.add_argument("--local_dir", default=LOCAL_DIR)
    args = parser.parse_args()

    ds = build()

    os.makedirs(args.local_dir, exist_ok=True)
    out_path = os.path.join(args.local_dir, "train.parquet")
    ds.to_parquet(out_path)
    print(f"\nWrote local parquet: {out_path}")

    if args.no_push:
        print("--no_push set; skipping HF push.")
        return
    private = not args.public
    print(f"\nPushing to https://huggingface.co/datasets/{args.repo} (private={private}) ...")
    ds.push_to_hub(args.repo, split="train", private=private)
    print("Push complete.")


if __name__ == "__main__":
    main()
