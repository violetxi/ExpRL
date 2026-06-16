"""Build a SciKnowEval MCQ subset for RL training/oracle generation.

Pulls reasoning-rich MCQ rows from the upstream `hicai-zju/SciKnowEval` v2 dataset:
filters by (type=mcq-4-choices or mcq-2-choices), level in KEEP_LEVELS, task in KEEP_TASKS.
Default selection targets GPQA/MMLU transfer — pure reasoning, no safety/toxicity/recall.

Each row is written in the verl prompt-eval schema:
    data_source / prompt / ability / reward_model / extra_info

The user prompt embeds the question + lettered choices and asks for reasoning + a single
letter answer.  reward_model.ground_truth = answerKey letter (A/B/C/D).

Usage:
    python -m scripts.prepare_sciknoweval_mcq \\
        --base_dir data/instruct \\
        --task sciknoweval_mcq_l3_loose
"""
import argparse
import os

from datasets import Dataset, load_dataset

REPO = "hicai-zju/SciKnowEval"
CONFIG = "v2"
DATA_SOURCE = "hicai-zju/SciKnowEval"  # reuses the existing verl reward route

# Tasks we keep — pure reasoning, no safety/toxicity, no overly-specialized
# prediction tasks.  Matches the "loose" slice (~2400 rows) discussed in the
# oracle-stack planning conversation.
KEEP_TASKS_LOOSE = {
    "general_physics_calculation",     # ~800 — physics setup → solve
    "reaction_prediction",              # ~400 — organic chem reasoning
    "retrosynthesis",                   # ~300 — organic chem reasoning
    "molar_weight_calculation",         # ~600 — mechanical (atomic-weight sum)
    "molecule_structure_prediction",    # ~300 — mechanical (count atoms)
}
KEEP_TASKS_TIGHT = {
    "general_physics_calculation",
    "reaction_prediction",
    "retrosynthesis",
}
KEEP_LEVELS = {"L3"}  # L4 is overwhelmingly safety/toxicity trivia — skip


# Instruct the solver to reason first, then output the final letter on its own line.
# This overrides the upstream prompt (which says "directly give the answer without any
# explanation") so we capture chain-of-thought for the GenRM PR judge.
SYS_PROMPT = (
    "You are an expert scientist. Answer the following multiple-choice question. "
    "Reason step by step, then on the final line write exactly: "
    "`Final Answer: <LETTER>` (where <LETTER> is A, B, C, or D)."
)


def _format_user_prompt(question: str, choices: dict) -> str:
    """Render the question + 4 lettered choices in a single user-prompt block."""
    labels = choices.get("label") or []
    texts = choices.get("text") or []
    body = question.rstrip() + "\n\n"
    for label, text in zip(labels, texts):
        body += f"{label}. {text}\n"
    return body.rstrip()


def _to_verl_row(example, idx: int):
    details = example.get("details") or {}
    user_content = _format_user_prompt(example["question"], example.get("choices") or {})
    chat = [
        {"role": "system", "content": SYS_PROMPT},
        {"role": "user", "content": user_content},
    ]
    return {
        "data_source": DATA_SOURCE,
        "prompt": chat,
        "ability": "Science",
        "reward_model": {"style": "rule", "ground_truth": example.get("answerKey") or ""},
        "extra_info": {
            "split": "test",
            "index": idx,
            "question": example["question"],
            "choices": example.get("choices") or {},
            "answerKey": example.get("answerKey") or "",
            "domain": example.get("domain"),
            "level": details.get("level"),
            "task": details.get("task"),
            "subtask": details.get("subtask"),
            "source": details.get("source"),
            "type": example.get("type"),
        },
    }


def build_mcq(task_set: set[str], levels: set[str]) -> Dataset:
    print(f"Loading {REPO}/{CONFIG}/test ...", flush=True)
    ds = load_dataset(REPO, CONFIG, split="test")

    def _keep(r):
        if r.get("type") not in ("mcq-4-choices", "mcq-2-choices"):
            return False
        det = r.get("details") or {}
        if det.get("level") not in levels:
            return False
        if det.get("task") not in task_set:
            return False
        # require valid answerKey + choices
        if not (r.get("answerKey") or "").strip():
            return False
        ch = r.get("choices") or {}
        if not ch.get("text") or not ch.get("label"):
            return False
        return True

    filtered = ds.filter(_keep)
    print(f"  raw: {len(ds)} rows  →  filtered: {len(filtered)} rows", flush=True)

    # Per-task breakdown
    from collections import Counter
    by_task = Counter(r["details"]["task"] for r in filtered)
    for t, n in sorted(by_task.items(), key=lambda x: -x[1]):
        print(f"    {t}: {n}", flush=True)

    mapped = [_to_verl_row(r, i) for i, r in enumerate(filtered)]
    return Dataset.from_list(mapped)


PRESETS = {
    "sciknoweval_mcq_l3_tight":  (KEEP_TASKS_TIGHT, KEEP_LEVELS),
    "sciknoweval_mcq_l3_loose":  (KEEP_TASKS_LOOSE, KEEP_LEVELS),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default="data/instruct")
    ap.add_argument("--task", default="sciknoweval_mcq_l3_loose",
                    choices=list(PRESETS),
                    help="Which preset to build (controls task set).")
    args = ap.parse_args()

    task_set, levels = PRESETS[args.task]
    out_dir = os.path.join(args.base_dir, args.task)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "test.parquet")
    print(f"\n=== {args.task} -> {out_path} ===", flush=True)
    ds = build_mcq(task_set, levels)
    ds.to_parquet(out_path)
    print(f"Wrote {len(ds)} rows to {out_path}", flush=True)


if __name__ == "__main__":
    main()
