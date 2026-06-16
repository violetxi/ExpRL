"""Build GPQA Diamond test parquet for the verl eval pipeline.

Ports recipe/r1/data_process.py::build_gpqa_dimond_dataset() into the
`data/instruct/<bench>/test.parquet` layout used by the existing inference
scripts (columns: data_source, prompt, ability, reward_model, extra_info).

Choice order is randomized per example. A fixed --seed is used so the answer
positions (and therefore the gold letter) are reproducible across runs.
"""

import argparse
import os
import random

import datasets


DEFAULT_DATA_SOURCE = "Idavidrein/gpqa"
DEFAULT_LOCAL_DIR = "data/instruct/gpqa_diamond"
DEFAULT_CONFIG = "gpqa_diamond"

GPQA_QUERY_TEMPLATE = (
    "Answer the following multiple choice question. The last line of your response should be of the following "
    "format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before "
    "answering.\n\n{Question}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}"
)


def make_map_fn(data_source, split, seed):
    rng = random.Random(seed)

    def process_fn(example, idx):
        choices = [
            example["Incorrect Answer 1"],
            example["Incorrect Answer 2"],
            example["Incorrect Answer 3"],
        ]
        rng.shuffle(choices)
        gold_index = rng.randint(0, 3)
        choices.insert(gold_index, example["Correct Answer"])

        question = GPQA_QUERY_TEMPLATE.format(
            A=choices[0], B=choices[1], C=choices[2], D=choices[3], Question=example["Question"]
        )
        gold_letter = "ABCD"[gold_index]

        prompt = [
            {"role": "user", "content": question},
        ]

        return {
            "data_source": data_source,
            "prompt": prompt,
            "ability": "science",
            "reward_model": {"style": "rule", "ground_truth": gold_letter},
            "extra_info": {
                "index": idx,
                "split": split,
                "question": question,
                "answer": gold_letter,
                "correct_answer_text": example["Correct Answer"],
                "subdomain": example.get("Subdomain", ""),
            },
        }

    return process_fn


def main():
    parser = argparse.ArgumentParser(description="Build GPQA Diamond test parquet for the verl eval pipeline.")
    parser.add_argument("--data_source", default=DEFAULT_DATA_SOURCE)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--local_dir", default=DEFAULT_LOCAL_DIR)
    parser.add_argument("--split", default="train", help="HF split to read (gpqa only ships 'train').")
    parser.add_argument("--seed", type=int, default=0, help="Seed for choice randomization (reproducibility).")
    args = parser.parse_args()

    print(f"Loading {args.data_source} (config={args.config}, split={args.split}) from huggingface...", flush=True)
    dataset = datasets.load_dataset(args.data_source, args.config, split=args.split)
    print(f"Loaded {len(dataset)} examples.", flush=True)

    columns_to_keep = ["data_source", "prompt", "ability", "reward_model", "extra_info"]
    dataset = dataset.map(
        function=make_map_fn(args.data_source, split="test", seed=args.seed),
        with_indices=True,
        remove_columns=dataset.column_names,
    )
    dataset = dataset.select_columns(columns_to_keep)

    os.makedirs(args.local_dir, exist_ok=True)
    out_path = os.path.join(args.local_dir, "test.parquet")
    dataset.to_parquet(out_path)

    print(f"Wrote {len(dataset)} rows to {out_path}")
    print("Sample row prompt[user]:")
    print(dataset[0]["prompt"][0]["content"][:500])
    print(f"Sample gold letter: {dataset[0]['reward_model']['ground_truth']}")


if __name__ == "__main__":
    main()
