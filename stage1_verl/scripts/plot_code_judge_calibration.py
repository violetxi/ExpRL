#!/usr/bin/env python3
"""Side-by-side calibration plot for the LCB code judge.

Each panel shows, for each reward bucket (1 = pass, 0 = fail), the
proportion of rows that landed in each llm_score bin (1..5). Bars sum to
100% within each reward group. Counts and percentages are annotated on
top of each bar so the fail->5 leak is easy to read off.

Usage:
    python scripts/plot_code_judge_calibration.py \\
        --v1 <hf-dataset-of-judge-run-A> \\
        --v2 <hf-dataset-of-judge-run-B> \\
        --out <output-png-path>

Datasets must contain 'llm_score' and 'reward' columns.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from datasets import load_dataset


def load_df(path: str):
    ds = load_dataset(path, split="train")
    df = ds.to_pandas().dropna(subset=["llm_score", "reward"])
    df["llm_score"] = df["llm_score"].astype(int)
    return df


def split_scores(df):
    pass_scores = df.loc[df["reward"] == 1, "llm_score"].tolist()
    fail_scores = df.loc[df["reward"] == 0, "llm_score"].tolist()
    return pass_scores, fail_scores


def bucket_counts(scores):
    return np.array([sum(1 for s in scores if s == k) for k in range(1, 6)])


def draw_panel(ax, title, pass_scores, fail_scores):
    bins = np.arange(1, 6)
    pass_n = bucket_counts(pass_scores)
    fail_n = bucket_counts(fail_scores)
    n_pass = pass_n.sum()
    n_fail = fail_n.sum()
    pass_p = pass_n / max(n_pass, 1)
    fail_p = fail_n / max(n_fail, 1)

    width = 0.4
    ax.bar(bins - width / 2, pass_p, width=width, color="#2ca02c",
           edgecolor="black", label=f"reward=1 (pass, n={n_pass})")
    ax.bar(bins + width / 2, fail_p, width=width, color="#d62728",
           edgecolor="black", label=f"reward=0 (fail, n={n_fail})")

    for k_idx, k in enumerate(bins):
        if pass_n[k_idx] > 0:
            ax.text(k - width / 2, pass_p[k_idx] + 0.01,
                    f"{pass_p[k_idx]*100:.1f}%\n(n={pass_n[k_idx]})",
                    ha="center", va="bottom", fontsize=8)
        if fail_n[k_idx] > 0:
            ax.text(k + width / 2, fail_p[k_idx] + 0.01,
                    f"{fail_p[k_idx]*100:.1f}%\n(n={fail_n[k_idx]})",
                    ha="center", va="bottom", fontsize=8)

    ax.set_title(title, fontsize=12)
    ax.set_xlabel("llm_score (1=Very Unlikely → 5=Very Likely)")
    ax.set_ylabel("proportion within reward group")
    ax.set_xticks(bins)
    ax.set_ylim(0, 1.2)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(loc="upper left", fontsize=9)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1", required=True, help="HF dataset (left panel)")
    parser.add_argument("--v2", required=True, help="HF dataset (right panel)")
    parser.add_argument("--out", required=True,
                        help="Output PNG path (relative to repo root)")
    args = parser.parse_args()

    print(f"Loading v1: {args.v1}")
    v1_df = load_df(args.v1)
    print(f"Loading v2: {args.v2}")
    v2_df = load_df(args.v2)

    common = set(v1_df["prompt_idx"]).intersection(set(v2_df["prompt_idx"]))
    print(f"v1 rows: {len(v1_df)} | v2 rows: {len(v2_df)} | "
          f"intersection on prompt_idx: {len(common)}")
    v1_df = v1_df[v1_df["prompt_idx"].isin(common)].sort_values("prompt_idx")
    v2_df = v2_df[v2_df["prompt_idx"].isin(common)].sort_values("prompt_idx")
    v1_pass, v1_fail = split_scores(v1_df)
    v2_pass, v2_fail = split_scores(v2_df)
    print(f"  v1 pass={len(v1_pass)} fail={len(v1_fail)}")
    print(f"  v2 pass={len(v2_pass)} fail={len(v2_fail)}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5), sharey=True)
    draw_panel(ax1, "Prompt v1 (original)", v1_pass, v1_fail)
    draw_panel(ax2, "Prompt v2 (checklist + anti-anchoring + tie-break)",
               v2_pass, v2_fail)
    fig.suptitle("Qwen4B-Instruct judge on LCB v6 stage1: llm_score distribution by reward",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = Path(args.out)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved {out_path}")

    print("\n--- delta summary (fail → 5 is the leak we care about) ---")
    print(f"v1 fail→5: {bucket_counts(v1_fail)[4]}/{len(v1_fail)} "
          f"({bucket_counts(v1_fail)[4]/max(len(v1_fail),1)*100:.1f}%)")
    print(f"v2 fail→5: {bucket_counts(v2_fail)[4]}/{len(v2_fail)} "
          f"({bucket_counts(v2_fail)[4]/max(len(v2_fail),1)*100:.1f}%)")
    print(f"v1 pass→5: {bucket_counts(v1_pass)[4]}/{len(v1_pass)} "
          f"({bucket_counts(v1_pass)[4]/max(len(v1_pass),1)*100:.1f}%)")
    print(f"v2 pass→5: {bucket_counts(v2_pass)[4]}/{len(v2_pass)} "
          f"({bucket_counts(v2_pass)[4]/max(len(v2_pass),1)*100:.1f}%)")


if __name__ == "__main__":
    main()
