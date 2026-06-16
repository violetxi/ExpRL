"""Plot 3-panel calibration figure for the multi-ref judge calibration runs.

Loads a judge dataset that has `reward`, `llm_score_ref`, `llm_score_noref`,
`llm_score_wrongref` columns (produced by `--multi_ref` runs of
llm_likelihood_judge*.py) and saves a side-by-side score-distribution figure
matching the style of judge_smoke100_score_dist*.png.

Usage:
    python scripts/plot_judge_multi_ref_calibration.py \
        --dataset violetxi/Qwen4b-Instruct-Judge-lcb-v6-calibration-multi-ref \
        --title "Qwen4B-Instruct judge on LCB v6 calibration (Qwen8B rollouts)" \
        --x_label_long "1=Wrong/Won't Run ➜ 5=Functionally Equivalent" \
        --out judge_lcb_v6_calibration_multi_ref_score_dist.png
"""

import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset


SCORES = [1, 2, 3, 4, 5]
PASS_COLOR = "#2ca02c"
FAIL_COLOR = "#d62728"
CONDITIONS = [
    ("ref",      "Correct reference",  "llm_score_ref"),
    ("noref",    "No reference",       "llm_score_noref"),
    ("wrongref", "Wrong reference",    "llm_score_wrongref"),
]


def panel_data(df: pd.DataFrame, col: str, reward_col: str = "reward"):
    sub = df.dropna(subset=[col])
    pass_rows = sub[sub[reward_col] == 1]
    fail_rows = sub[sub[reward_col] == 0]
    pass_counts = [int((pass_rows[col] == s).sum()) for s in SCORES]
    fail_counts = [int((fail_rows[col] == s).sum()) for s in SCORES]
    n_pass = sum(pass_counts)
    n_fail = sum(fail_counts)
    pass_props = [c / n_pass if n_pass else 0.0 for c in pass_counts]
    fail_props = [c / n_fail if n_fail else 0.0 for c in fail_counts]
    return pass_counts, pass_props, fail_counts, fail_props, n_pass, n_fail


def auroc(df: pd.DataFrame, col: str, reward_col: str = "reward") -> float:
    sub = df.dropna(subset=[col, reward_col])
    y = sub[reward_col].values
    s = sub[col].values
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s)
    y_sorted = y[order]
    s_sorted = s[order]
    # Mann-Whitney U with tie-aware ranks: pandas.rank uses average ranks.
    ranks = pd.Series(s_sorted).rank().values
    pos_ranks_sum = ranks[y_sorted == 1].sum()
    U = pos_ranks_sum - n_pos * (n_pos + 1) / 2
    return float(U / (n_pos * n_neg))


def build_figure(df: pd.DataFrame, title: str, x_label_long: str, out_path: str,
                 reward_col: str = "reward"):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    x = np.arange(len(SCORES))
    width = 0.38

    for ax, (key, label, col) in zip(axes, CONDITIONS):
        pc, pp, fc, fp, n_pass, n_fail = panel_data(df, col, reward_col)
        a = auroc(df, col, reward_col)
        p5_pass = pc[-1] / n_pass if n_pass else 0
        p5_fail = fc[-1] / n_fail if n_fail else 0

        bars_p = ax.bar(x - width / 2, pp, width,
                        color=PASS_COLOR, edgecolor="black",
                        label=f"{reward_col}=1 (pass), n={n_pass}")
        bars_f = ax.bar(x + width / 2, fp, width,
                        color=FAIL_COLOR, edgecolor="black",
                        label=f"{reward_col}=0 (fail), n={n_fail}")

        for bars, counts, props in [(bars_p, pc, pp), (bars_f, fc, fp)]:
            for b, c, p in zip(bars, counts, props):
                if c == 0:
                    continue
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015,
                        f"{p*100:.1f}%\n(n={c})",
                        ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(SCORES)
        ax.set_ylim(0, 1.15)
        ax.set_xlabel(f"llm_score ({x_label_long})")
        ax.set_title(
            f"{label} ({key})\n"
            f"AUROC={a:.3f}   P(5|pass)={p5_pass:.2f}   P(5|fail)={p5_fail:.2f}",
            fontsize=10,
        )
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.legend(loc="upper left", fontsize=9)

    axes[0].set_ylabel("proportion within reward group")
    fig.suptitle(title, fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"saved: {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, help="HF dataset path of multi-ref judge output")
    p.add_argument("--title", default="Judge calibration")
    p.add_argument("--x_label_long", default="1=Very Unlikely ➜ 5=Very Likely",
                   help="Long-form description that follows the 'llm_score' axis label")
    p.add_argument("--out", required=True, help="Output PNG path")
    p.add_argument("--reward_col", default="reward",
                   help="Column holding the 0/1 ground-truth label (e.g. 'original_score' for sciknow)")
    args = p.parse_args()

    ds = load_dataset(args.dataset, split="train")
    df = ds.to_pandas()
    print(f"loaded {len(df)} rows from {args.dataset}")
    build_figure(df, args.title, args.x_label_long, args.out, args.reward_col)


if __name__ == "__main__":
    main()
