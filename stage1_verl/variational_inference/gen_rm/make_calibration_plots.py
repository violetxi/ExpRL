#!/usr/bin/env python3
"""
Misplacement / reliability summary for the multi-ref LLM judge outputs produced
by the slurm_judge_qwen*_outcome_{math,sciknow}.sh runs.

Judge emits a Likert score in {1..5} per condition in columns
llm_score_{ref,noref,wrongref}. Ground truth is the BINARY correctness of the
underlying rollout (`score` for math, `original_score` for sciknow).

Because the label is binary, absolute-probability calibration (ECE/Brier) is
misleading: the label cannot distinguish a genuinely-correct "4" from a "5", or
a clearly-wrong "1" from a borderline "3", so it penalizes legitimate
within-class score spread. Instead we quantify *misplacement*: threshold the
Likert at >=4 ("judge asserts correct") vs <=3 ("judge asserts incorrect") and
measure how often the judge lands on the wrong side of the binary boundary:
  - FNR = P(score < 4 | correct)    correct rollouts misplaced as low
  - FPR = P(score >= 4 | incorrect) incorrect rollouts misplaced as high
  - overall = (FN + FP) / N         pooled (base-rate weighted)
  - AUROC                           threshold-free discrimination (companion)

Outputs, under calibration_figs/: one PNG per dataset (1x4 grid of model sizes,
each overlaying the 3 judge conditions) plus
  - misplacement_table.md / .csv    one markdown table per dataset, rows = size
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_dataset

OUT_DIR = os.path.join(os.path.dirname(__file__), "calibration_figs")
os.makedirs(OUT_DIR, exist_ok=True)

# (size label, HF name prefix). Order = ascending model size.
SIZES = [
    ("0.6B", "Qwen0_6b-NoThink"),
    ("4B",   "Qwen4b-Instruct"),
    ("8B",   "Qwen8b-NoThink"),
    ("14B",  "Qwen14b-NoThink"),
]

# domain key -> (HF name suffix, ground-truth column, pretty title)
DOMAINS = [
    ("math",        "Judge-int-qwen8b-multi-ref",                      "score",          "Math (INT / Qwen8B rollouts)"),
    ("sciknow-mcq", "Judge-sciknoweval-mcq-l3-loose-qwen8b-multi-ref", "original_score", "SciKnowEval MCQ (L3 loose)"),
    ("sciknow-oe",  "Judge-sciknoweval-oe-v1-qwen8b-multi-ref",        "original_score", "SciKnowEval OE (v1)"),
]

# Single-judge benchmarks: one judge size, full ref/noref/wrongref conditions
# (columns llm_score_{ref,noref,wrongref}). Unlike DOMAINS there's no size sweep.
# (dkey, hf_path, gt_col, judge_size, title)
SINGLE_JUDGE = [
    ("lcb", "violetxi/Qwen4b-Instruct-Judge-lcb-v6-calibration-multi-ref",
     "reward", "4B", "LiveCodeBench v6 (code)"),
]

LIKERT = [1, 2, 3, 4, 5]
PRED_P = {s: (s - 1) / 4.0 for s in LIKERT}   # 1->0.0 ... 5->1.0 (for plotting only)
THRESH = 4        # Likert >= THRESH => judge asserts "correct"; < THRESH => "incorrect"
BOUNDARY_X = (PRED_P[THRESH - 1] + PRED_P[THRESH]) / 2.0   # decision boundary in pred space

CONDITIONS = ["ref", "noref", "wrongref"]
COND_STYLE = {
    "ref":      {"color": "#1a9850", "label": "ref (correct reference)"},
    "noref":    {"color": "#4575b4", "label": "noref (no reference)"},
    "wrongref": {"color": "#d73027", "label": "wrongref (mismatched reference)"},
}


def load_one(prefix, suffix):
    path = f"violetxi/{prefix}-{suffix}"
    ds = load_dataset(path, split="train")
    return ds, path


def _auroc(scores, correct):
    """AUROC via the Mann-Whitney U statistic with average-rank tie handling.

    Threshold-free discrimination: does the judge rank correct rollouts above
    incorrect ones? Robust to the binary-label coarseness for the same reason
    the misplacement rate is -- only correct-vs-incorrect ordering matters, not
    the 4-vs-5 spread within a class. Returns NaN if either class is absent.
    """
    n_pos = int(correct.sum())
    n_neg = int((~correct).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ss = scores[order]
    ranks = np.empty(len(ss), dtype=float)
    i = 0
    while i < len(ss):                       # average rank (1-based) within tie groups
        j = i
        while j < len(ss) and ss[j] == ss[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1.0
        i = j
    rank_pos = ranks[correct[order]].sum()
    return float((rank_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def misplacement(llm_scores, y_true):
    """Return per-Likert reliability points + a dict of misplacement metrics.

    metrics keys: n, n_correct, n_incorrect,
      fnr      = P(score <  THRESH | correct),
      fpr      = P(score >= THRESH | incorrect),
      misplace = (FN + FP) / n   (pooled, base-rate weighted; inverts on imbalance),
      balanced = (FNR + FPR) / 2 (base-rate independent; = 1 - balanced accuracy),
      auroc.
    """
    scores, ys = [], []
    for s, y in zip(llm_scores, y_true):
        if s is None or y is None:
            continue
        try:
            s = int(s)
            y = float(y)
        except (TypeError, ValueError):
            continue
        if s not in PRED_P:
            continue
        scores.append(s)
        ys.append(y)
    scores = np.array(scores)
    ys = np.array(ys)
    n = len(ys)
    empty = dict(n=0, n_correct=0, n_incorrect=0,
                 fnr=float("nan"), fpr=float("nan"),
                 misplace=float("nan"), balanced=float("nan"), auroc=float("nan"))
    if n == 0:
        return [], empty

    # reliability points (empirical accuracy at each Likert level) -- for the plot
    pts = []
    for s in LIKERT:
        mask = scores == s
        nn = int(mask.sum())
        if nn:
            pts.append((PRED_P[s], float(ys[mask].mean()), nn))

    correct = ys >= 0.5                       # binary label; >=0.5 guards stray floats
    incorrect = ~correct
    nc = int(correct.sum())
    ni = int(incorrect.sum())
    judge_says_correct = scores >= THRESH
    fn = int((correct & ~judge_says_correct).sum())   # correct scored < THRESH
    fp = int((incorrect & judge_says_correct).sum())  # incorrect scored >= THRESH
    fnr = (fn / nc) if nc else float("nan")
    fpr = (fp / ni) if ni else float("nan")
    metrics = dict(
        n=n, n_correct=nc, n_incorrect=ni,
        fnr=fnr, fpr=fpr,
        misplace=(fn + fp) / n,          # pooled (base-rate weighted)
        balanced=(fnr + fpr) / 2.0,      # base-rate independent = 1 - balanced acc
        auroc=_auroc(scores, correct),
    )
    return pts, metrics


def make_dataset_fig(dkey, suffix, gt_col, title):
    """One PNG for a dataset: 1x4 grid of sizes, each overlaying the 3 conditions."""
    fig, axes = plt.subplots(1, len(SIZES), figsize=(4.6 * len(SIZES), 4.8), squeeze=False)
    axes = axes[0]
    summary = []  # rows for table
    for ax, (size_label, prefix) in zip(axes, SIZES):
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="perfect")
        ax.axvline(BOUNDARY_X, color="grey", ls=":", lw=1.2, alpha=0.7)
        try:
            ds, path = load_one(prefix, suffix)
        except Exception as e:
            print(f"[skip] {prefix}-{suffix}: {type(e).__name__}: {e}")
            ax.set_title(f"{size_label}  (missing)", fontsize=11)
            continue
        if gt_col not in ds.column_names:
            print(f"[skip] {path}: missing {gt_col}")
            continue
        for cond in CONDITIONS:
            col = f"llm_score_{cond}"
            if col not in ds.column_names:
                continue
            pts, m = misplacement(ds[col], ds[gt_col])
            if not pts:
                continue
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            ns = [p[2] for p in pts]
            c = COND_STYLE[cond]["color"]
            ax.plot(xs, ys, "-o", color=c, lw=2,
                    label=f"{cond} (misplace={m['balanced'] * 100:.0f}%)")
            ax.scatter(xs, ys, s=[20 + 240 * (n / max(ns)) for n in ns],
                       color=c, alpha=0.30, zorder=3)
            summary.append(dict(domain=dkey, size=size_label, condition=cond, **m))
        ax.set_title(f"Qwen3-{size_label}", fontsize=11)
        ax.set_xlabel("Judge Likert level  [(s-1)/4]")
        ax.set_ylabel("Empirical accuracy")
        ax.set_xlim(-0.03, 1.03)
        ax.set_ylim(-0.03, 1.03)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, loc="upper left")
        ax.set_aspect("equal")
    fig.suptitle(f"Judge misplacement — {title}   "
                 f"(curve = empirical acc per Likert level; dotted line = score>={THRESH} boundary)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = os.path.join(OUT_DIR, f"calibration_{dkey}.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}")
    return summary


def _pct(v):
    return "n/a" if v != v else f"{v * 100:.1f}%"


def main():
    all_rows = []
    for dkey, suffix, gt_col, title in DOMAINS:
        all_rows += make_dataset_fig(dkey, suffix, gt_col, title)

    # ---- Outputs: a single Misplace rate column everywhere ----
    # Misplace rate = (FNR + FPR) / 2; judge asserts correct iff Likert >= THRESH.
    def mr(r):
        return "n/a" if r["balanced"] != r["balanced"] else f"{r['balanced'] * 100:.1f}%"

    # Single-judge multi-ref benchmarks: one judge, all conditions. Rows join the
    # combined CSV/MD and also get a standalone per-benchmark CSV.
    for dkey, path, gt_col, size, _title in SINGLE_JUDGE:
        try:
            ds = load_dataset(path, split="train")
        except Exception as e:
            print(f"[skip] {path}: {type(e).__name__}: {e}")
            continue
        srows = []
        for cond in CONDITIONS:
            col = f"llm_score_{cond}"
            if col not in ds.column_names:
                continue
            _, m = misplacement(ds[col], ds[gt_col])
            row = dict(domain=dkey, size=size, condition=cond, **m)
            all_rows.append(row)
            srows.append(row)
        p = os.path.join(OUT_DIR, f"misplacement_{dkey}.csv")
        with open(p, "w") as f:
            f.write("LLM Judge,Reference condition,Misplace rate\n")
            for r in srows:
                f.write(f"Qwen3-{size},{r['condition']},{mr(r)}\n")
        print(f"wrote {p}")

    # Combined CSV (all domains)
    csv_path = os.path.join(OUT_DIR, "misplacement_table.csv")
    with open(csv_path, "w") as f:
        f.write("domain,LLM Judge,Reference condition,Misplace rate\n")
        for r in all_rows:
            f.write(f"{r['domain']},Qwen3-{r['size']},{r['condition']},{mr(r)}\n")

    # Combined Markdown: one table per dataset
    md_path = os.path.join(OUT_DIR, "misplacement_table.md")
    hdr = ["LLM Judge", "Reference condition", "Misplace rate"]
    with open(md_path, "w") as f:
        f.write("# Judge misplace rate\n\n")
        f.write(f"Misplace rate = (FNR + FPR) / 2; judge asserts *correct* iff Likert ≥ {THRESH}.\n\n")
        titles = {d[0]: d[3] for d in DOMAINS}
        titles.update({s[0]: s[4] for s in SINGLE_JUDGE})
        for dkey in [d[0] for d in DOMAINS] + [s[0] for s in SINGLE_JUDGE]:
            rows = [r for r in all_rows if r["domain"] == dkey]
            if not rows:
                continue
            f.write(f"### {titles[dkey]}\n\n")
            f.write("| " + " | ".join(hdr) + " |\n")
            f.write("|" + "---|" * len(hdr) + "\n")
            for r in rows:
                f.write(f"| Qwen3-{r['size']} | {r['condition']} | {mr(r)} |\n")
            f.write("\n")
    print(f"wrote {csv_path}\nwrote {md_path}")

    # Per-domain CSVs: LLM Judge, Reference condition, Misplace rate
    for dkey, _, _, _ in DOMAINS:
        rows = [r for r in all_rows if r["domain"] == dkey]
        if not rows:
            continue
        p = os.path.join(OUT_DIR, f"misplacement_{dkey}.csv")
        with open(p, "w") as f:
            f.write("LLM Judge,Reference condition,Misplace rate\n")
            for r in rows:
                f.write(f"Qwen3-{r['size']},{r['condition']},{mr(r)}\n")
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
