"""Generate Gemini oracle solutions for a dataset, verify each sample, pick the
best verified one as `extra_info.solution`, write a `<dataset>_with_oracle/`
parquet ready for dense_outcome / dense_pr_delta training.

Currently supports two datasets (selected via --dataset):

  livecodebench_v6   verification = sandboxed code execution against the
                    base64/zlib/pickle/json-encoded test cases in
                    reward_model.ground_truth (matches the encoding in
                    scripts/prepare_livecodebench_gpqa.py:95)

  sciknoweval_oe_v1  verification = Gemini judge call comparing the generated
                    solution's final answer to the reference free-text answer
                    in reward_model.ground_truth

Both stages run concurrently against the Gemini API. State is checkpointed to
<output_dir>/raw_samples.parquet after every batch so the script can resume
after a crash without losing progress.

Final output: data/instruct/<dataset>_with_oracle/train.parquet, identical
schema to the input plus extra_info.solution populated with the picked sample,
extra_info.oracle_n_samples (how many were verified out of N), and
extra_info.oracle_picked_idx.

Usage:
  python -m scripts.build_oracle_dataset --dataset livecodebench_v6 \\
    --n_samples 8 --concurrency 8 --solver_model gemini-3-pro-preview
  python -m scripts.build_oracle_dataset --dataset sciknoweval_oe_v1 \\
    --n_samples 8 --concurrency 8 \\
    --solver_model gemini-3-pro-preview --judge_model gemini-flash-latest
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import pickle
import signal
import sys
import time
import traceback
import zlib
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

# ----- Gemini client (matches QED's verifier_api.py pattern) -----
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "instruct"


# ============================================================================
# Prompt templates
# ============================================================================

LCB_SOLVER_PROMPT = """You are an expert competitive programmer. Solve the following problem.

{problem}

Provide your reasoning, then write the FINAL Python solution inside a single \
```python ... ``` fence. The code must be self-contained and runnable as-is."""

SCIKNOWEVAL_SOLVER_PROMPT = """You are an expert in {domain}. Answer the following \
question with detailed reasoning, then state your final answer.

Question: {question}

Provide step-by-step reasoning, then on a new line write:
FINAL ANSWER: <your concise final answer>"""

SCIKNOWEVAL_JUDGE_PROMPT = """You are an evaluator. Given the question, a reference answer, \
and a candidate answer, decide whether the candidate answer is essentially correct \
(captures the same scientific meaning as the reference, even if worded differently).

Question:
{question}

Reference answer:
{reference}

Candidate answer (final answer extracted from a longer reasoning):
{candidate}

Respond with exactly one word: YES or NO."""


# ============================================================================
# Gemini wrapper
# ============================================================================


def load_aiza_key() -> str:
    """Get GEMINI_API_KEY: prefer the env var (so it picks up the exported key),
    else fall back to parsing ~/.bashrc and skipping commented lines."""
    env = os.environ.get("GEMINI_API_KEY")
    if env and env.startswith("AIzaSy"):
        return env
    bashrc = Path.home() / ".bashrc"
    for line in bashrc.read_text().splitlines():
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        if "GEMINI_API_KEY=AIzaSy" in stripped:
            return stripped.split("=", 1)[1].strip().strip('"').strip("'")
    raise RuntimeError("No active GEMINI_API_KEY found (env var or ~/.bashrc)")


_SEM = None
_CLIENT = None


def _client(api_key: str) -> "genai.Client":
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = genai.Client(api_key=api_key)
    return _CLIENT


def _build_config(thinking_level: str | None):
    """Construct a GenerateContentConfig with thinking_level set, or return None."""
    if not thinking_level:
        return None
    try:
        level_enum = genai_types.ThinkingLevel(thinking_level.upper())
    except ValueError:
        raise ValueError(f"Invalid thinking_level: {thinking_level!r}. "
                         f"Choose from MINIMAL, LOW, MEDIUM, HIGH.")
    return genai_types.GenerateContentConfig(
        thinking_config=genai_types.ThinkingConfig(thinking_level=level_enum)
    )


async def gemini_call(api_key: str, model: str, prompt: str,
                     thinking_level: str | None = None, max_retries: int = 4) -> str:
    """One Gemini call with exponential backoff for 429/transient errors."""
    client = _client(api_key)
    config = _build_config(thinking_level)
    delay = 2.0
    last_err = None
    for attempt in range(max_retries):
        try:
            kwargs = {"model": model, "contents": prompt}
            if config is not None:
                kwargs["config"] = config
            r = await client.aio.models.generate_content(**kwargs)
            return (r.text or "").strip()
        except genai_errors.ClientError as e:
            last_err = e
            code = getattr(e, "code", None)
            if code in (429, 503):
                await asyncio.sleep(delay)
                delay *= 2.0
                continue
            raise
        except Exception as e:
            last_err = e
            await asyncio.sleep(delay)
            delay *= 2.0
    raise last_err  # type: ignore


async def gemini_call_throttled(api_key, model, prompt, thinking_level=None) -> str:
    assert _SEM is not None
    async with _SEM:
        return await gemini_call(api_key, model, prompt, thinking_level=thinking_level)


# ============================================================================
# Per-dataset adapters
# ============================================================================


def build_solver_prompt(dataset: str, row: pd.Series) -> str:
    if dataset == "livecodebench_v6":
        # LCB row.prompt is [{role: user, content: <problem incl. instructions>}]
        problem = row["prompt"][0]["content"]
        return LCB_SOLVER_PROMPT.format(problem=problem)
    elif dataset == "sciknoweval_oe_v1":
        # SciKnowEval row.prompt may include a system msg; question lives in extra_info
        ei = row["extra_info"]
        question = ei.get("question") or row["prompt"][-1]["content"]
        domain = ei.get("domain") or "science"
        return SCIKNOWEVAL_SOLVER_PROMPT.format(domain=domain, question=question)
    elif dataset.startswith("sciknoweval_mcq"):
        # The parquet's prompt already has the SYS_PROMPT (asking for reasoning + a
        # final-line letter answer) + the user's question with lettered choices.
        # Just concatenate system + user so the Gemini call sees the same prompt
        # the trainer would.
        parts = []
        for msg in row["prompt"]:
            parts.append(msg["content"])
        return "\n\n".join(parts)
    raise ValueError(f"Unknown dataset: {dataset}")


def extract_python_code(text: str) -> str | None:
    """Extract the LAST ```python ... ``` block from a Gemini response."""
    import re
    matches = re.findall(r"```python\s*\n(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if not matches:
        matches = re.findall(r"```\s*\n(.*?)```", text, re.DOTALL)
    return matches[-1].strip() if matches else None


def extract_mcq_letter(text: str) -> str | None:
    """Extract the chosen MCQ letter (A/B/C/D) from a model response.

    Looks for, in order: "Final Answer: X", "Answer: X", "**X**" on the last line,
    "(X)" pattern, or a lone letter on the last line.
    """
    import re
    if not text:
        return None
    # Strict format we asked for first
    m = re.search(r"Final\s*Answer\s*:?\s*[*]*\s*([A-D])\b", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # Common fallback: "Answer: X"
    m = re.search(r"\bAnswer\s*:?\s*[*]*\s*([A-D])\b", text, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    # Markdown-bolded letter
    m = re.search(r"\*\*\s*([A-D])\s*\*\*", text)
    if m:
        return m.group(1).upper()
    # "(X)" pattern at end of text
    tail = text.strip().splitlines()[-1] if text.strip() else ""
    m = re.search(r"\(([A-D])\)", tail)
    if m:
        return m.group(1).upper()
    # Lone letter on the last non-empty line
    if re.fullmatch(r"\s*([A-D])[\.\)]?\s*", tail):
        return re.search(r"([A-D])", tail).group(1).upper()
    return None


def verify_mcq_sample(response_text: str, ground_truth: str) -> dict:
    """Letter-extract + string-compare. No API call."""
    if not response_text:
        return {"ok": False, "predicted": None, "reason": "empty_response"}
    if not ground_truth:
        return {"ok": False, "predicted": None, "reason": "empty_ground_truth"}
    pred = extract_mcq_letter(response_text)
    gt = ground_truth.strip().upper()
    if pred is None:
        return {"ok": False, "predicted": None, "reason": "no_letter_extracted"}
    return {
        "ok": pred == gt,
        "predicted": pred,
        "pass_rate": 1.0 if pred == gt else 0.0,
        "reason": "ok",
    }


def extract_final_answer(text: str) -> str:
    """Grab the FINAL ANSWER: line, or fall back to last non-empty line."""
    import re
    m = re.search(r"FINAL\s*ANSWER\s*:\s*(.+?)(?:\n|$)", text, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


# ============================================================================
# LCB verifier: sandboxed code execution
# ============================================================================


def _decode_lcb_ground_truth(packed: str) -> dict:
    """Reverse of prepare_livecodebench_gpqa.py:95."""
    decoded = pickle.loads(zlib.decompress(base64.b64decode(packed.encode("utf-8"))))
    if isinstance(decoded, str):
        decoded = json.loads(decoded)
    return decoded  # {"inputs": [...], "outputs": [...], "fn_name": str|None}


def verify_lcb_sample(code: str | None, gt: dict, time_per_test: float = 6.0) -> dict:
    """Reuses the same `check_correctness` driver as the trainer's reward function
    (recipe.r1.tasks.livecodebench), so it handles both stdin/stdout problems
    (`fn_name=None`) and LeetCode-style class-method problems (`fn_name=...`)."""
    if not code:
        return {"passed": 0, "total": 1, "pass_rate": 0.0, "ok": False, "reason": "no_code"}
    inputs = gt.get("inputs") or []
    n = len(inputs)
    if n == 0:
        return {"passed": 0, "total": 0, "pass_rate": 0.0, "ok": False, "reason": "no_tests"}

    try:
        from recipe.r1.tasks.livecodebench import check_correctness
        res, _meta = check_correctness(in_outs=gt, generation=code,
                                       timeout=int(time_per_test), debug=False)
    except Exception as e:
        return {"passed": 0, "total": n, "pass_rate": 0.0, "ok": False,
                "reason": f"runner_err:{type(e).__name__}"}

    passed = sum(1 for r in res if r is True)
    rate = passed / n if n else 0.0
    return {"passed": passed, "total": n, "pass_rate": rate, "ok": rate == 1.0, "reason": "ok"}


# ============================================================================
# sciknoweval verifier: Gemini judge
# ============================================================================


async def verify_sciknoweval_sample(api_key, judge_model, question, reference, candidate_text) -> dict:
    candidate = extract_final_answer(candidate_text)
    if not candidate:
        return {"ok": False, "judge": "?", "reason": "no_answer"}
    prompt = SCIKNOWEVAL_JUDGE_PROMPT.format(
        question=question, reference=reference, candidate=candidate
    )
    try:
        verdict = (await gemini_call_throttled(api_key, judge_model, prompt)).strip().upper()
    except Exception as e:
        return {"ok": False, "judge": "?", "reason": f"judge_err:{type(e).__name__}"}
    is_yes = verdict.startswith("YES")
    return {"ok": is_yes, "judge": verdict[:20], "reason": "ok"}


# ============================================================================
# Per-row generation + verification orchestration
# ============================================================================


async def process_row_lcb(api_key, solver_model, judge_model, row, n_samples,
                          solver_thinking_level=None) -> list[dict]:
    """Generate n_samples solutions, run code verifier in a thread pool."""
    del judge_model  # LCB uses sandboxed execution, no LLM judge
    prompt = build_solver_prompt("livecodebench_v6", row)
    gt = _decode_lcb_ground_truth(row["reward_model"]["ground_truth"])

    # Fan out the Gemini calls.
    raw_resps = await asyncio.gather(
        *[gemini_call_throttled(api_key, solver_model, prompt,
                                thinking_level=solver_thinking_level)
          for _ in range(n_samples)],
        return_exceptions=True,
    )

    out = []
    loop = asyncio.get_event_loop()
    for i, resp in enumerate(raw_resps):
        if isinstance(resp, Exception):
            out.append({"sample_idx": i, "raw": "", "solution": None,
                        "verify": {"ok": False, "reason": f"gen_err:{type(resp).__name__}"}})
            continue
        code = extract_python_code(resp)
        verify = await loop.run_in_executor(None, verify_lcb_sample, code, gt)
        out.append({"sample_idx": i, "raw": resp, "solution": code, "verify": verify})
    return out


async def process_row_mcq(api_key, solver_model, judge_model, row, n_samples,
                          solver_thinking_level=None) -> list[dict]:
    """Process a SciKnowEval MCQ row: generate n_samples responses, letter-verify locally."""
    del judge_model  # MCQ uses local string-match, no LLM judge
    # All sciknoweval_mcq_* presets use the same prompt structure, so any name works.
    prompt = build_solver_prompt("sciknoweval_mcq_l3_loose", row)
    ground_truth = (row["reward_model"] or {}).get("ground_truth") or ""

    raw_resps = await asyncio.gather(
        *[gemini_call_throttled(api_key, solver_model, prompt,
                                thinking_level=solver_thinking_level)
          for _ in range(n_samples)],
        return_exceptions=True,
    )
    out = []
    for i, resp in enumerate(raw_resps):
        if isinstance(resp, Exception):
            out.append({"sample_idx": i, "raw": "", "solution": None,
                        "verify": {"ok": False, "reason": f"gen_err:{type(resp).__name__}"}})
            continue
        verify = verify_mcq_sample(resp, ground_truth)
        out.append({"sample_idx": i, "raw": resp, "solution": resp, "verify": verify})
    return out


async def process_row_sciknoweval(api_key, solver_model, judge_model, row, n_samples,
                                  solver_thinking_level=None) -> list[dict]:
    prompt = build_solver_prompt("sciknoweval_oe_v1", row)
    reference = row["reward_model"]["ground_truth"]
    question = row["extra_info"].get("question") or row["prompt"][-1]["content"]

    raw_resps = await asyncio.gather(
        *[gemini_call_throttled(api_key, solver_model, prompt,
                                thinking_level=solver_thinking_level)
          for _ in range(n_samples)],
        return_exceptions=True,
    )

    out = []
    verify_tasks = []
    for i, resp in enumerate(raw_resps):
        if isinstance(resp, Exception):
            out.append({"sample_idx": i, "raw": "", "solution": None,
                        "verify": {"ok": False, "reason": f"gen_err:{type(resp).__name__}"}})
            verify_tasks.append(None)
            continue
        out.append({"sample_idx": i, "raw": resp, "solution": resp, "verify": None})
        verify_tasks.append(verify_sciknoweval_sample(api_key, judge_model, question, reference, resp))

    # Run judge calls concurrently
    verdicts = await asyncio.gather(*[t for t in verify_tasks if t is not None],
                                    return_exceptions=True)
    vi = 0
    for entry, t in zip(out, verify_tasks):
        if t is None:
            continue
        v = verdicts[vi]; vi += 1
        if isinstance(v, Exception):
            v = {"ok": False, "judge": "?", "reason": f"judge_err:{type(v).__name__}"}
        entry["verify"] = v
    return out


# ============================================================================
# Top-level orchestration
# ============================================================================


@dataclass
class JobConfig:
    dataset: str
    n_samples: int
    concurrency: int
    solver_model: str
    judge_model: str
    solver_thinking_level: str | None
    max_rows: int | None
    input_path: Path
    output_dir: Path
    batch_size: int = 25
    resume: bool = True


def load_input(cfg: JobConfig) -> pd.DataFrame:
    df = pd.read_parquet(cfg.input_path)
    if cfg.max_rows:
        df = df.head(cfg.max_rows).copy()
    df = df.reset_index(drop=True)
    df["_row_idx"] = df.index
    return df


def load_checkpoint(cfg: JobConfig) -> set[int]:
    p = cfg.output_dir / "raw_samples.jsonl"
    if not (cfg.resume and p.exists()):
        return set()
    done = set()
    with p.open() as f:
        for line in f:
            try:
                done.add(json.loads(line)["row_idx"])
            except Exception:
                pass
    return done


def append_checkpoint(cfg: JobConfig, row_idx: int, samples: list[dict]):
    p = cfg.output_dir / "raw_samples.jsonl"
    with p.open("a") as f:
        f.write(json.dumps({"row_idx": row_idx, "samples": samples}, ensure_ascii=False) + "\n")


PROCESSORS = {
    "livecodebench_v6": process_row_lcb,
    "sciknoweval_oe_v1": process_row_sciknoweval,
    "sciknoweval_mcq_l3_loose": process_row_mcq,
    "sciknoweval_mcq_l3_tight": process_row_mcq,
}


async def run(cfg: JobConfig):
    global _SEM
    _SEM = asyncio.Semaphore(cfg.concurrency)
    api_key = load_aiza_key()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    df = load_input(cfg)
    done = load_checkpoint(cfg)
    todo = [i for i in df["_row_idx"] if i not in done]

    print(f"[oracle] dataset={cfg.dataset} solver={cfg.solver_model} judge={cfg.judge_model} n_samples={cfg.n_samples}")
    print(f"[oracle] solver_thinking_level={cfg.solver_thinking_level}")
    print(f"[oracle] input={cfg.input_path} ({len(df)} rows total)")
    print(f"[oracle] output={cfg.output_dir}")
    print(f"[oracle] resume: {len(done)} already done, {len(todo)} remaining")
    print(f"[oracle] concurrency={cfg.concurrency}, batch_size={cfg.batch_size}")

    proc = PROCESSORS[cfg.dataset]
    t0 = time.time()

    for batch_start in range(0, len(todo), cfg.batch_size):
        batch = todo[batch_start: batch_start + cfg.batch_size]
        t = time.time()

        async def one(idx):
            row = df.iloc[idx]
            try:
                samples = await proc(api_key, cfg.solver_model, cfg.judge_model, row,
                                     cfg.n_samples,
                                     solver_thinking_level=cfg.solver_thinking_level)
            except Exception as e:
                samples = [{
                    "sample_idx": 0, "raw": "", "solution": None,
                    "verify": {"ok": False, "reason": f"row_err:{type(e).__name__}:{str(e)[:120]}"},
                }]
            return idx, samples

        results = await asyncio.gather(*[one(i) for i in batch])
        for idx, samples in results:
            append_checkpoint(cfg, idx, samples)

        n_ok = sum(1 for _, samples in results
                   for s in samples if s.get("verify", {}).get("ok"))
        n_tot = sum(len(samples) for _, samples in results)
        elapsed = time.time() - t
        rate = (batch_start + len(batch)) / max(time.time() - t0, 0.01)
        eta = (len(todo) - batch_start - len(batch)) / max(rate, 0.001)
        print(f"[oracle] batch {batch_start//cfg.batch_size + 1} "
              f"({len(batch)} rows, {elapsed:.1f}s): "
              f"{n_ok}/{n_tot} samples verified  |  "
              f"throughput {rate*60:.1f} rows/min, ETA {eta/60:.1f} min", flush=True)

    print(f"[oracle] generation+verify done in {(time.time()-t0)/60:.1f} min")
    finalize_dataset(cfg, df)


def finalize_dataset(cfg: JobConfig, df: pd.DataFrame):
    """Read raw_samples.jsonl and write the final parquet.
    Keep every row that has any generated content (verified preferred, else best by pass_rate)."""
    p = cfg.output_dir / "raw_samples.jsonl"
    by_row = {}
    with p.open() as f:
        for line in f:
            d = json.loads(line)
            by_row[d["row_idx"]] = d["samples"]

    rows_out = []
    n_verified = n_unverified = n_no_content = 0
    for _, row in df.iterrows():
        samples = by_row.get(row["_row_idx"], [])
        with_content = [s for s in samples if s.get("raw") or s.get("solution")]
        if not with_content:
            n_no_content += 1
            continue
        verified = [s for s in with_content if s.get("verify", {}).get("ok")]
        if verified:
            verified.sort(key=lambda s: -s.get("verify", {}).get("pass_rate", 0.0))
            picked = verified[0]
            is_verified = True
            n_verified += 1
        else:
            with_content.sort(key=lambda s: -s.get("verify", {}).get("pass_rate", 0.0))
            picked = with_content[0]
            is_verified = False
            n_unverified += 1

        new_row = row.drop("_row_idx").to_dict()
        # solution = raw response (reasoning + code/answer) — used as Reference Solution by GenRM
        # solution_code_only = extracted code for LCB (empty for non-code datasets)
        raw_resp = picked.get("raw") or picked.get("solution") or ""
        if cfg.dataset == "livecodebench_v6":
            code_only = picked.get("solution") or extract_python_code(raw_resp) or ""
        else:
            code_only = ""

        ei = dict(new_row.get("extra_info", {}) or {})
        ei["solution"] = raw_resp
        ei["solution_code_only"] = code_only
        ei["oracle_verified"] = is_verified
        ei["oracle_pass_rate"] = picked.get("verify", {}).get("pass_rate", 0.0)
        ei["oracle_n_verified"] = sum(1 for s in samples if s.get("verify", {}).get("ok"))
        ei["oracle_n_total"] = len(samples)
        ei["oracle_picked_idx"] = picked["sample_idx"]
        ei["oracle_solver_model"] = cfg.solver_model
        ei["oracle_judge_model"] = cfg.judge_model
        if "question" not in ei:
            if cfg.dataset == "livecodebench_v6":
                ei["question"] = new_row["prompt"][0]["content"]
            else:
                ei["question"] = new_row["prompt"][-1]["content"]
        new_row["extra_info"] = ei

        # For LCB, the 4MB-per-row test cases in reward_model.ground_truth blow up
        # the parquet to ~4GB and crash pyarrow's write. Strip them.
        if cfg.dataset == "livecodebench_v6":
            rm = dict(new_row.get("reward_model", {}) or {})
            rm["ground_truth"] = ""
            rm.setdefault("style", "rule")
            new_row["reward_model"] = rm

        rows_out.append(new_row)

    out_df = pd.DataFrame(rows_out)
    out_path = cfg.output_dir / "train.parquet"
    out_df.to_parquet(out_path, index=False)

    print(f"[oracle] kept {n_verified + n_unverified} rows "
          f"(verified={n_verified}, unverified={n_unverified}, no_content_dropped={n_no_content})")
    print(f"[oracle] wrote {out_path} ({len(out_df)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(PROCESSORS))
    ap.add_argument("--n_samples", type=int, default=8)
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--solver_model", default="gemini-3-pro-preview",
                    help="Model used to generate candidate solutions.")
    ap.add_argument("--judge_model", default="gemini-flash-latest",
                    help="Model used as the SciKnowEval YES/NO judge (ignored for LCB).")
    ap.add_argument("--solver_thinking_level", default="MEDIUM",
                    choices=["MINIMAL", "LOW", "MEDIUM", "HIGH", "NONE"],
                    help="Gemini 3 thinking level for the solver. NONE = use API default.")
    ap.add_argument("--max_rows", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=25)
    ap.add_argument("--input_path", type=str, default=None)
    ap.add_argument("--output_dir", type=str, default=None)
    ap.add_argument("--no_resume", action="store_true")
    args = ap.parse_args()

    in_path = Path(args.input_path) if args.input_path else (DATA_DIR / args.dataset / ("test.parquet" if args.dataset != "int_dataset" else "train.parquet"))
    out_dir = Path(args.output_dir) if args.output_dir else (DATA_DIR / f"{args.dataset}_with_oracle")

    cfg = JobConfig(
        dataset=args.dataset,
        n_samples=args.n_samples,
        concurrency=args.concurrency,
        solver_model=args.solver_model,
        judge_model=args.judge_model,
        solver_thinking_level=None if args.solver_thinking_level == "NONE" else args.solver_thinking_level,
        max_rows=args.max_rows,
        input_path=in_path,
        output_dir=out_dir,
        batch_size=args.batch_size,
        resume=not args.no_resume,
    )
    asyncio.run(run(cfg))


if __name__ == "__main__":
    main()
