"""OlympiadBench open-ended (OE) physics scorer — Gemini Flash Lite LLM judge.

All rows in this dataset are open-ended free-response physics problems whose
reference answer is a numerical value, a symbolic expression, an equation, or a
small set of these (multi-answer). There is no MCQ path: every row is graded by
the LLM judge.

The judge is physics-aware. It receives, in addition to the question and the
reference answer:
  - `unit`  — the expected unit (compared loosely; the model may restate it
              differently or omit it when obvious).
  - `error` — the dataset's per-problem absolute tolerance for NUMERICAL
              answers (e.g. "1e-1", "5e6", "0"). The judge is told to accept a
              numeric answer that falls within this tolerance.
  - `is_multiple_answer` — when true, EVERY part of the reference must be
              matched for credit.

Returns 1.0 / 0.0. On unrecoverable failure the scorer returns 0.0 and logs to
stderr rather than crashing the generation job (matches sciknoweval_judge
behavior). 3 retries with exponential backoff for transient API errors.

Requires:
  - GEMINI_API_KEY env var
  - `google-generativeai` package (pip install google-generativeai)
"""

import os
import re
import sys
import time
from typing import Optional


JUDGE_MODEL = os.environ.get("OLYMPIADBENCH_JUDGE_MODEL", "gemini-3.1-flash-lite")
JUDGE_TIMEOUT_S = float(os.environ.get("OLYMPIADBENCH_JUDGE_TIMEOUT", "60"))
JUDGE_MAX_RETRIES = 3


_JUDGE_TEMPLATE = """You are grading a model's free-form solution to a competition physics problem.

[Question]
{question}

[Reference final answer]
{reference}
{unit_line}{tolerance_line}{multi_line}
[Model response]
{response}

Decide whether the model's FINAL answer matches the reference final answer. Apply physics grading conventions:
- NUMERICAL answers: accept if the model's value agrees with the reference within the stated tolerance (if no tolerance is given, allow ~1% relative). Ignore differences in how the unit is written, and accept correct unit conversions of the same physical quantity. Do not penalize for restating or omitting an obvious unit.
- SYMBOLIC expressions / equations: accept if algebraically equivalent to the reference (equivalent forms, factoring, renamed-but-clearly-corresponding symbols, both sides of an equation rearranged). The model may name the result with a different left-hand-side variable; judge the expression itself.
- MULTI-PART answers: give credit only if EVERY part of the reference is correctly matched.

Only the model's final answer matters, not its derivation. Ignore extra correct working. Do NOT reward an answer that is numerically/algebraically wrong, has the wrong order of magnitude, or is off-topic.

Respond with EXACTLY one token: YES or NO."""


_model = None
_model_init_error = None


def _get_model():
    """Lazy-import + cache the Gemini client, inside the scoring worker so the
    heavy import only happens in processes that actually score."""
    global _model, _model_init_error
    if _model is not None or _model_init_error is not None:
        return _model
    try:
        import google.generativeai as genai
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not set")
        genai.configure(api_key=api_key)
        _model = genai.GenerativeModel(JUDGE_MODEL)
    except Exception as e:
        _model_init_error = e
        print(f"[olympiadbench_oe_judge] init failed: {e}", file=sys.stderr)
    return _model


def _judge_once(prompt: str) -> Optional[float]:
    """Single attempt. Returns None on transient error (caller retries)."""
    model = _get_model()
    if model is None:
        return 0.0  # init failure — don't retry

    try:
        out = model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.0,
                "max_output_tokens": 8,
            },
            request_options={"timeout": JUDGE_TIMEOUT_S},
        )
        text = (out.text or "").strip().upper()
        if re.match(r"^YES\b", text):
            return 1.0
        if re.match(r"^NO\b", text):
            return 0.0
        print(f"[olympiadbench_oe_judge] unparseable judge reply: {text!r}", file=sys.stderr)
        return 0.0
    except Exception as e:
        msg = str(e)
        if any(s in msg for s in ("429", "500", "502", "503", "504", "timeout", "DEADLINE")):
            return None  # transient — let caller retry
        print(f"[olympiadbench_oe_judge] hard error: {e}", file=sys.stderr)
        return 0.0


def _build_prompt(question: str, reference: str, extra_info: dict) -> str:
    unit = (extra_info.get("unit") or "").strip()
    error = (str(extra_info.get("error") or "")).strip()
    is_multi = bool(extra_info.get("is_multiple_answer"))

    unit_line = f"\n[Expected unit] {unit}\n" if unit else ""
    # error == "0" means an exact numeric answer; still informative to the judge.
    tolerance_line = (
        f"[Absolute tolerance for numerical answers] {error}\n" if error else ""
    )
    multi_line = (
        "[Note] This is a multi-part answer; every part must be correct.\n"
        if is_multi else ""
    )
    return _JUDGE_TEMPLATE.format(
        question=question.strip(),
        reference=reference.strip(),
        unit_line=unit_line,
        tolerance_line=tolerance_line,
        multi_line=multi_line,
        response=(extra_info.get("_response") or "").strip(),
    )


def compute_score(solution_str: str, ground_truth: str, extra_info: dict | None = None) -> float:
    """Returns 1.0 if the model response is judged correct, else 0.0.

    `extra_info["question"]` must hold the full problem text (context + question);
    `unit`, `error`, and `is_multiple_answer` are optional physics-grading hints.
    """
    extra_info = extra_info or {}
    question = extra_info.get("question", "") or ""
    if not question:
        print("[olympiadbench_oe_judge] no question in extra_info; returning 0", file=sys.stderr)
        return 0.0

    reference = ground_truth or ""
    # stash response so _build_prompt can read it without a wider signature
    extra_info = {**extra_info, "_response": solution_str or ""}
    prompt = _build_prompt(question, reference, extra_info)

    for attempt in range(JUDGE_MAX_RETRIES):
        score = _judge_once(prompt)
        if score is not None:
            return score
        time.sleep(2 ** attempt)  # 1s, 2s, 4s
    return 0.0
