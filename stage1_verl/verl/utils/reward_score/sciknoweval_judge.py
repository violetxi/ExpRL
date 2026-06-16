"""SciKnowEval scorer. Routes per row based on `extra_info["type"]`:

  - MCQ rows (`mcq-4-choices`, `mcq-2-choices`): deterministic letter extraction
    from the model's final answer, compared to `reward_model.ground_truth`
    (a single letter). No API call.
  - Open-ended rows (`open-ended-qa`): Gemini Flash Lite LLM judge compares the
    free-form response to the reference answer. 3 retries with exponential
    backoff for transient API errors.

Both paths return 1.0 / 0.0. On unrecoverable failure the scorer returns 0.0
and logs to stderr — a hard fail here would crash the whole generation job,
which is worse than under-scoring one row.

LLM judge requires:
  - GEMINI_API_KEY env var
  - `google-generativeai` package (pip install google-generativeai)
"""

import os
import re
import sys
import time
from typing import Optional


JUDGE_MODEL = os.environ.get("SCIKNOWEVAL_JUDGE_MODEL", "gemini-3.1-flash-lite")
JUDGE_TIMEOUT_S = float(os.environ.get("SCIKNOWEVAL_JUDGE_TIMEOUT", "60"))
JUDGE_MAX_RETRIES = 3

# extra_info["type"] values that trigger the deterministic MCQ path
MCQ_TYPES = {"mcq-4-choices", "mcq-2-choices"}

_JUDGE_TEMPLATE = """You are evaluating a model's free-form response to an open-ended scientific question.

[Question]
{question}

[Reference answer]
{reference}

[Model response]
{response}

Does the model response correctly answer the question, consistent with the reference answer? Allow for:
- Paraphrasing and alternative phrasings.
- Additional correct detail not present in the reference.
- Minor omissions, as long as the core scientific content is right.

Do NOT reward responses that are off-topic, factually wrong, or contradict the reference's core claims.

Respond with EXACTLY one token: YES or NO."""


_model = None
_model_init_error = None


def _get_model():
    """Lazy-import + cache the Gemini client. Called inside the scoring worker
    so the heavy import only happens in processes that actually score."""
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
        print(f"[sciknoweval_judge] init failed: {e}", file=sys.stderr)
    return _model


def _judge_once(question: str, reference: str, response: str) -> Optional[float]:
    """Single attempt. Returns None on transient error (caller retries)."""
    model = _get_model()
    if model is None:
        return 0.0  # init failure — don't retry

    prompt = _JUDGE_TEMPLATE.format(
        question=question.strip(),
        reference=reference.strip(),
        response=response.strip(),
    )
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
        # Unexpected output — treat as a hard-no with a warning.
        print(f"[sciknoweval_judge] unparseable judge reply: {text!r}", file=sys.stderr)
        return 0.0
    except Exception as e:
        msg = str(e)
        if any(s in msg for s in ("429", "500", "502", "503", "504", "timeout", "DEADLINE")):
            return None  # transient — let caller retry
        print(f"[sciknoweval_judge] hard error: {e}", file=sys.stderr)
        return 0.0


def extract_mcq_letter(text: str) -> Optional[str]:
    """Extract a chosen MCQ letter (A/B/C/D) from a model response.

    Defensive against Qwen3/Qwen-8B no-thinking outputs that may emit "Final Answer:"
    multiple times (e.g., once mid-reasoning, once at the end). At each layer we
    take the LAST match so a re-stated final answer wins over a tentative earlier one.

    Priority order:
      1. `Final Answer: X`  (what the system prompt explicitly asks for)
      2. `Answer: X`        (common natural-language fallback)
      3. `**X**`            (markdown-bolded letter)
      4. `(X)`              (parenthesized letter on the last non-empty line)
      5. Lone letter        (single A/B/C/D on the last non-empty line)
    """
    if not text:
        return None

    # 1. "Final Answer: X" — preferred, take LAST occurrence
    matches = re.findall(r"Final\s*Answer\s*:?\s*[*]*\s*([A-D])\b", text, re.IGNORECASE)
    if matches:
        return matches[-1].upper()

    # 2. "Answer: X" — take LAST
    matches = re.findall(r"\bAnswer\s*:?\s*[*]*\s*([A-D])\b", text, re.IGNORECASE)
    if matches:
        return matches[-1].upper()

    # 3. "**X**" markdown-bolded letter — take LAST
    matches = re.findall(r"\*\*\s*([A-D])\s*\*\*", text, re.IGNORECASE)
    if matches:
        return matches[-1].upper()

    # 4/5. Last non-empty line — "(X)" or lone letter
    tail_lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if tail_lines:
        last = tail_lines[-1]
        m = re.search(r"\(([A-D])\)", last, re.IGNORECASE)
        if m:
            return m.group(1).upper()
        m = re.fullmatch(r"[*]*\s*([A-D])\s*[\.\)\*]*\s*", last, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    return None


def _score_mcq(solution_str: str, ground_truth: str) -> float:
    """Letter-extract + exact match. No API call. 1.0 on match, 0.0 otherwise."""
    if not ground_truth:
        print("[sciknoweval_judge] MCQ: empty ground_truth; returning 0", file=sys.stderr)
        return 0.0
    gt = ground_truth.strip().upper()
    pred = extract_mcq_letter(solution_str or "")
    if pred is None:
        # Couldn't find any A/B/C/D in the response — count as wrong.
        return 0.0
    return 1.0 if pred == gt else 0.0


def compute_score(solution_str: str, ground_truth: str, extra_info: dict | None = None) -> float:
    """Returns 1.0 if the response is judged correct, else 0.0.

    Routing (driven by `extra_info["type"]`):
      - `mcq-4-choices` / `mcq-2-choices`  -> deterministic letter match (free, fast)
      - everything else (incl. `open-ended-qa`)  -> Gemini Flash Lite LLM judge

    The question text must be available in `extra_info["question"]` for the LLM
    judge path (added by `scripts/prepare_sciknoweval.py`).
    """
    ei_type = (extra_info or {}).get("type", "") or ""
    if ei_type in MCQ_TYPES:
        return _score_mcq(solution_str, ground_truth)

    question = ""
    if extra_info is not None:
        question = extra_info.get("question", "") or ""
    if not question:
        # Without the question, the judge has no context — fall back to 0.
        print("[sciknoweval_judge] no question in extra_info; returning 0", file=sys.stderr)
        return 0.0

    reference = ground_truth or ""
    response = solution_str or ""

    for attempt in range(JUDGE_MAX_RETRIES):
        score = _judge_once(question, reference, response)
        if score is not None:
            return score
        # exponential backoff: 1s, 2s, 4s
        time.sleep(2 ** attempt)
    return 0.0
