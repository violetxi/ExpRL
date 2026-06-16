from __future__ import annotations

import atexit
import asyncio
import contextlib
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import httpx
import openai
import torch


logger = logging.getLogger(__name__)

_PROMPT_DIR = Path(__file__).resolve().parents[2] / "utils" / "prompt_templates" / "proof_evaluator_prompts"
_PROCESS_WHY_PATTERN = re.compile(r"^Why:\s*(.+)$")
_PROCESS_ALIGNED_PATH_PATTERN = re.compile(r"^Aligned path:\s*(.+?)\s*$")
_PROCESS_SCORE_PATTERN = re.compile(r"^Score:\s*([1-5])\s*$")
_GRADER_CLIENTS: dict[str, Any] = {}
DEFAULT_STEP_DELIMITER = "### "
_TRAILING_CHAT_TOKENS = ("<|im_end|>", "</s>", "<|endoftext|>")


@dataclass
class GraderResponse:
    output_text: str
    reasoning_text: str
    input_tokens: int | None = None
    output_tokens: int | None = None


@dataclass
class ProofVerificationResult:
    score: int
    output_text: str = ""
    reasoning_text: str = ""
    input_tokens: int | None = None
    output_tokens: int | None = None
    failure_cause: str | None = None


@dataclass
class ProcessJudgeResult:
    prefix_index: int
    score: int
    aligned_path: str | None = None
    why: str | None = None
    output_text: str = ""
    reasoning_text: str = ""
    parse_failed: bool = False
    failure_cause: str | None = None


@dataclass(frozen=True)
class RewardChunk:
    index: int
    token_span: tuple[int, int]


def strip_trailing_chat_tokens(text: str) -> str:
    stripped = text
    for token in _TRAILING_CHAT_TOKENS:
        while stripped.endswith(token):
            stripped = stripped[: -len(token)].rstrip()
    return stripped


def load_evaluator_prompt(prompt_name: str | os.PathLike | None) -> str:
    if prompt_name is None:
        raise ValueError("prompt_name must be set for proof grading")

    prompt_str = str(prompt_name)
    prompt_path = Path(prompt_str)
    if prompt_path.is_file():
        return prompt_path.read_text(encoding="utf-8")

    filename = Path(prompt_str).name
    if not filename.endswith(".md"):
        filename = f"{filename}.md"

    prompt_path = (_PROMPT_DIR / filename).resolve()
    if not prompt_path.is_file():
        raise FileNotFoundError(f"Evaluator prompt '{filename}' not found in {_PROMPT_DIR}")
    return prompt_path.read_text(encoding="utf-8")


def parse_schema(schema: Any) -> str:
    if isinstance(schema, str):
        return schema
    if not isinstance(schema, list):
        raise TypeError("Schema must be a string or a list of dicts")

    sections: list[str] = []
    for idx, entry in enumerate(schema):
        if not isinstance(entry, dict):
            raise ValueError(f"Schema entry at index {idx} must be a dict")
        title = entry.get("title")
        points = entry.get("points")
        description = entry.get("desc") or entry.get("description")
        if title is None or points is None or description is None:
            raise ValueError(f"Schema entry at index {idx} missing title, points, or description")
        sections.append(f"# {title} ({points} points)\nDescription: {description}".strip())
    return "\n\n".join(sections)


def resolve_provider(model: str | None, explicit: str | None = None) -> str:
    if explicit and explicit != "auto":
        return explicit
    if not model:
        return "openai"
    name = model.lower()
    if name.startswith("google/") or "gemini" in name:
        return "gemini"
    return "openai"


def _strip_google_prefix(model: str) -> str:
    return model.split("/", 1)[1] if model.lower().startswith("google/") else model


def _extract_reasoning_from_response(response: Any) -> str:
    reasoning_chunks: list[str] = []
    for item in getattr(response, "output", None) or []:
        if getattr(item, "type", None) == "reasoning":
            for content_item in getattr(item, "content", []) or []:
                text = getattr(content_item, "text", None)
                if text:
                    reasoning_chunks.append(text)
    return "\n\n".join(reasoning_chunks)


def _normalize_openai_response(response: Any) -> GraderResponse:
    output_text = getattr(response, "output_text", None) or ""
    reasoning_text = _extract_reasoning_from_response(response)
    usage = getattr(response, "usage", None)
    input_tokens = None
    output_tokens = None
    if usage is not None:
        input_tokens = getattr(usage, "input_tokens", None)
        output_tokens = getattr(usage, "output_tokens", None)
        if input_tokens is None and isinstance(usage, dict):
            input_tokens = usage.get("input_tokens")
        if output_tokens is None and isinstance(usage, dict):
            output_tokens = usage.get("output_tokens")
    return GraderResponse(
        output_text=output_text,
        reasoning_text=reasoning_text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def _build_gemini_config(api_kwargs: dict[str, Any] | None):
    from google.genai import types

    remaining = dict(api_kwargs or {})
    thinking_kwargs: dict[str, Any] = {"include_thoughts": True}
    if "thinking_level" in remaining:
        thinking_kwargs["thinking_level"] = remaining.pop("thinking_level")
    if "thinking_budget" in remaining:
        thinking_kwargs["thinking_budget"] = remaining.pop("thinking_budget")
    if "include_thoughts" in remaining:
        thinking_kwargs["include_thoughts"] = bool(remaining.pop("include_thoughts"))

    config_kwargs: dict[str, Any] = {
        "thinking_config": types.ThinkingConfig(**thinking_kwargs),
    }
    passthrough = {
        "temperature",
        "top_p",
        "top_k",
        "max_output_tokens",
        "stop_sequences",
        "candidate_count",
        "response_mime_type",
    }
    for key in list(remaining.keys()):
        if key in passthrough:
            config_kwargs[key] = remaining.pop(key)

    if remaining:
        logger.warning("Ignoring unsupported Gemini sampling_kwargs: %s", sorted(remaining.keys()))
    return types.GenerateContentConfig(**config_kwargs)


def _normalize_gemini_response(response: Any) -> GraderResponse:
    reasoning_chunks: list[str] = []
    output_chunks: list[str] = []
    candidates = getattr(response, "candidates", None) or []
    if candidates:
        content = getattr(candidates[0], "content", None)
        parts = getattr(content, "parts", None) or []
        for part in parts:
            text = getattr(part, "text", None)
            if not text:
                continue
            if getattr(part, "thought", False):
                reasoning_chunks.append(text)
            else:
                output_chunks.append(text)
    output_text = "\n".join(output_chunks).strip()
    if not output_text:
        output_text = (getattr(response, "text", None) or "").strip()
    reasoning_text = "\n\n".join(reasoning_chunks).strip()

    usage = getattr(response, "usage_metadata", None)
    input_tokens = None
    output_tokens = None
    if usage is not None:
        input_tokens = getattr(usage, "prompt_token_count", None)
        candidates_tokens = getattr(usage, "candidates_token_count", None)
        thought_tokens = getattr(usage, "thoughts_token_count", None)
        if candidates_tokens is not None or thought_tokens is not None:
            output_tokens = (candidates_tokens or 0) + (thought_tokens or 0)
    return GraderResponse(
        output_text=output_text,
        reasoning_text=reasoning_text,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )


def _resolve_gemini_async_client(client: Any):
    return getattr(client, "aio", client)


def _build_gemini_client(api_key: str):
    from google import genai
    from google.genai import types

    # Force the async path onto httpx. The default aiohttp connector in the
    # installed google-genai build emits an ignored destructor traceback during
    # interpreter shutdown when clients are left to GC.
    http_options = types.HttpOptions(
        async_client_args={"transport": httpx.AsyncHTTPTransport()}
    )
    return genai.Client(api_key=api_key, http_options=http_options)


async def _aclose_grader_clients(clients: Sequence[Any]) -> None:
    coroutines = []
    for client in clients:
        async_client = getattr(client, "aio", None)
        aclose = getattr(async_client, "aclose", None)
        if callable(aclose):
            coroutines.append(aclose())

    if coroutines:
        await asyncio.gather(*coroutines, return_exceptions=True)


def close_grader_clients() -> None:
    clients = list(_GRADER_CLIENTS.values())
    _GRADER_CLIENTS.clear()
    if not clients:
        return

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        loop.create_task(_aclose_grader_clients(clients))
    else:
        with contextlib.suppress(Exception):
            asyncio.run(_aclose_grader_clients(clients))

    for client in clients:
        close = getattr(client, "close", None)
        if callable(close):
            with contextlib.suppress(Exception):
                close()


atexit.register(close_grader_clients)


def get_grader_client(provider: str):
    if provider in _GRADER_CLIENTS:
        return _GRADER_CLIENTS[provider]

    if provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("OPENAI_BASE_URL")
        if not api_key or not base_url:
            raise RuntimeError("Missing OPENAI_API_KEY or OPENAI_BASE_URL environment variable")
        client = openai.OpenAI(api_key=api_key, base_url=base_url)
    elif provider == "gemini":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing GEMINI_API_KEY environment variable for Gemini grader")
        client = _build_gemini_client(api_key)
    else:
        raise RuntimeError(f"Unknown grader provider: {provider}")

    _GRADER_CLIENTS[provider] = client
    return client


def _classify_gemini_exception(exc: Exception) -> str:
    try:
        from google.genai import errors as genai_errors  # type: ignore
    except Exception:
        genai_errors = None  # type: ignore

    if genai_errors is not None and isinstance(exc, getattr(genai_errors, "APIError", tuple())):
        status = getattr(exc, "code", None) or getattr(exc, "status_code", None)
        if status in (429,):
            return "rate_limit"
        if status in (408, 504):
            return "timeout"
    msg = str(exc).lower()
    if "rate limit" in msg or "quota" in msg or " 429" in msg or "resource_exhausted" in msg:
        return "rate_limit"
    if "timeout" in msg or "timed out" in msg or "deadline" in msg:
        return "timeout"
    return "other"


def format_variants_block(variants: Sequence[str] | None) -> str:
    cleaned_variants = [
        variant.strip()
        for variant in (variants or [])
        if isinstance(variant, str) and variant.strip()
    ]
    if not cleaned_variants:
        return "(no variants provided)"
    return "\n\n".join(
        f"--- Variant {idx + 1} ---\n{variant}"
        for idx, variant in enumerate(cleaned_variants)
    )


def coerce_variants(variants_payload: Any) -> list[str]:
    if variants_payload is None:
        return []

    if isinstance(variants_payload, str):
        stripped = variants_payload.strip()
        if not stripped or stripped.upper() == "N/A":
            return []
        if stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                return [stripped]
            return coerce_variants(parsed)
        return [stripped]

    if isinstance(variants_payload, (list, tuple)):
        variants: list[str] = []
        for variant in variants_payload:
            if not isinstance(variant, str):
                continue
            cleaned = variant.strip()
            if cleaned and cleaned.upper() != "N/A":
                variants.append(cleaned)
        return variants

    normalized = str(variants_payload).strip()
    return [normalized] if normalized else []


def parse_process_judge_response(text: str) -> ProcessJudgeResult | None:
    why = None
    aligned_path = None
    score = None

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if why is None:
            why_match = _PROCESS_WHY_PATTERN.match(line)
            if why_match is not None:
                why = why_match.group(1)
                continue
        if aligned_path is None:
            aligned_match = _PROCESS_ALIGNED_PATH_PATTERN.match(line)
            if aligned_match is not None:
                aligned_path = aligned_match.group(1)
                continue
        if score is None:
            score_match = _PROCESS_SCORE_PATTERN.match(line)
            if score_match is not None:
                score = int(score_match.group(1))
                continue

    if score is None:
        return None
    return ProcessJudgeResult(
        prefix_index=-1,
        score=score,
        aligned_path=aligned_path,
        why=why,
        output_text=text,
    )
async def verify_proof(
    problem: str,
    ref_solution: str,
    schema: str,
    generation: str,
    prompt_name: str | os.PathLike | None,
    model: str,
    sampling_kwargs: dict[str, Any] | None = None,
    client: Any | None = None,
    timeout_seconds: int = 900,
    max_retries: int = 3,
    retry_backoff: Sequence[int] = (15, 30, 60, 90, 120),
    provider: str | None = None,
) -> ProofVerificationResult:
    if not generation.strip():
        return ProofVerificationResult(score=0, failure_cause="no_input")

    resolved_provider = resolve_provider(model, provider)
    client = client or get_grader_client(resolved_provider)
    prompt_template = load_evaluator_prompt(prompt_name)
    prompt_text = prompt_template.format(
        problem=problem,
        human_solution=ref_solution,
        marking_scheme=schema,
        solution=generation,
    )
    api_kwargs = dict(sampling_kwargs or {})
    loop = asyncio.get_event_loop()

    async def _call_openai():
        return await loop.run_in_executor(
            None,
            lambda: client.responses.create(model=model, input=prompt_text, **api_kwargs),
        )

    async def _call_gemini():
        gemini_model = _strip_google_prefix(model)
        gemini_config = _build_gemini_config(api_kwargs)
        gemini_async_client = _resolve_gemini_async_client(client)
        return await gemini_async_client.models.generate_content(
            model=gemini_model,
            contents=prompt_text,
            config=gemini_config,
        )

    for attempt in range(1, max_retries + 1):
        try:
            if resolved_provider == "gemini":
                raw = await asyncio.wait_for(_call_gemini(), timeout=timeout_seconds)
                normalized = _normalize_gemini_response(raw)
            else:
                raw = await asyncio.wait_for(_call_openai(), timeout=timeout_seconds)
                normalized = _normalize_openai_response(raw)

            match = re.search(r"<score>(\d+)</score>", normalized.output_text)
            if match is None:
                return ProofVerificationResult(
                    score=0,
                    output_text=normalized.output_text,
                    reasoning_text=normalized.reasoning_text,
                    input_tokens=normalized.input_tokens,
                    output_tokens=normalized.output_tokens,
                    failure_cause="no_score_tag",
                )
            return ProofVerificationResult(
                score=int(match.group(1)),
                output_text=normalized.output_text,
                reasoning_text=normalized.reasoning_text,
                input_tokens=normalized.input_tokens,
                output_tokens=normalized.output_tokens,
            )
        except openai.RateLimitError:
            cause = "rate_limit"
        except asyncio.TimeoutError:
            cause = "timeout"
        except Exception as exc:  # pragma: no cover - depends on provider runtime.
            cause = _classify_gemini_exception(exc) if resolved_provider == "gemini" else "other"

        if attempt < max_retries:
            await asyncio.sleep(retry_backoff[min(attempt - 1, len(retry_backoff) - 1)])
            continue
        return ProofVerificationResult(score=0, failure_cause=cause)

    return ProofVerificationResult(score=0, failure_cause="all_attempts_failed")


async def score_proof_prefixes(
    problem: str,
    ref_solution: str,
    variants: Sequence[str],
    prefix_texts: Sequence[str],
    prompt_name: str | os.PathLike | None,
    model: str,
    sampling_kwargs: dict[str, Any] | None = None,
    client: Any | None = None,
    timeout_seconds: int = 900,
    max_retries: int = 3,
    retry_backoff: Sequence[int] = (15, 30, 60, 90, 120),
    provider: str | None = None,
    max_concurrency: int | None = None,
) -> list[ProcessJudgeResult]:
    if not prefix_texts:
        return []

    prompt_template = load_evaluator_prompt(prompt_name)
    variants_text = format_variants_block(variants)
    prompt_texts = [
        prompt_template.format(
            problem=problem,
            reasoning_so_far=prefix_text,
            reference_solution=ref_solution,
            variants_block=variants_text,
        )
        for prefix_text in prefix_texts
    ]
    return await score_process_judge_prompts(
        prompt_texts=prompt_texts,
        model=model,
        sampling_kwargs=sampling_kwargs,
        client=client,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        retry_backoff=retry_backoff,
        provider=provider,
        max_concurrency=max_concurrency,
        prompt_indices=list(range(len(prefix_texts))),
    )


async def score_process_judge_prompts(
    prompt_texts: Sequence[str],
    model: str,
    sampling_kwargs: dict[str, Any] | None = None,
    client: Any | None = None,
    timeout_seconds: int = 900,
    max_retries: int = 3,
    retry_backoff: Sequence[int] = (15, 30, 60, 90, 120),
    provider: str | None = None,
    max_concurrency: int | None = None,
    prompt_indices: Sequence[int] | None = None,
) -> list[ProcessJudgeResult]:
    if not prompt_texts:
        return []

    resolved_provider = resolve_provider(model, provider)
    client = client or get_grader_client(resolved_provider)
    api_kwargs = dict(sampling_kwargs or {})
    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency and max_concurrency > 0 else None
    loop = asyncio.get_event_loop()
    indexed_prompt_ids = list(prompt_indices) if prompt_indices is not None else list(range(len(prompt_texts)))
    if len(indexed_prompt_ids) != len(prompt_texts):
        raise ValueError("prompt_indices must match prompt_texts length")

    async def _call_openai(prompt_text: str):
        return await loop.run_in_executor(
            None,
            lambda: client.responses.create(model=model, input=prompt_text, **api_kwargs),
        )

    async def _call_gemini(prompt_text: str):
        gemini_model = _strip_google_prefix(model)
        gemini_config = _build_gemini_config(api_kwargs)
        gemini_async_client = _resolve_gemini_async_client(client)
        return await gemini_async_client.models.generate_content(
            model=gemini_model,
            contents=prompt_text,
            config=gemini_config,
        )

    async def _score_single_prompt(prompt_index: int, prompt_text: str) -> ProcessJudgeResult:
        for attempt in range(1, max_retries + 1):
            try:
                async def _call_once():
                    if resolved_provider == "gemini":
                        raw = await _call_gemini(prompt_text)
                        return _normalize_gemini_response(raw)
                    raw = await _call_openai(prompt_text)
                    return _normalize_openai_response(raw)

                if semaphore is None:
                    normalized = await asyncio.wait_for(_call_once(), timeout=timeout_seconds)
                else:
                    async with semaphore:
                        normalized = await asyncio.wait_for(_call_once(), timeout=timeout_seconds)

                parsed = parse_process_judge_response(normalized.output_text)
                if parsed is not None:
                    parsed.prefix_index = prompt_index
                    parsed.output_text = normalized.output_text
                    parsed.reasoning_text = normalized.reasoning_text
                    return parsed
                cause = "no_score_tag"
            except openai.RateLimitError:
                cause = "rate_limit"
            except asyncio.TimeoutError:
                cause = "timeout"
            except Exception as exc:  # pragma: no cover - depends on provider runtime.
                cause = _classify_gemini_exception(exc) if resolved_provider == "gemini" else "other"

            if attempt < max_retries:
                await asyncio.sleep(retry_backoff[min(attempt - 1, len(retry_backoff) - 1)])
                continue
            return ProcessJudgeResult(
                prefix_index=prompt_index,
                score=0,
                parse_failed=True,
                failure_cause=cause,
            )
        return ProcessJudgeResult(
            prefix_index=prompt_index,
            score=0,
            parse_failed=True,
            failure_cause="all_attempts_failed",
        )

    return list(
        await asyncio.gather(
            *(
                _score_single_prompt(prompt_index, prompt_text)
                for prompt_index, prompt_text in zip(indexed_prompt_ids, prompt_texts, strict=True)
            )
        )
    )


def validate_ordered_spans(length: int, spans: Sequence[tuple[int, int]]) -> None:
    for start, end in spans:
        if start < 0 or end > length:
            raise ValueError(f"Span {start}:{end} is out of bounds for length {length}")
        if start > end:
            raise ValueError(f"Span {start}:{end} is invalid")
    for (start1, end1), (start2, end2) in zip(spans, spans[1:]):
        if start2 < end1:
            raise ValueError(f"Spans {start1}:{end1} and {start2}:{end2} overlap")


def _find_delimiter_starts(input_ids: torch.Tensor, delimiter_token_id: int) -> list[int]:
    if input_ids.ndim != 1:
        raise ValueError(f"input_ids must be 1D, got shape {tuple(input_ids.shape)}")
    if not isinstance(delimiter_token_id, int):
        raise ValueError(f"delimiter_token_id must be an int, got {type(delimiter_token_id).__name__}")
    return torch.nonzero(input_ids == delimiter_token_id, as_tuple=False).flatten().tolist()


def split_reward_chunks(input_ids: torch.Tensor, delimiter_token_id: int) -> list[RewardChunk]:
    if input_ids.ndim != 1:
        raise ValueError(f"input_ids must be 1D, got shape {tuple(input_ids.shape)}")

    num_tokens = int(input_ids.numel())
    delimiter_starts = _find_delimiter_starts(input_ids=input_ids, delimiter_token_id=delimiter_token_id)
    if not delimiter_starts:
        token_spans = [(0, num_tokens)]
    else:        
        token_spans = [(0, delimiter_starts[0])] + [
            (start, next_start)
            for start, next_start in zip(delimiter_starts, delimiter_starts[1:])
        ]
        token_spans.append((delimiter_starts[-1], num_tokens))
    validate_ordered_spans(num_tokens, token_spans)
    return [RewardChunk(index=chunk_idx, token_span=token_span) for chunk_idx, token_span in enumerate(token_spans)]


def normalize_prefix_scores(prefix_scores: Sequence[float], min_score: float = 1.0, max_score: float = 5.0) -> list[float]:
    if max_score <= min_score:
        raise ValueError("max_score must be greater than min_score")

    score_range = max_score - min_score
    normalized_scores: list[float] = []
    for score in prefix_scores:
        score_value = float(score)
        if score_value <= 0.0:
            normalized_scores.append(0.0)
            continue
        clipped_score = min(max(score_value, min_score), max_score)
        normalized_scores.append((clipped_score - min_score) / score_range)
    return normalized_scores


def validate_unit_interval(values: Sequence[float], name: str) -> None:
    for idx, value in enumerate(values):
        numeric_value = float(value)
        if numeric_value < 0.0 or numeric_value > 1.0:
            raise ValueError(f"{name}[{idx}]={numeric_value} is outside [0, 1]")


def compute_chunk_rewards(prefix_scores: Sequence[float]) -> list[float]:
    return [float(score) for score in prefix_scores]


def compute_chunk_advantages(prefix_scores: Sequence[float]) -> list[float]:
    if not prefix_scores:
        return []
    final_score = float(prefix_scores[-1])
    advantages = [float(prefix_scores[0])]
    for idx in range(1, len(prefix_scores)):
        advantages.append(float(prefix_scores[idx] - prefix_scores[idx - 1] + final_score))
    return advantages


def maybe_clip_chunk_advantages_for_length(
    output_token_ids: Sequence[int],
    eos_token_id: int | None,
    chunk_advantages: Sequence[float],
    is_clip_length: bool = False,
) -> tuple[list[float], bool, bool]:
    clipped_advantages = [float(value) for value in chunk_advantages]
    if not isinstance(eos_token_id, int):
        return clipped_advantages, False, False

    is_overflow = bool(output_token_ids) and eos_token_id not in output_token_ids
    if not is_clip_length or not is_overflow:
        return clipped_advantages, is_overflow, False
    return [0.0] * len(clipped_advantages), is_overflow, True


def assign_chunk_values_to_output_tokens(
    num_output_tokens: int,
    chunk_token_spans: Sequence[tuple[int, int]],
    chunk_values: Sequence[float]
) -> list[float]:
    if len(chunk_token_spans) != len(chunk_values):
        raise ValueError(
            "chunk_token_spans and chunk_values must have the same length, got "
            f"{len(chunk_token_spans)} and {len(chunk_values)}"
        )
    validate_ordered_spans(num_output_tokens, chunk_token_spans)

    values = [0.0] * num_output_tokens
    for ((start, end), chunk_value) in zip(chunk_token_spans, chunk_values):
        for token_idx in range(start, end):
            values[token_idx] = float(chunk_value)
    return values
