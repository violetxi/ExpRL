"""
Offline DP+Ray vLLM runner for judge scripts.

Mirrors the architecture of verl/trainer/vllm_generation_ray_dp.py: spawns N
Ray actors, each holding a vLLM LLM on its own GPU set, shards the chat-style
inputs across them, and returns generated text in original order.

Why offline instead of the OpenAI HTTP API server: vLLM 0.8.5's V1 engine
does not actually skip torch.compile under `--enforce-eager` when invoked via
the api_server — the @support_torch_compile decorator on the model class
still pulls in `torch._inductor.codecache.get_system`, which trips
`ImportError: cannot import name 'triton_key' from triton.compiler.compiler`
in this conda env's triton/torch combination. The offline `LLM(...)` API
takes `enforce_eager=True` directly and never invokes inductor — the same
code path the sampling pipeline (vllm_generation_ray_dp.py) uses without
issue.
"""

import os
from typing import Dict, List, Optional

import ray
from transformers import AutoTokenizer

# Map external short names (the --model values judge scripts have always
# accepted) to their HF repo paths.
LOCAL_MODELS: Dict[str, str] = {
    "qwen/qwen3_0_6b":         "Qwen/Qwen3-0.6B",
    "qwen/qwen3_1_7b":         "Qwen/Qwen3-1.7B",
    "qwen/qwen3_4b":           "Qwen/Qwen3-4B",
    "qwen/qwen3_8b":           "Qwen/Qwen3-8B",
    "qwen/qwen3_14b":          "Qwen/Qwen3-14B",
    "qwen/qwen3_4b_instruct":  "Qwen/Qwen3-4B-Instruct-2507",
    "qwen/qwen3_30b_instruct": "Qwen/Qwen3-30B-A3B-Instruct-2507",
}


def resolve_local_model(model_name: str) -> str:
    """Translate short --model names to HF repo paths; pass through unknown values."""
    return LOCAL_MODELS.get(model_name, model_name)


def visible_gpu_count() -> int:
    cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_devices:
        return len([d for d in cuda_devices.split(",") if d.strip()])
    return 0


@ray.remote(num_gpus=1)
class JudgeVLLMWorker:
    """Holds one vLLM `LLM` on one GPU set (`tensor_parallel_size` GPUs)."""

    def __init__(
        self,
        worker_id: int,
        model_path: str,
        tensor_parallel_size: int,
        gpu_memory_utilization: float,
        max_model_len: Optional[int],
        max_num_batched_tokens: int,
        max_num_seqs: int,
        enforce_eager: bool = True,
        trust_remote_code: bool = False,
    ):
        # Lazy imports inside the actor — vLLM init imports torch/cuda and we
        # want that to happen in the actor process, not the driver.
        from vllm import LLM

        self.worker_id = worker_id
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=trust_remote_code,
        )

        llm_kwargs = dict(
            model=model_path,
            tokenizer=model_path,
            trust_remote_code=trust_remote_code,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
            enforce_eager=enforce_eager,
            disable_log_stats=True,
        )
        if max_model_len is not None:
            llm_kwargs["max_model_len"] = max_model_len

        print(f"[JudgeWorker {worker_id}] Initializing vLLM "
              f"(model={model_path}, TP={tensor_parallel_size}, "
              f"enforce_eager={enforce_eager})...")
        self.llm = LLM(**llm_kwargs)
        print(f"[JudgeWorker {worker_id}] vLLM ready.")

    def generate(
        self,
        messages_list: List[List[Dict[str, str]]],
        temperature: float,
        max_tokens: int,
        top_p: float = 1.0,
        enable_thinking: Optional[bool] = None,
    ) -> List[str]:
        """Tokenize via chat template, generate one completion per input."""
        from vllm import SamplingParams

        template_kwargs: Dict[str, object] = {}
        if enable_thinking is not None:
            # Qwen3 chat templates expose this; other models silently ignore it.
            template_kwargs["enable_thinking"] = enable_thinking

        prompts = [
            self.tokenizer.apply_chat_template(
                m, add_generation_prompt=True, tokenize=False, **template_kwargs,
            )
            for m in messages_list
        ]
        sp = SamplingParams(
            n=1,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
        outputs = self.llm.generate(prompts, sp)
        return [o.outputs[0].text.strip() if o.outputs else "" for o in outputs]


def run_judge_dp(
    model_name: str,
    messages_list: List[List[Dict[str, str]]],
    *,
    temperature: float = 0.0,
    max_tokens: int = 4096,
    data_parallel_size: Optional[int] = None,
    tensor_parallel_size: int = 1,
    gpu_memory_utilization: float = 0.85,
    max_model_len: Optional[int] = None,
    max_num_batched_tokens: int = 32768,
    max_num_seqs: int = 2048,
    top_p: float = 1.0,
    trust_remote_code: bool = False,
    enable_thinking: Optional[bool] = None,
) -> List[str]:
    """Run a batch of chat-style judge calls through DP+Ray offline vLLM.

    Returns one response string per input, in the original order.

    `data_parallel_size` defaults to `len(CUDA_VISIBLE_DEVICES) // tensor_parallel_size`.
    With TP=1 on a 4-GPU node, that's DP=4.
    """
    n = len(messages_list)
    if n == 0:
        return []

    if data_parallel_size is None:
        visible = visible_gpu_count()
        data_parallel_size = max(visible // max(tensor_parallel_size, 1), 1)

    model_path = resolve_local_model(model_name)
    print(f"[run_judge_dp] DP={data_parallel_size} TP={tensor_parallel_size} "
          f"calls={n} model={model_path}")

    if not ray.is_initialized():
        # Cap num_cpus: on big nodes (e.g. 288-core gh200) Ray otherwise
        # prestarts hundreds of python workers that can hang during import
        # and stall actor scheduling. We only need DP workers + a few helpers.
        ray_num_cpus = max(data_parallel_size * 4, 8)
        ray.init(ignore_reinit_error=True, num_cpus=ray_num_cpus)

    workers = [
        JudgeVLLMWorker.remote(
            worker_id=i,
            model_path=model_path,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
            max_num_batched_tokens=max_num_batched_tokens,
            max_num_seqs=max_num_seqs,
            enforce_eager=True,
            trust_remote_code=trust_remote_code,
        )
        for i in range(data_parallel_size)
    ]

    shard_size = (n + data_parallel_size - 1) // data_parallel_size
    futures = []
    ranges = []
    for w_idx in range(data_parallel_size):
        start = w_idx * shard_size
        end = min(start + shard_size, n)
        if start >= n:
            continue
        ranges.append((start, end))
        futures.append(workers[w_idx].generate.remote(
            messages_list[start:end], temperature, max_tokens, top_p, enable_thinking,
        ))
    print(f"[run_judge_dp] dispatched {len(futures)} shards of ~{shard_size} prompts each")

    results: List[str] = [""] * n
    for (start, end), fut in zip(ranges, futures):
        shard = ray.get(fut)
        for i, r in enumerate(shard):
            results[start + i] = r
    return results


def parse_score_likert(response_str: str) -> Optional[int]:
    """Parse a `Score: N` line where N is 1-5. Returns None if unparseable."""
    try:
        if "Score:" in response_str:
            score_str = response_str.split("Score:", 1)[1].lstrip().split("\n", 1)[0]
        else:
            score_str = response_str.split("Score", 1)[1].lstrip().split("\n", 1)[0]
        score_str = score_str.replace("**", "").replace("*", "").strip()
        score = int(score_str[0])
        if score in (1, 2, 3, 4, 5):
            return score
    except Exception:
        pass
    return None
