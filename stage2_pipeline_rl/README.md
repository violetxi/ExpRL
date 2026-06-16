# ExpRL — Stage 2: Downstream Sparse-Reward RL (PipelineRL fork)

This is the **Stage 2** code for ExpRL (see the [top-level README](../README.md)).
It is a fork of [PipelineRL](https://github.com/ServiceNow/PipelineRL) used to run
standard **sparse-reward GRPO** (binary final-answer reward) initialized from a
Stage-1 checkpoint. PipelineRL's asynchronous pipeline speeds up the long Stage-2
runs (~500 optimization steps for the answer-based math setting).

The binary reward / answer grading lives in
`pipelinerl/domains/math/` (`verifier_api.py` extracts the `\boxed{}` answer and
checks equivalence with `math-verify`; `rollouts.py` shapes the reward).

## Setup (single node, 8×H100)

Requires a **CUDA 12.4 toolkit** with `CUDA_HOME` set (flash-attn / deepspeed
build from source):

```bash
conda create -n prl python=3.11 -y && conda activate prl
export CUDA_HOME=/usr/local/cuda-12.4    # adjust to your toolkit
./install.sh
```

`pyproject.toml` pins `fastapi==0.115.6` (newer starlette breaks the prometheus
instrumentation in vLLM's OpenAI server), `vllm==0.8.5.post1`,
`transformers==4.51.1`, `flash-attn==2.7.4.post1`. Set `HF_TOKEN` (and
`WANDB_API_KEY` if logging).

## Run

Each script continues sparse-reward GRPO from the matching Stage-1 init:

```bash
export HF_NAMESPACE=<your-hf-username>     # where your Stage-1 checkpoints live
bash scripts/run_exp_stage2_grpo.sh          # from the GRPO-primed init
bash scripts/run_exp_stage2_dense_outcome.sh # from the ExpRL-Outcome init
bash scripts/run_exp_stage2_dense_process.sh # from the ExpRL-Process init
bash scripts/run_exp_stage2_sft.sh           # from the SFT init
bash scripts/run_exp_stage2_self_distill.sh  # from the self-distillation init
```

Each `run_exp_stage2_*.sh` calls `python -m pipelinerl.launch
--config-name=exp_stage2_<method>` and writes to `./results/...`.

**Choosing the Stage-1 checkpoint.** Configs default to
`${HF_NAMESPACE:-violetxi}/exp_stage1_qwen3-4b_<method>` (set via the
`model_path` field in `conf/exp_stage2_*.yaml`). Override per run on the CLI:

```bash
bash scripts/run_exp_stage2_grpo.sh \
  model_path=/path/to/your/stage1_ckpt \
  output_dir=./results/my-stage2-grpo
```

Stage-2 training uses the same InT + POPE prompt mixture as Stage 1 but with all
reference-solution information removed — only the binary final-answer reward.

## Layout

- `conf/base.yaml` + `conf/exp_stage2_*.yaml` — Hydra configs (pull the
  `finetune/`, `rewards/`, `streams/`, `deepspeed/`, `accelerate/` groups).
- `pipelinerl/` — the async RL pipeline (orchestrator, actor, preprocessor,
  trainer, verifier). `pipelinerl/domains/math/` is the answer-based reward.
- `scripts/` — `run_exp_stage2_*.sh` entry points, plus `load_dataset.py` and
  `upload_to_hf.py` utilities; `upload_checkpoints.py/.sh` push trained models.
- `LICENSE` / `NOTICE` are upstream PipelineRL's, retained for attribution.

> The math domain still contains dormant proof-grading code paths
> (`verify_proof`, `MathProofEnvironment`) inherited from upstream; they are not
> exercised by any of the `exp_stage2_*` configs.
