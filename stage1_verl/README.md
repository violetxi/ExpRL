# ExpRL — Stage 1: Reference-Guided RL Priming (verl fork)

This is the **Stage 1** code for ExpRL (see the [top-level README](../README.md)).
It is a fork of [verl](https://github.com/volcengine/verl) extended to run
on-policy RL with **dense, reference-guided rewards**: an LLM judge scores each
on-policy rollout (or rollout prefix) against a hidden reference solution, and the
policy is optimized on those dense scores.

## What's ExpRL-specific here

- `verl/trainer/ppo/core_algos.py` — the dense-reward advantage estimators
  `GPPO` / `GPR_E` / `PR_BN` / `PR_DELTA` (outcome- and process-level).
- `verl/trainer/main_rl_genrm.py` + `verl/trainer/ppo/ray_trainer_genrm.py` —
  RL with the LLM-judge reward (ExpRL-Outcome / ExpRL-Process).
- `verl/trainer/main_onpolicy_distill.py` — the self-distillation baseline.
- `verl/utils/prompt_templates/math_likelihood_{judge,process}.py` — the judge
  prompts (full-trace outcome reward / per-step process reward).
- `variational_inference/gen_rm/` — LLM-judge reward implementation + calibration.
- `variational_inference/math/exp_rl/` — all entry-point scripts.
- `variational_inference/math/data_processing/` — dataset construction (`instruct.py`).

## Setup

```bash
conda create -n verl python=3.10 -y && conda activate verl
pip install -e .
pip install torchao==0.9.0   # newer torchao breaks transformers import on torch 2.6
```

Set `HF_TOKEN` (and `WANDB_API_KEY` if logging). Scripts activate the `verl` env
and `cd` to this directory — override the location with
`export EXPRL_ROOT=/path/to/ExpRL/stage1_verl` (needed under SLURM).

## Data

Training scripts read `data/instruct/<dataset>/*.parquet` (math mid-training mix:
InT + POPE; plus SciKnow + LCB for mixed-domain). Build these with
`variational_inference/math/data_processing/instruct.py` and the `download_*`
helpers in that directory. Source datasets are pulled from the `${HF_NAMESPACE}`
namespace (default `violetxi`) on the Hugging Face Hub.

## Run (single node, 8×H100)

```bash
# ExpRL priming variants
bash variational_inference/math/exp_rl/train_stage1/qwen4b/dense_outcome.sh   # ExpRL-Outcome
bash variational_inference/math/exp_rl/train_stage1/qwen4b/dense_pr_delta.sh  # ExpRL-Process (DeltaNorm, main)
bash variational_inference/math/exp_rl/train_stage1/qwen4b/dense_pr_end.sh    # ExpRL-Process (EndNorm)
bash variational_inference/math/exp_rl/train_stage1/qwen4b/dense_pr_gppo.sh   # ExpRL-Process (GroupNorm)
# baselines
bash variational_inference/math/exp_rl/train_stage1/qwen4b/grpo.sh            # sparse-reward GRPO
bash variational_inference/math/exp_rl/train_stage1/qwen4b/self_distill_lr1e-7.sh
bash variational_inference/math/exp_rl/train_stage1/qwen4b/sft.sh
```

8B mixed-domain runs live under `train_stage1/qwen8b/`.

Process rewards slice each rollout into prefixes on the `###` step delimiter and
query the judge per prefix (see `main_rl_genrm` + `core_algos.py`). ExpRL-Outcome
uses a GRPO-style normalized dense reward; ExpRL-Process uses REINFORCE-style
segment advantages without group normalization.

## Evaluate

Sample rollouts on held-out benchmarks, then score:

```bash
# 1) sample (vLLM data-parallel) — writes a results dataset under ${HF_NAMESPACE}
bash variational_inference/math/exp_rl/inference_stage1/aime_25/dense_outcome.sh
# 2) score pass@1 / pass@k
python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime25_stage1_dense_outcome data.response_key=response
```

`inference_stage1/` evaluates Stage-1 checkpoints; `inference_stage2/` evaluates
Stage-2 checkpoints; `eval/eval_stage1.sh` / `eval_stage2.sh` collect the
`main_eval` commands per benchmark.

## Notes

- The checkpoint to evaluate is set via `model.path=` in the inference scripts.
  The ExpRL variants default to the public checkpoints
  `${HF_NAMESPACE:-violetxi}/ExpRL-Outcome-Qwen3-4B-Instruct`,
  `…/ExpRL-Process-Qwen3-4B-Instruct`, and `…/ExpRL-Outcome-Qwen3-8B` (8B mixed-domain);
  baselines point at their own `exp_stage1_*` inits. Set `HF_NAMESPACE` to your
  account or pass `model.path=<local-or-hf-path>` on the CLI.
- `LICENSE` / `Notice.txt` are upstream verl's, retained for attribution.
