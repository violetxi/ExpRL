# ExpRL: Exploratory RL for LLM Mid-Training

Official code for the paper **"ExpRL: Exploratory RL for LLM Mid-Training"**
([`paper.pdf`](paper.pdf)).

Sparse-reward RL for LLM reasoning works only when the base model already has
**coverage** over productive reasoning paths. ExpRL is an RL-based *mid-training*
(priming) method that builds this coverage from large corpora of question–answer
data. Rather than imitating reference solutions, ExpRL uses them as **reward
scaffolds**: the reference is hidden from the policy and used only to let an LLM
judge score on-policy reasoning traces for partial progress. Optimizing these
**dense, reference-guided rewards** shifts probability mass toward productive
reasoning paths, yielding a stronger initialization for subsequent sparse-reward
RL than SFT, sparse-reward GRPO, or self-distillation.

The method has **two stages**, each implemented on a separate training stack that
is vendored into this repo:

| Stage | What it does | Stack | Directory |
|-------|--------------|-------|-----------|
| **Stage 1 — RL priming** | On-policy RL with dense reference-guided rewards (ExpRL-Outcome / ExpRL-Process), plus the SFT / sparse-GRPO / self-distillation baselines. An LLM judge scores rollouts against reference solutions. | fork of [verl](https://github.com/volcengine/verl) | [`stage1_verl/`](stage1_verl/) |
| **Stage 2 — downstream RL** | Standard sparse-reward GRPO (binary final-answer reward) initialized from each Stage-1 checkpoint. | fork of [PipelineRL](https://github.com/ServiceNow/PipelineRL) | [`stage2_pipeline_rl/`](stage2_pipeline_rl/) |

Base policy: **Qwen3-4B-Instruct-2507** (also used as the judge). The mixed-domain
experiments use an 8B policy with a 4B judge. Held-out benchmarks: AIME 2025,
AIME 2026, HMMT (Nov 2025), IMO-AnswerBench (+ GPQA / Olympiad-Physics / LCB for
mixed-domain).

---

## Repository layout

```
ExpRL/
├── paper.pdf                 # the paper
├── stage1_verl/              # Stage 1: dense reference-guided RL priming (verl fork)
│   └── variational_inference/math/exp_rl/   # entry-point scripts
│       ├── train_stage1/     #   priming runs (ExpRL-Outcome/-Process + baselines)
│       ├── inference_stage1/ #   sample rollouts for pass@k on held-out benchmarks
│       ├── inference_stage2/ #   sample rollouts from Stage-2 checkpoints
│       └── eval/             #   pass@1 / pass@k scoring
└── stage2_pipeline_rl/       # Stage 2: downstream sparse-reward GRPO (PipelineRL fork)
    ├── conf/                 #   exp_stage2_*.yaml configs
    └── scripts/              #   run_exp_stage2_*.sh entry points
```

Each stage has its own README with detailed setup and run commands:
[`stage1_verl/README.md`](stage1_verl/README.md),
[`stage2_pipeline_rl/README.md`](stage2_pipeline_rl/README.md).

---

## Environment

The two stages use **separate conda environments** (different, partly conflicting
pins for vLLM / torch / flash-attn).

**Stage 1 (`verl`):**
```bash
conda create -n verl python=3.10 -y && conda activate verl
cd stage1_verl && pip install -e .
# newer torchao breaks the transformers import on torch 2.6:
pip install torchao==0.9.0
```

**Stage 2 (`prl`):** needs a **CUDA 12.4 toolkit** with `CUDA_HOME` set
(flash-attn / deepspeed build from source):
```bash
conda create -n prl python=3.11 -y && conda activate prl
cd stage2_pipeline_rl && ./install.sh
```
The Stage-2 `pyproject.toml` pins `fastapi==0.115.6` (newer starlette breaks the
prometheus instrumentation in vLLM's OpenAI server).

Set the usual credentials yourself: `HF_TOKEN` (model/dataset access),
optionally `WANDB_API_KEY` (logging) and `OPENAI_API_KEY` / `OPENAI_BASE_URL`
(if you point the judge at an external endpoint).

---

## Conventions for paths and checkpoints

The entry-point scripts are parametrized with two environment variables so they
run on any machine without editing:

- **`EXPRL_ROOT`** — absolute path to the stage checkout (Stage-1 scripts `cd`
  here). Defaults to auto-detecting the repo root from the script location; set
  it explicitly when launching via SLURM:
  `export EXPRL_ROOT=/path/to/ExpRL/stage1_verl`.
- **`HF_NAMESPACE`** — Hugging Face org/user for checkpoints and intermediate
  datasets (defaults to `violetxi`, the authors' namespace). Set it to your own
  account when you push your Stage-1 outputs:
  `export HF_NAMESPACE=<your-hf-username>`.

Stage-2 configs additionally take a **`model_path`** override on the command line
(the Stage-1 checkpoint to continue from), e.g.
`bash scripts/run_exp_stage2_grpo.sh model_path=/path/to/your/stage1_ckpt`.

> The `violetxi/*` names are the authors' artifacts. Either set `HF_NAMESPACE` to
> use your own re-trained checkpoints, or use the public `violetxi/*` checkpoints
> once released. SLURM scripts ship with placeholder `--account=YOUR_ACCOUNT`
> / `--partition=YOUR_PARTITION`.

---

## End-to-end workflow

1. **Prepare data** (Stage 1). The training scripts read `data/instruct/*.parquet`
   built from the math mid-training sources (InT + POPE; plus SciKnow + LCB for
   mixed-domain). See [`stage1_verl/README.md`](stage1_verl/README.md) and
   `variational_inference/math/data_processing/instruct.py`.

2. **Stage 1 — prime the base model** (env `verl`, single node 8×H100):
   ```bash
   cd stage1_verl
   bash variational_inference/math/exp_rl/train_stage1/qwen4b/dense_outcome.sh   # ExpRL-Outcome
   bash variational_inference/math/exp_rl/train_stage1/qwen4b/dense_pr_delta.sh  # ExpRL-Process
   # baselines: grpo.sh, self_distill_lr1e-7.sh, sft.sh
   ```

3. **Stage 2 — downstream sparse-reward RL** from a Stage-1 checkpoint
   (env `prl`, single node 8×H100):
   ```bash
   cd stage2_pipeline_rl
   export HF_NAMESPACE=<your-hf-username>   # where your Stage-1 ckpts live
   bash scripts/run_exp_stage2_grpo.sh      # or dense_outcome / dense_process / sft / self_distill
   ```

4. **Evaluate** — sample rollouts (`inference_stage1/`, `inference_stage2/`) and
   score with `eval/` (`verl.trainer.main_eval`) to get pass@1 / pass@k on the
   held-out benchmarks.

---

## Method → script map (Stage 1, `train_stage1/qwen4b/`)

| Paper method | Script | Trainer |
|--------------|--------|---------|
| ExpRL-Outcome | `dense_outcome.sh` | `verl.trainer.main_rl_genrm` (dense outcome reward, GRPO-norm) |
| ExpRL-Process (DeltaNorm, main) | `dense_pr_delta.sh` | `main_rl_genrm` (segment advantages) |
| ExpRL-Process (EndNorm / GroupNorm ablations) | `dense_pr_end.sh` / `dense_pr_gppo.sh` | `main_rl_genrm` |
| Sparse-reward GRPO baseline | `grpo.sh` | `verl.trainer.main_ppo` |
| Self-distillation baseline | `self_distill_lr1e-7.sh` | `verl.trainer.main_onpolicy_distill` |
| SFT baseline | `sft.sh` | `verl.trainer.fsdp_sft_trainer` |

The dense-reward advantage estimators (`GPPO`, `GPR_E`, `PR_BN`, `PR_DELTA`) live
in `stage1_verl/verl/trainer/ppo/core_algos.py`; the LLM-judge prompts in
`stage1_verl/verl/utils/prompt_templates/math_likelihood_{judge,process}.py`.
The 8B mixed-domain runs are under `train_stage1/qwen8b/`.

---

## Citation

```bibtex
@article{xiang2026exprl,
  title  = {ExpRL: Exploratory RL for LLM Mid-Training},
  author = {Xiang, Violet and Setlur, Amrith and Blagden, Chase and Haber, Nick and Kumar, Aviral},
  year   = {2026}
}
```

## License & attribution

This repo vendors forks of [verl](https://github.com/volcengine/verl) and
[PipelineRL](https://github.com/ServiceNow/PipelineRL); their upstream licenses
and `NOTICE`/`Notice.txt` files are retained inside each stage directory. See
[`LICENSE`](LICENSE) for this repository's license.
