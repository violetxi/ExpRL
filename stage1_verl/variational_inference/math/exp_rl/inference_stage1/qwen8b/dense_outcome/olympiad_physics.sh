#!/bin/bash
#SBATCH --job-name=olympiad_physics_qwen8b_dense_outcome_stage1
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

cd "${EXPRL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && while [ ! -f setup.py ] && [ "$PWD" != / ]; do cd ..; done && pwd)}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate verl
unset ROCR_VISIBLE_DEVICES

export VLLM_TORCH_COMPILE_LEVEL=0

# --- Preflight: open-ended rows score via the Gemini LLM judge ---------------
# All rows score via verl/utils/reward_score/olympiadbench_oe_judge.py, which
# calls Gemini. That scorer is fail-soft: a missing package or unset key makes
# it return 0.0 for EVERY row, silently producing an all-zero `score` column
# with no job failure. Fail loudly here instead.
: "${GEMINI_API_KEY:?GEMINI_API_KEY is not set — OE scoring would silently zero every row. Export it before launching.}"
python3 -c "import google.generativeai" 2>/dev/null || {
    echo "ERROR: 'google-generativeai' is not installed in this env — OE scoring would silently zero every row." >&2
    echo "       Run: pip install google-generativeai" >&2
    exit 1
}
# -----------------------------------------------------------------------------


###### OlympiadBench-OE Physics (OE_TO_physics_en_COMP) — Dense Outcome (exp_rl_all_domains_stage1_qwen8b_dense_outcome)
test_path=data/instruct/olympiadbench_oe_physics/train.parquet
python3 -m verl.trainer.vllm_generation_ray_dp \
--config-path=config --config-name=vllm_generation \
model.path=${HF_NAMESPACE:-violetxi}/exp_rl_all_domains_stage1_qwen8b_dense_outcome \
model.revision=main \
data.path=$test_path \
data.prompt_key=prompt \
data.n_samples=128 \
data.batch_size=2048 \
data.use_chat_template=True \
data.enable_thinking=False \
data.output_path=${HF_NAMESPACE:-violetxi}/olympiad_physics_stage1_qwen8b_dense_outcome \
data.chunk_output_dir=./chunks_olympiad_physics_stage1_qwen8b_dense_outcome \
rollout.prompt_length=3072 \
rollout.response_length=16384 \
rollout.temperature=0.8 \
rollout.data_parallel_size=4 \
rollout.tensor_model_parallel_size=1 \
rollout.gpu_memory_utilization=0.8 \
rollout.max_num_batched_tokens=40000 \
rollout.max_num_seqs=4096 \
rollout.enforce_eager=True \
scoring.enabled=true
