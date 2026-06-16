#!/bin/bash
#SBATCH --job-name=judge_qwen4b_outcome_code
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

mkdir -p slurm_logs

echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"

# envs
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate verl
unset ROCR_VISIBLE_DEVICES

export TORCHDYNAMO_VERBOSE=1
export VLLM_TORCH_COMPILE_LEVEL=0
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Configuration
DATA_PATH="${HF_NAMESPACE:-violetxi}/judge_calibration_lcb_v5_qwen8b"
ORACLE_PATH="data/instruct/livecodebench_v6_with_oracle/train.parquet"
MODEL_NAME="qwen/qwen3_4b_instruct"
OUTPUT_PATH="${HF_NAMESPACE:-violetxi}/Qwen4b-Instruct-Judge-lcb-v6-calibration-multi-ref"
TEMPERATURE=0.0
# DP=4 (one vLLM worker per GPU, TP=1) — Qwen3-4B fits comfortably on one
# H100/GH200, and the DP+Ray path mirrors verl/trainer/vllm_generation_ray_dp.py.
TENSOR_PARALLEL_SIZE=1
DATA_PARALLEL_SIZE=4

# Run from repo root so `from verl.utils.prompt_templates...` resolves
cd "${EXPRL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && while [ ! -f setup.py ] && [ "$PWD" != / ]; do cd ..; done && pwd)}"
echo "Starting LLM likelihood judge (code)..."
echo "Model: $MODEL_NAME"
echo "Data: $DATA_PATH"
echo "Oracle: $ORACLE_PATH"
echo "Output: $OUTPUT_PATH"

python variational_inference/gen_rm/llm_likelihood_judge_code.py \
    --data "$DATA_PATH" \
    --oracle "$ORACLE_PATH" \
    --model "$MODEL_NAME" \
    --use_local_model \
    --tensor_parallel_size $TENSOR_PARALLEL_SIZE \
    --data_parallel_size $DATA_PARALLEL_SIZE \
    --temperature $TEMPERATURE \
    --multi_ref \
    --output "$OUTPUT_PATH"

exit_code=$?
echo "End Time: $(date)"
if [ $exit_code -eq 0 ]; then
    echo "Job completed successfully!"
else
    echo "Job failed with exit code: $exit_code"
fi

exit $exit_code
