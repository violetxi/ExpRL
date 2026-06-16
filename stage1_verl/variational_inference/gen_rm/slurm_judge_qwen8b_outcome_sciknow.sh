#!/bin/bash
#SBATCH --job-name=judge_qwen8b_sciknow
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

# Usage:
#   sbatch slurm_judge_qwen8b_outcome_sciknow.sh [mcq|oe|all]
# Default "all" runs MCQ then OE in the same allocation.

set -euo pipefail
mkdir -p slurm_logs

DATASET_KEY="${1:-all}"
case "$DATASET_KEY" in
    mcq|oe|all) ;;
    *)
        echo "ERROR: unknown dataset key '$DATASET_KEY' (expected: mcq | oe | all)" >&2
        exit 2
        ;;
esac

declare -A SCIKNOW_DATA=(
    [mcq]="${HF_NAMESPACE:-violetxi}/judge_calibration_sciknoweval_mcq_l3_loose_qwen8b"
    [oe]="${HF_NAMESPACE:-violetxi}/judge_calibration_sciknoweval_oe_v1_qwen8b"
)
declare -A SCIKNOW_OUTPUT=(
    [mcq]="${HF_NAMESPACE:-violetxi}/Qwen8b-NoThink-Judge-sciknoweval-mcq-l3-loose-qwen8b-multi-ref"
    [oe]="${HF_NAMESPACE:-violetxi}/Qwen8b-NoThink-Judge-sciknoweval-oe-v1-qwen8b-multi-ref"
)
if [ "$DATASET_KEY" = "all" ]; then
    KEYS=(mcq oe)
else
    KEYS=("$DATASET_KEY")
fi

echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Dataset key: $DATASET_KEY"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate verl
unset ROCR_VISIBLE_DEVICES

export TORCHDYNAMO_VERBOSE=1
export VLLM_TORCH_COMPILE_LEVEL=0
export CUDA_VISIBLE_DEVICES=0,1,2,3

MODEL_NAME="qwen/qwen3_8b"
TEMPERATURE=0.0
TENSOR_PARALLEL_SIZE=1
DATA_PARALLEL_SIZE=4

cd "${EXPRL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && while [ ! -f setup.py ] && [ "$PWD" != / ]; do cd ..; done && pwd)}"
overall_exit=0
for key in "${KEYS[@]}"; do
    DATA_PATH="${SCIKNOW_DATA[$key]}"
    OUTPUT_PATH="${SCIKNOW_OUTPUT[$key]}"
    echo "================================================================"
    echo "[$key] Starting LLM likelihood judge (SciKnowEval, multi-ref)"
    echo "  Model:  $MODEL_NAME (no-thinking)"
    echo "  Data:   $DATA_PATH"
    echo "  Output: $OUTPUT_PATH"
    echo "  Time:   $(date)"
    echo "================================================================"

    set +e
    python variational_inference/gen_rm/llm_likelihood_judge_sciknow.py \
        --data "$DATA_PATH" \
        --model "$MODEL_NAME" \
        --use_local_model \
        --tensor_parallel_size $TENSOR_PARALLEL_SIZE \
        --data_parallel_size $DATA_PARALLEL_SIZE \
        --temperature $TEMPERATURE \
        --no_thinking \
        --multi_ref \
        --output "$OUTPUT_PATH"
    rc=$?
    set -e

    if [ $rc -eq 0 ]; then
        echo "[$key] Completed successfully."
    else
        echo "[$key] Failed with exit code $rc."
        overall_exit=$rc
    fi
done

echo "End Time: $(date)"
exit $overall_exit
