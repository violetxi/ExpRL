#!/bin/bash
#SBATCH --job-name=judge_qwen4b_omni-l7_h100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --gres=gpu:H100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_PARTITION
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

# Create logs directory if it doesn't exist
mkdir -p slurm_logs

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Number of GPUs: $SLURM_GPUS_PER_NODE"
echo "Start Time: $(date)"

# envs
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate verl

##### Fix AMD/NVIDIA env conflict (*** ONLY ON HAI CLUSTER ***) #####
unset ROCR_VISIBLE_DEVICES
########################################################################

# Set CUDA visible devices
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Configuration
DATA_PATH="${HF_NAMESPACE:-violetxi}/omni-rule-l7-above-gemini-pro-filtered_qwen3-4b-instruct-2507-steps"
MODEL_NAME="qwen/qwen3_4b_instruct_2507"  # Options: qwen/qwen3_4b, qwen/qwen3_8b, qwen/qwen3_14b, qwen/qwen3_4b_instruct_2507
OUTPUT_PATH="${HF_NAMESPACE:-violetxi}/Qwen4b-2507-Judge-partial-qwen4b-instruct-2507-omni-l7-step-score"
TEMPERATURE=0.0
MAX_CONCURRENCY=256

# Navigate to script directory
cd "${EXPRL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && while [ ! -f setup.py ] && [ "$PWD" != / ]; do cd ..; done && pwd)}"
# Run the LLM judge with local model launch
echo "Starting LLM likelihood judge..."
echo "Model: $MODEL_NAME"
echo "Data: $DATA_PATH"
echo "Output: $OUTPUT_PATH"

python llm_likelihood_judge_process.py \
    --data "$DATA_PATH" \
    --model "$MODEL_NAME" \
    --use_local_model \
    --launch_model \
    --temperature $TEMPERATURE \
    --max-concurrency $MAX_CONCURRENCY \
    --output "$OUTPUT_PATH"

# Print completion status
exit_code=$?
echo "End Time: $(date)"
if [ $exit_code -eq 0 ]; then
    echo "Job completed successfully!"
else
    echo "Job failed with exit code: $exit_code"
fi

exit $exit_code
