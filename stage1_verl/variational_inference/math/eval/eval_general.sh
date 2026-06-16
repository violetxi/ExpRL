#!/bin/bash
#SBATCH --job-name=eval_general
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=256
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

# envs
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate verl

##### Fix AMD/NVIDIA env conflict (*** ONLY ON HAI CLUSTER ***) #####
unset ROCR_VISIBLE_DEVICES
########################################################################

# project dir
cd "${EXPRL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && while [ ! -f setup.py ] && [ "$PWD" != / ]; do cd ..; done && pwd)}"
set -x

# 
# python verl/trainer/main_eval.py \
# data.path=${HF_NAMESPACE:-violetxi}/omni-math-above-l7-gemini-flash-sol \
# data.response_key=llm_solutions \
# data.ground_truth_key=answer

# python verl/trainer/main_eval.py \
# data.path=${HF_NAMESPACE:-violetxi}/omni-math-above-l7-rule-gemini-pro-sol \
# data.response_key=llm_solutions \
# data.ground_truth_key=answer

# python verl/trainer/main_eval.py \
# data.path=${HF_NAMESPACE:-violetxi}/omni-rule-l7-above-gemini-pro-filtered_qwen3-4b-instruct-2507 \
# data.ground_truth_key=answer

# python verl/trainer/main_eval.py \
# data.path=${HF_NAMESPACE:-violetxi}/process_rollout-omni-rule-l7-above-gemini-pro-filtered \
# data.ground_truth_key=answer

# python verl/trainer/main_eval.py \
# data.path=${HF_NAMESPACE:-violetxi}/omni-rule-l7-above-gemini-pro-filtered_qwen3-4b-instruct-k32 \
# data.ground_truth_key=answer

# #### SFT OpenThoughts 8k
# # AIME 2024
# python verl/trainer/main_eval.py \
# data.path=${HF_NAMESPACE:-violetxi}/test_aime24_open-thoughts-8k-qwen3-4b-sft \
# data.ground_truth_key=answer

# # AIME 2025
# python verl/trainer/main_eval.py \
# data.path=${HF_NAMESPACE:-violetxi}/test_aime25_open-thoughts-8k-qwen3-4b-sft \
# data.ground_truth_key=answer

# # OMNI L7 Above
# python verl/trainer/main_eval.py \
# data.path=${HF_NAMESPACE:-violetxi}/test_omni-l7-above_open-thoughts-8k-qwen3-4b-sft \
# data.ground_truth_key=answer

# #### Base Qwen3-4B
# # AIME 2024
# python verl/trainer/main_eval.py \
# data.path=${HF_NAMESPACE:-violetxi}/test_aime24_qwen3-4b-base \
# data.ground_truth_key=answer

# # AIME 2025
# python verl/trainer/main_eval.py \
# data.path=${HF_NAMESPACE:-violetxi}/test_aime25_qwen3-4b-base \
# data.ground_truth_key=answer

# # OMNI L7 Above
# python verl/trainer/main_eval.py \
# data.path=${HF_NAMESPACE:-violetxi}/test_omni-l7-above_qwen3-4b-base \
# data.ground_truth_key=answer

#### Qwen3-1.7B-NoThinking
python verl/trainer/main_eval.py \
data.path=${HF_NAMESPACE:-violetxi}/omni-rule-l7-above-qwen1.7b-instruct-no-thinking \
data.ground_truth_key=answer