#!/bin/bash
#SBATCH --job-name=download_sft_thoughts
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

# envs
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate verl

cd "${EXPRL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && while [ ! -f setup.py ] && [ "$PWD" != / ]; do cd ..; done && pwd)}"
# ###### OpenThoughts dataset ######
# python variational_inference/math/data_processing/sft_base.py \
#     --data_path ${HF_NAMESPACE:-violetxi}/OpenThoughts-16k \
#     --local_dir data/sft_base/openthoughts_16k \
#     --which_prompt qwen3

python variational_inference/math/data_processing/sft_base.py \
    --data_path ${HF_NAMESPACE:-violetxi}/OpenThoughts-8k \
    --local_dir data/sft_base/openthoughts_8k \
    --which_prompt qwen3

# ##### rewrite only
# python variational_inference/math/data_processing/final_sft_thoughts.py \
#     --rewrite_path ${HF_NAMESPACE:-violetxi}/qwen8b-rewrite-omni-l1_2-score \
#     --thinking_path ${HF_NAMESPACE:-violetxi}/qwen4b-thinking-omni-l1_2-score \
#     --local_dir data/sft_thoughts/omni-math-difficulty-1_2 \
#     --data_source ${HF_NAMESPACE:-violetxi}/qwen8b-rewrite-omni-l1_2 \
#     --which_prompt basic


# python variational_inference/math/data_processing/final_sft_thoughts.py \
#     --rewrite_path ${HF_NAMESPACE:-violetxi}/qwen8b-rewrite-omni-l2_3-score \
#     --thinking_path ${HF_NAMESPACE:-violetxi}/qwen4b-thinking-omni-l2_3-score \
#     --local_dir data/sft_thoughts/omni-math-difficulty-2_3 \
#     --data_source ${HF_NAMESPACE:-violetxi}/qwen8b-rewrite-omni-l2_3 \
#     --which_prompt basic

# python variational_inference/math/data_processing/final_sft_thoughts.py \
#     --rewrite_path ${HF_NAMESPACE:-violetxi}/qwen8b-rewrite-omni-l4-score \
#     --thinking_path ${HF_NAMESPACE:-violetxi}/qwen4b-thinking-omni-l4-score \
#     --local_dir data/sft_thoughts/omni-math-difficulty-4 \
#     --data_source ${HF_NAMESPACE:-violetxi}/qwen8b-rewrite-omni-l4 \
#     --which_prompt basic

# ##### rewrite and human solution
# python variational_inference/math/data_processing/final_sft_thoughts.py \
#     --rewrite_path ${HF_NAMESPACE:-violetxi}/qwen8b-rewrite-omni-l1_2-score \
#     --thinking_path ${HF_NAMESPACE:-violetxi}/qwen4b-thinking-omni-l1_2-score \
#     --local_dir data/sft_thoughts/omni-math-difficulty-1_2_human \
#     --data_source ${HF_NAMESPACE:-violetxi}/qwen8b-rewrite-omni-l1_2 \
#     --mix_human_solution \
#     --which_prompt basic


# python variational_inference/math/data_processing/final_sft_thoughts.py \
#     --rewrite_path ${HF_NAMESPACE:-violetxi}/qwen8b-rewrite-omni-l2_3-score \
#     --thinking_path ${HF_NAMESPACE:-violetxi}/qwen4b-thinking-omni-l2_3-score \
#     --local_dir data/sft_thoughts/omni-math-difficulty-2_3_human \
#     --data_source ${HF_NAMESPACE:-violetxi}/qwen8b-rewrite-omni-l2_3 \
#     --mix_human_solution \
#     --which_prompt basic

# python variational_inference/math/data_processing/final_sft_thoughts.py \
#     --rewrite_path ${HF_NAMESPACE:-violetxi}/qwen8b-rewrite-omni-l4-score \
#     --thinking_path ${HF_NAMESPACE:-violetxi}/qwen4b-thinking-omni-l4-score \
#     --local_dir data/sft_thoughts/omni-math-difficulty-4_human \
#     --data_source ${HF_NAMESPACE:-violetxi}/qwen8b-rewrite-omni-l4 \
#     --mix_human_solution \
#     --which_prompt basic