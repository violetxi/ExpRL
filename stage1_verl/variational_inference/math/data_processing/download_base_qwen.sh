#!/bin/bash
#SBATCH --job-name=download_base_qwen
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=4:00:00
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
# gemini filtered omni
python variational_inference/math/data_processing/base.py \
    --data_source ${HF_NAMESPACE:-violetxi}/omni-math-above-l7-rule-gemini-pro-filtered \
    --local_dir data/base/omni-math-above-l7-rule-gemini-pro-filtered \
    --problem_key=problem \
    --answer_key=answer \
    --solution_key=llm_solutions \
    --which_prompt=qwen3

# AIME datasets
python variational_inference/math/data_processing/base.py \
    --data_source HuggingFaceH4/aime_2024 \
    --local_dir data/base/aime-2024 \
    --problem_key=problem \
    --answer_key=answer \
    --which_prompt=qwen3

python variational_inference/math/data_processing/base.py \
    --data_source opencompass/AIME2025 \
    --local_dir data/base/aime-2025 \
    --problem_key=question \
    --answer_key=answer \
    --which_prompt=qwen3

# OlympiadBench dataset
python variational_inference/math/data_processing/base.py \
    --data_source Hothan/OlympiadBench \
    --local_dir data/base/olympiadbench \
    --problem_key question \
    --answer_key final_answer \
    --which_prompt=qwen3

# AMC dataset
python variational_inference/math/data_processing/base.py \
    --data_source AI-MO/aimo-validation-amc \
    --local_dir data/base/amc \
    --problem_key problem \
    --answer_key answer \
    --which_prompt=qwen3

# # AceReason dataset
# python variational_inference/math/data_processing/base.py \
#     --data_source CohenQu/AceReason-Math-Qwen3-4B \
#     --local_dir data/base/ace_reason \
#     --problem_key=problem \
#     --answer_key=answer \
#     --which_prompt=qwen3

# python variational_inference/math/data_processing/base.py \
#     --data_source ${HF_NAMESPACE:-violetxi}/omni-math-difficulty-2 \
#     --local_dir data/base/omni-math-difficulty-2 \
#     --problem_key=problem \
#     --answer_key=answer \
#     --solution_key=solution \
#     --which_prompt=qwen3

# python variational_inference/math/data_processing/base.py \
#     --data_source ${HF_NAMESPACE:-violetxi}/omni-math-difficulty-5 \
#     --local_dir data/base/omni-math-difficulty-5 \
#     --problem_key=problem \
#     --answer_key=answer \
#     --solution_key=solution \
#     --which_prompt=qwen3

# python variational_inference/math/data_processing/base.py \
#     --data_source ${HF_NAMESPACE:-violetxi}/omni-math-difficulty-8 \
#     --local_dir data/base/omni-math-difficulty-8 \
#     --problem_key=problem \
#     --answer_key=answer \
#     --solution_key=solution \
#     --which_prompt=qwen3

# # POPE hard dataset
# python variational_inference/math/data_processing/base.py \
#     --data_source CohenQu/POPE-hard-dataset-Qwen3-4B-Instruct-32k-128-filtered \
#     --local_dir data/base/pope_hard_filtered \
#     --problem_key=problem \
#     --answer_key=answer \
#     --which_prompt=qwen3