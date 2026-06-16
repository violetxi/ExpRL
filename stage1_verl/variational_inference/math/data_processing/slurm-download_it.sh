#!/bin/bash
#SBATCH --job-name=download_data
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
set -x

# ###### Math training datasets ####
# #### POPE FULL DATASET ####
# python variational_inference/math/data_processing/instruct.py \
#     --data_source CohenQu/POPE-hard-dataset-Qwen3-4B-Instruct-32k-128-filtered-iter3-gemini-success \
#     --local_dir data/instruct/pope_full \
#     --problem_key=problem \
#     --answer_key=answer \
#     --solution_key=gemini_solution \
#     --which_prompt=qwen3

# #### INT Dataset ####
# python variational_inference/math/data_processing/instruct.py \
#     --data_source d1shs0ap/unified-hard-set-with-student-solutions-guided-rl \
#     --local_dir data/instruct/int_dataset \
#     --problem_key=problem \
#     --answer_key=answer \
#     --solution_key=solution \
#     --which_prompt=qwen3

# ###### Math testing datasets ####
# python variational_inference/math/data_processing/instruct.py \
#     --data_source Hwilner/imo-answerbench \
#     --local_dir data/instruct/imo-answerbench \
#     --problem_key=problem \
#     --answer_key=answer \
#     --which_prompt=qwen3

# python variational_inference/math/data_processing/instruct.py \
#     --data_source MathArena/aime_2026 \
#     --local_dir data/instruct/aime-2026 \
#     --problem_key=problem \
#     --answer_key=answer \
#     --which_prompt=qwen3

# python variational_inference/math/data_processing/instruct.py \
#     --data_source MathArena/hmmt_nov_2025 \
#     --local_dir data/instruct/hmmt-nov-2025 \
#     --problem_key=problem \
#     --answer_key=answer \
#     --which_prompt=qwen3

# python variational_inference/math/data_processing/instruct.py \
#     --data_source opencompass/AIME2025 \
#     --local_dir data/instruct/aime-2025 \
#     --problem_key=question \
#     --answer_key=answer \
#     --which_prompt=qwen3



####### Training data for coding and science ####
# Hydrate LCB v6 oracle (HF) with the full reward_model.ground_truth from local source.
# Output is the ~4GB train.parquet that verl trains against (preserves the rule-based
# outcome reward path while adding extra_info.solution + extra_info.solution_code_only).
# python -m scripts.prepare_livecodebench_gpqa \
#     --base_dir data/instruct \
#     --tasks livecodebench_v6

# python scripts/hydrate_lcb_with_ground_truth.py \
#     --hf_repo ${HF_NAMESPACE:-violetxi}/livecodebench_v6_gemini-3-pro \
#     --hf_filename train.parquet \
#     --source data/instruct/livecodebench_v6/test.parquet \
#     --output data/instruct/livecodebench_v6_with_oracle/train.parquet

#### SciKnowEval OE v1 ####
# download processed data oe_v1_physics
python variational_inference/math/data_processing/instruct.py \
    --data_source ${HF_NAMESPACE:-violetxi}/sciknoweval_oe_v1_physics_gemini-3-pro \
    --prompt_key=prompt \
    --local_dir data/instruct/sciknoweval_oe_v1_physics \
    --which_prompt=qwen3


#### SciKnowEval MCQ L3 (loose subset) ####
# Pre-built with build_oracle_dataset.py (n=4 Gemini-3-pro samples, letter-verified
# against ground_truth, ~92% verified) and pushed to HF. 2400 rows.
python variational_inference/math/data_processing/instruct.py \
    --data_source ${HF_NAMESPACE:-violetxi}/sciknoweval_mcq_l3_loose_gemini-3-pro \
    --prompt_key=prompt \
    --local_dir data/instruct/sciknoweval_mcq_l3_loose \
    --which_prompt=qwen3


##### Test data for coding and science ####
##### Coding testing datasets ####
python -m scripts.prepare_livecodebench_gpqa \
    --base_dir data/instruct \
    --tasks livecodebench_v5,gpqa_diamond