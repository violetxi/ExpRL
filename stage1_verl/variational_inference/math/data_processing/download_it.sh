#!/bin/bash
#SBATCH --job-name=download_it
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
#### POPE FULL DATASET ####
python variational_inference/math/data_processing/instruct.py \
    --data_source CohenQu/POPE-hard-dataset-Qwen3-4B-Instruct-32k-128-filtered-iter3-gemini-success \
    --local_dir data/instruct/pope_full \
    --problem_key=problem \
    --answer_key=answer \
    --solution_key=gemini_solution \
    --which_prompt=qwen3

#### INT Dataset ####
python variational_inference/math/data_processing/instruct.py \
    --data_source d1shs0ap/unified-hard-set-with-student-solutions-guided-rl \
    --local_dir data/instruct/int_dataset \
    --problem_key=problem \
    --answer_key=answer \
    --solution_key=solution \
    --which_prompt=qwen3

# # Set2 from ICML26
# python variational_inference/math/data_processing/instruct.py \
#     --data_source d1shs0ap/unified-hard-set-with-student-solutions-guided-rl \
#     --local_dir data/instruct/icml26_set2 \
#     --problem_key=problem \
#     --answer_key=answer \
#     --solution_key=solution \
#     --which_prompt=qwen3


# python variational_inference/math/data_processing/instruct.py \
#     --data_source ${HF_NAMESPACE:-violetxi}/omni-math-above-l7-rule-gemini-pro-filtered \
#     --local_dir data/instruct/omni-math-above-l7-rule-gemini-pro-filtered \
#     --problem_key=problem \
#     --answer_key=answer \
#     --solution_key=llm_solutions \
#     --which_prompt=qwen3


# python variational_inference/math/data_processing/instruct.py \
#     --data_source ${HF_NAMESPACE:-violetxi}/omni-math-above-l6 \
#     --local_dir data/instruct/omni-math-above-l6 \
#     --problem_key=problem \
#     --answer_key=answer \
#     --solution_key=solution \
#     --which_prompt=qwen3

# python variational_inference/math/data_processing/instruct.py \
#     --data_source ${HF_NAMESPACE:-violetxi}/omni-math-above-l7 \
#     --local_dir data/instruct/omni-math-above-l7 \
#     --problem_key=problem \
#     --answer_key=answer \
#     --solution_key=solution \
#     --which_prompt=qwen3


# # this is for evaluating models trained with hints on the hard 
# python variational_inference/math/data_processing/instruct.py \
#     --data_source ${HF_NAMESPACE:-violetxi}/pope-hard-w-guide-gemini-solution \
#     --local_dir data/instruct/pope_hard_w_hint_test \
#     --problem_key=problem \
#     --answer_key=answer \
#     --solution_key=gemini_solution \
#     --which_prompt=qwen3

# python variational_inference/math/data_processing/download_hints.py \
#     --data_source ${HF_NAMESPACE:-violetxi}/pope-hard-w-guide-gemini-solution-hint \
#     --local_dir data/instruct/pope_hard_w_guide_gemini_solution_hint \
#     --problem_key=problem \
#     --answer_key=answer \
#     --solution_key=gemini_solution

# python variational_inference/math/data_processing/instruct.py \
#     --data_source ${HF_NAMESPACE:-violetxi}/pope-hard-w-guide-gemini-solution \
#     --local_dir data/instruct/pope_hard_w_guide_gemini_solution \
#     --problem_key=problem \
#     --answer_key=answer \
#     --solution_key=solution \
#     --which_prompt=qwen3

# # ### SET 1 ###
# python variational_inference/math/data_processing/instruct.py \
#     --data_source ${HF_NAMESPACE:-violetxi}/omni-filtered-by-cohen \
#     --local_dir data/instruct/omni-filtered-by-cohen \
#     --problem_key=problem \
#     --answer_key=answer \
#     --solution_key=solution \
#     --which_prompt=qwen3

# #### model's own thinking+sol as 'solution' ####
# python variational_inference/math/data_processing/instruct.py \
#     --data_source ${HF_NAMESPACE:-violetxi}/qwen4b-thinking-omni-l1_2-score \
#     --solution_key=responses \
#     --local_dir data/instruct/qwen4b-thinking-omni-l1_2-score \
#     --which_prompt=qwen3

# python variational_inference/math/data_processing/instruct.py \
#     --data_source ${HF_NAMESPACE:-violetxi}/qwen4b-thinking-omni-l2_3-score \
#     --solution_key=responses \
#     --local_dir data/instruct/qwen4b-thinking-omni-l2_3-score \
#     --which_prompt=qwen3

# python variational_inference/math/data_processing/instruct.py \
#     --data_source ${HF_NAMESPACE:-violetxi}/qwen4b-thinking-omni-l4-score \
#     --solution_key=responses \
#     --local_dir data/instruct/qwen4b-thinking-omni-l4-score \
#     --which_prompt=qwen3
# #### model's own thinking+sol as 'solution' ####

# python variational_inference/math/data_processing/instruct.py \
#     --data_source ${HF_NAMESPACE:-violetxi}/qwen4b-thinking-omni-l1_2-score \
#     --local_dir data/instruct/qwen4b-thinking-omni-l1_2-score \
#     --which_prompt=qwen3

# python variational_inference/math/data_processing/instruct.py \
#     --data_source ${HF_NAMESPACE:-violetxi}/qwen8b-rewrite-omni-l1_2-score \
#     --local_dir data/instruct/qwen8b-rewrite-omni-l1_2 \
#     --solution_key=responses \
#     --which_prompt=qwen3

# python variational_inference/math/data_processing/instruct.py \
#     --data_source ${HF_NAMESPACE:-violetxi}/test-qwen8b-rewrite-omni-l1_2-score \
#     --local_dir data/instruct/test-qwen8b-rewrite-omni-l1_2 \
#     --solution_key=responses \
#     --which_prompt=qwen3

# python variational_inference/math/data_processing/instruct.py \
#     --data_source ${HF_NAMESPACE:-violetxi}/test-qwen8b-rewrite-omni-l2_3-score \
#     --local_dir data/instruct/test-qwen8b-rewrite-omni-l2_3 \
#     --which_prompt=qwen3

# python variational_inference/math/data_processing/instruct.py \
#     --data_source ${HF_NAMESPACE:-violetxi}/qwen8b-rewrite-omni-l5 \
#     --local_dir data/instruct/qwen8b-rewrite-omni-l5 \
#     --which_prompt=qwen3

# python variational_inference/math/data_processing/instruct.py \
#     --data_source ${HF_NAMESPACE:-violetxi}/qwen8b-rewrite-omni-l8 \
#     --local_dir data/instruct/qwen8b-rewrite-omni-l8 \
#     --which_prompt=qwen3

# # AceReason dataset
# python variational_inference/math/data_processing/instruct.py \
#     --data_source CohenQu/AceReason-Math-Qwen3-4B \
#     --local_dir data/instruct/ace_reason \
#     --problem_key=problem \
#     --answer_key=answer \
#     --which_prompt=qwen3

# python variational_inference/math/data_processing/instruct.py \
#     --data_source ${HF_NAMESPACE:-violetxi}/qwen4b-rewrite-omni-l2 \
#     --local_dir data/instruct/qwen4b-rewrite-omni-l2 \
#     --problem_key=extra_info \
#     --solution_key=solution \
#     --which_prompt=qwen3


# python variational_inference/math/data_processing/instruct.py \
#     --data_source ${HF_NAMESPACE:-violetxi}/omni-math-difficulty-1_2 \
#     --local_dir data/instruct/omni-math-difficulty-1_2 \
#     --problem_key=problem \
#     --answer_key=answer \
#     --solution_key=solution \
#     --which_prompt=qwen3

# python variational_inference/math/data_processing/instruct.py \
#     --data_source ${HF_NAMESPACE:-violetxi}/omni-math-difficulty-2_3 \
#     --local_dir data/instruct/omni-math-difficulty-2_3 \
#     --problem_key=problem \
#     --answer_key=answer \
#     --solution_key=solution \
#     --which_prompt=qwen3
    
# python variational_inference/math/data_processing/instruct.py \
#     --data_source ${HF_NAMESPACE:-violetxi}/test-qwen8b-rewrite-omni-l2 \
#     --local_dir data/instruct/qwen8b-rewrite-omni-l2 \
#     --problem_key=extra_info \
#     --solution_key=solution \
#     --which_prompt=qwen3

# python variational_inference/math/data_processing/instruct.py \
#     --data_source ${HF_NAMESPACE:-violetxi}/test-qwen4b-rewrite-omni-l8 \
#     --local_dir data/instruct/qwen4b-rewrite-omni-l8 \
#     --problem_key=extra_info \
#     --solution_key=solution \
#     --which_prompt=qwen3

# python variational_inference/math/data_processing/instruct.py \
#     --data_source ${HF_NAMESPACE:-violetxi}/test-qwen8b-rewrite-omni-l8 \
#     --local_dir data/instruct/qwen8b-rewrite-omni-l8 \
#     --problem_key=extra_info \
#     --solution_key=solution \
#     --which_prompt=qwen3

# python variational_inference/math/data_processing/instruct.py \
#     --data_source ${HF_NAMESPACE:-violetxi}/omni-math-difficulty-2 \
#     --local_dir data/instruct/omni-math-difficulty-2 \
#     --problem_key=problem \
#     --answer_key=answer \
#     --solution_key=solution \
#     --which_prompt=qwen3

# python variational_inference/math/data_processing/instruct.py \
#     --data_source ${HF_NAMESPACE:-violetxi}/omni-math-difficulty-5 \
#     --local_dir data/instruct/omni-math-difficulty-5 \
#     --problem_key=problem \
#     --answer_key=answer \
#     --solution_key=solution \
#     --which_prompt=qwen3

# python variational_inference/math/data_processing/instruct.py \
#     --data_source ${HF_NAMESPACE:-violetxi}/omni-math-difficulty-8 \
#     --local_dir data/instruct/omni-math-difficulty-8 \
#     --problem_key=problem \
#     --answer_key=answer \
#     --solution_key=solution \
#     --which_prompt=qwen3

# python variational_inference/math/data_processing/instruct.py \
#     --data_source ${HF_NAMESPACE:-violetxi}/omni-math-difficulty-4 \
#     --local_dir data/instruct/omni-math-difficulty-4 \
#     --problem_key=problem \
#     --answer_key=answer \
#     --solution_key=solution \
#     --which_prompt=qwen3

# python variational_inference/math/data_processing/instruct.py \
#     --data_source HerrHruby/acemath_rl_4b_inst_hard \
#     --local_dir data/instruct/ace_rl_4b_inst_hard \
#     --problem_key=problem \
#     --answer_key=answer \
#     --which_prompt=qwen3

# python variational_inference/math/data_processing/instruct.py \
#     --data_source ${HF_NAMESPACE:-violetxi}/omni-math-difficulty-2_5 \
#     --local_dir data/instruct/omni-math-difficulty-2_5 \
#     --problem_key=problem \
#     --answer_key=answer \
#     --solution_key=solution \
#     --which_prompt=qwen3

# python variational_inference/math/data_processing/instruct.py \
#     --data_source ${HF_NAMESPACE:-violetxi}/omni-math-difficulty-4_6 \
#     --local_dir data/instruct/omni-math-difficulty-4_6 \
#     --problem_key=problem \
#     --answer_key=answer \
#     --solution_key=solution \
#     --which_prompt=qwen3

# python variational_inference/math/data_processing/instruct.py \
#     --data_source ${HF_NAMESPACE:-violetxi}/omni-math-difficulty-5_7 \
#     --local_dir data/instruct/omni-math-difficulty-5_7 \
#     --problem_key=problem \
#     --answer_key=answer \
#     --solution_key=solution \
#     --which_prompt=qwen3

# # POPE hard dataset
# python variational_inference/math/data_processing/instruct.py \
#     --data_source CohenQu/POPE-hard-dataset-Qwen3-4B-Instruct-32k-128-filtered \
#     --local_dir data/instruct/pope_hard_filtered \
#     --problem_key=problem \
#     --answer_key=answer \
#     --which_prompt=qwen3

# python variational_inference/math/data_processing/instruct.py \
#     --data_source HuggingFaceH4/aime_2024 \
#     --local_dir data/instruct/aime-2024 \
#     --problem_key=problem \
#     --answer_key=answer \
#     --which_prompt=qwen3

python variational_inference/math/data_processing/instruct.py \
    --data_source opencompass/AIME2025 \
    --local_dir data/instruct/aime-2025 \
    --problem_key=question \
    --answer_key=answer \
    --which_prompt=qwen3

# # # python variational_inference/math/data_processing/instruct.py \
# # #     --data_source agentica-org/DeepScaleR-Preview-Dataset \
# # #     --local_dir data/instruct/deepscale \
# # #     --problem_key=problem \
# # #     --answer_key=answer \
# # #     --solution_key=solution \
# # #     --which_prompt=qwen3

# # # python variational_inference/math/data_processing/instruct.py \
# # #     --data_source HuggingFaceH4/MATH-500 \
# # #     --local_dir data/instruct/math \
# # #     --problem_key=problem \
# # #     --answer_key=answer \
# # #     --solution_key=solution \
# # #     --which_prompt=qwen3

# python variational_inference/math/data_processing/instruct.py \
#     --data_source Hothan/OlympiadBench \
#     --local_dir data/instruct/olympiadbench \
#     --problem_key=question \
#     --answer_key=final_answer \
#     --which_prompt=qwen3

# python variational_inference/math/data_processing/instruct.py \
#     --data_source AI-MO/aimo-validation-amc \
#     --local_dir data/instruct/amc \
#     --problem_key=problem \
#     --answer_key=answer \
#     --solution_key=solution \
#     --which_prompt=qwen3

#### More evaluation datasets ####
python variational_inference/math/data_processing/instruct.py \
    --data_source Hwilner/imo-answerbench \
    --local_dir data/instruct/imo-answerbench \
    --problem_key=problem \
    --answer_key=answer \
    --which_prompt=qwen3

python variational_inference/math/data_processing/instruct.py \
    --data_source MathArena/aime_2026 \
    --local_dir data/instruct/aime-2026 \
    --problem_key=problem \
    --answer_key=answer \
    --which_prompt=qwen3

python variational_inference/math/data_processing/instruct.py \
    --data_source MathArena/hmmt_nov_2025 \
    --local_dir data/instruct/hmmt-nov-2025 \
    --problem_key=problem \
    --answer_key=answer \
    --which_prompt=qwen3

#### OlympiadBench OE physics (English / text-only / open-ended) ####
# Custom builder (not instruct.py): selects the OE_TO_physics_en_COMP config and
# carries the physics-grading hints (unit / error / is_multiple_answer) into
# extra_info for the Gemini judge (verl/utils/reward_score/olympiadbench_oe_judge.py).
# Writes data/instruct/olympiadbench_oe_physics/train.parquet (local only).
python -m scripts.build_and_push_olympiadbench_oe_physics --no_push