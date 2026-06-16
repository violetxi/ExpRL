#!/bin/bash
#SBATCH --job-name=eval_cohen
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=16G
#SBATCH --time=24:00:00
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
# ###### Evaluation on Cohen ######

# echo "Evaluating Qwen3-4B"
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/test_omni_filtered_by_cohen_qwen3-4b


# echo "Evaluating GRPO Qwen3-4B with PR Smooth Delta Cont 30"
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/test_omni_filtered_by_cohen_cohen_qwen3-4b_grpo_pr_smooth_delta_cont_30

# echo "Evaluating GRPO Qwen3-4B with PR Smooth Delta Cont 60"
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/test_omni_filtered_by_cohen_cohen_qwen3-4b_grpo_pr_smooth_delta_cont_60

# echo "Evaluating GRPO Qwen3-4B with PR Smooth Delta Cont 90"
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/test_omni_filtered_by_cohen_cohen_qwen3-4b_grpo_pr_smooth_delta_cont_90


# echo "Evaluating GRPO Qwen3-4B with Outcome Cont 30"
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/test_omni_filtered_by_cohen_cohen_qwen3-4b_grpo_outcome_smooth_cont_30

# echo "Evaluating GRPO Qwen3-4B with Outcome Cont 90"
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/test_omni_filtered_by_cohen_cohen_qwen3-4b_grpo_outcome_smooth_cont_90

# echo "Evaluating GRPO Qwen3-4B with Outcome Cont 60"
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/test_omni_filtered_by_cohen_cohen_qwen3-4b_grpo_outcome_smooth_cont_60

# echo "Evaluating GRPO Qwen3-4B with PR Smooth Delta Cont 90 Continued"
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/test_omni_filtered_by_cohen_qwen3-4b_grpo_pr_smooth_delta_cont_90_continued

# echo "Evaluating GRPO Qwen3-4B with PR Smooth Delta Step 90"
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/test_omni_filtered_by_cohen_qwen3-4b_process_smooth_delta_global_step_90

# echo "Evaluating GRPO Qwen3-4B with Outcome Smooth Step 90"
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/test_omni_filtered_by_cohen_qwen3-4b_outcome_smooth_delta_global_step_90

# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/judge-calibration_set1_16k_qwen3-4b data.response_key=response

#### PR OnPolicy ####
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/set1_16k_onpolicy_distill_lr1e-6 data.response_key=response

#### Set 1 self-distill (32k) ####
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/set1_32k_onpolicy_distill_lr1e-6 data.response_key=response

##### SET 2 cont ####
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/set1_16k_self_distill_set2_cont data.response_key=response

# python verl/trainer/main_eval.py data.path${HF_NAMESPACE:-violetxi}/set2_16k_self_distill_set2_cont data.response_key=response

# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime24_16k_self_distill_set2_cont data.response_key=response

# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime25_16k_self_distill_set2_cont data.response_key=response

# ##### SET 1 cont ####
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/set1_16k_self_distill_set1_cont data.response_key=response

# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime24_16k_self_distill_set1_cont data.response_key=response

# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime25_16k_self_distill_set1_cont data.response_key=response

# ##### N=32 GRPO ####
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/set1_16k_cont_grpo_n32 data.response_key=response

# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime24_16k_cont_grpo_n32 data.response_key=response

# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime25_16k_cont_grpo_n32 data.response_key=response


##### N=32 PR-Delta ####
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/set1_16k_cont_pr_delta_n32 data.response_key=response

# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime24_16k_cont_pr_delta_n32 data.response_key=response

# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime25_16k_cont_pr_delta_n32 data.response_key=response

# #### Stage 2 PR-Delta Cont ####
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/set1_16k_cont_pr_delta_n32_cont data.response_key=response

python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime24_16k_cont_pr_delta_n32_cont data.response_key=response

# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime25_16k_cont_pr_delta_n32_cont data.response_key=response

# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime26_16k_cont_pr_delta_n32_cont data.response_key=response

# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/hmmt-nov-2025_16k_cont_pr_delta_n32_cont data.response_key=response

# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/imo-answerbench_16k_cont_pr_delta_n32_cont data.response_key=response

# #### N=32 Self Distill ####
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/set1_16k_self_distill_set1_cont_n32 data.response_key=response

# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime24_16k_self_distill_set1_cont_n32 data.response_key=response

# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime25_16k_self_distill_set1_cont_n32 data.response_key=response

# #### Stage 2 GRPO Cont ####
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/set1_16k_cont_grpo_n32_cont data.response_key=response

# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime24_16k_cont_grpo_n32_cont data.response_key=response

# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime25_16k_cont_grpo_n32_cont data.response_key=response

# ##### MT - Set 1 GRPO ####
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/set1_16k_qwen3-4b-w-feedback_user data.response_key=response_turn_2

# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/set1_16k_qwen3-4b-w-feedback_system data.response_key=feedback

##### AIME 2026 ####
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime26_16k_cont_self_distill_lr1e-7_n32_cont

# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime26_16k_cont_grpo_n32_cont data.response_key=response

# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime26_16k_cont_self_distill_lr1e-7_n32_cont data.response_key=response

##### HMMT Nov 2025 ####
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/hmmt-nov-2025_16k_cont_self_distill_lr1e-7_n32_cont

# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/hmmt-nov-2025_16k_cont_grpo_n32_cont data.response_key=response

# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/hmmt-nov-2025_16k_cont_self_distill_lr1e-7_n32_cont data.response_key=response

##### IMO AnswerBench ####
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/imo-answerbench_16k_cont_self_distill_lr1e-7_n32_cont

# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/imo-answerbench_16k_cont_grpo_n32_cont data.response_key=response
