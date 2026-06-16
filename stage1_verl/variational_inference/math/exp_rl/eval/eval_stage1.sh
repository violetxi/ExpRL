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

### GRPO Evaluation ####
# # AIME 2025
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime25_stage1_grpo data.response_key=response

# # AIME 2026
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime26_stage1_grpo data.response_key=response

# # HMMT Nov 2025
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/hmmt-nov-2025_stage1_grpo data.response_key=response

# # IMO AnswerBench
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/imo-answerbench_stage1_grpo data.response_key=response

# # POPE
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/pope_hard_stage1_grpo data.response_key=response

### SFT Evaluation ####
# # AIME 2025
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime25_stage1_sft data.response_key=response

# # AIME 2026
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime26_stage1_sft data.response_key=response

# # HMMT Nov 2025
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/hmmt-nov-2025_stage1_sft data.response_key=response

# IMO AnswerBench
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/imo-answerbench_stage1_sft data.response_key=response

# # POPE
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/pope_hard_stage1_sft data.response_key=response

### Self-Distill Evaluation ####
# # AIME 2025
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime25_stage1_self_distill data.response_key=response

# # AIME 2026
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime26_stage1_self_distill data.response_key=response

# # HMMT Nov 2025
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/hmmt-nov-2025_stage1_self_distill data.response_key=response

# # IMO AnswerBench
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/imo-answerbench_stage1_self_distill data.response_key=response

# # POPE
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/pope_hard_stage1_self_distill data.response_key=response

### PR Dense Outcome Evaluation (none is ready yet) ####
# # AIME 2025
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime25_stage1_dense_outcome data.response_key=response

# # AIME 2026
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime26_stage1_dense_outcome data.response_key=response

# # HMMT Nov 2025
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/hmmt-nov-2025_stage1_dense_outcome data.response_key=response

# # IMO AnswerBench
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/imo-answerbench_stage1_dense_outcome data.response_key=response

# # POPE
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/pope_hard_stage1_dense_outcome data.response_key=response

### Qwen3-4B Instruct ###
# # AIME 2025
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime25_stage1_qwen4b_instruct data.response_key=response

# # AIME 2026
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime26_stage1_qwen4b_instruct data.response_key=response

# # HMMT Nov 2025
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/hmmt-nov-2025_stage1_qwen4b_instruct data.response_key=response

# # IMO AnswerBench
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/imo-answerbench_stage1_qwen4b_instruct data.response_key=response

# # POPE
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/pope_hard_stage1_qwen4b_instruct data.response_key=response

### Dense Process Evaluation ###
# # AIME 2025
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime25_stage1_pr_delta_process data.response_key=response

# # AIME 2026
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime26_stage1_pr_delta_process data.response_key=response

# # HMMT Nov 2025
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/hmmt-nov-2025_stage1_pr_delta_process data.response_key=response

# # IMO AnswerBench
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/imo-answerbench_stage1_pr_delta_process data.response_key=response

# POPE
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/pope_hard_stage1_pr_delta_process data.response_key=response


### InT Evaluation ###
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/int_stage1_qwen4b_instruct data.response_key=response

# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/int_stage1_sft data.response_key=response

# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/int_stage1_dense_outcome data.response_key=response

# (Not ready yet)

# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/int_stage1_grpo data.response_key=response

# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/int_stage1_pr_delta_process data.response_key=response

# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/int_stage1_self_distill data.response_key=response

#### PR-GPPO ####
# IMO AnswerBench
python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/imo-answerbench_stage1_pr_gppo data.response_key=response

# HMMT Nov 2025
python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/hmmt_stage1_pr_gppo data.response_key=response

# AIME 2025
python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime25_stage1_pr_gppo data.response_key=response

# AIME 2026
python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime26_stage1_pr_gppo data.response_key=response


#### PR-E ####
# IMO AnswerBench
python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/imo-answerbench_stage1_pr_e data.response_key=response

# HMMT Nov 2025
python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/hmmt_stage1_pr_e data.response_key=response

# AIME 2025
python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime25_stage1_pr_e data.response_key=response

# AIME 2026
python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime26_stage1_pr_e data.response_key=response