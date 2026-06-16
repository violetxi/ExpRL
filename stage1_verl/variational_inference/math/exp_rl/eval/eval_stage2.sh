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
### POPE Hard Evaluation ###
# GRPO
python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/pope_hard_stage2_grpo data.response_key=response

# SFT
python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/pope_hard_stage2_sft data.response_key=response

# Self Distill
python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/pope_hard_stage2_self_distill data.response_key=response

# Dense Outcome
python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/pope_hard_stage2_dense_outcome data.response_key=response

# Dense Process
python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/pope_hard_stage2_pr_delta_process data.response_key=response

### IMO AnswerBench Evaluation ###
# GRPO
python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/imo-answerbench_stage2_grpo data.response_key=response

# SFT
python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/imo-answerbench_stage2_sft data.response_key=response

# Self Distill
python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/imo-answerbench_stage2_self_distill data.response_key=response

# Dense Outcome
python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/imo-answerbench_stage2_dense_outcome data.response_key=response

# Dense Process
python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/imo-answerbench_stage2_pr_delta_process data.response_key=response

### HMMT Nov 2025 Evaluation ###
# GRPO
python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/hmmt-nov-2025_stage2_grpo data.response_key=response

# SFT
python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/hmmt-nov-2025_stage2_sft data.response_key=response

# Self Distill
python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/hmmt-nov-2025_stage2_self_distill data.response_key=response

# Dense Outcome
python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/hmmt-nov-2025_stage2_dense_outcome data.response_key=response

# Dense Process
python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/hmmt-nov-2025_stage2_pr_delta_process data.response_key=response


# ### AIME 2025 Evaluation ####
# # GRPO
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime25_stage2_grpo data.response_key=response

# # SFT
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime25_stage2_sft data.response_key=response

# # Self Distill
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime25_stage2_self_distill data.response_key=response

# # Dense Outcome
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime25_stage2_dense_outcome data.response_key=response

# # Dense Process
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime25_stage2_pr_delta_process data.response_key=response

### AIME 2026 Evaluation ###
# # GRPO
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime26_stage2_grpo data.response_key=response

# # SFT
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime26_stage2_sft data.response_key=response

# # Self Distill
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime26_stage2_self_distill data.response_key=response

# # Dense Outcome
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime26_stage2_dense_outcome data.response_key=response

# # Dense Process
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime26_stage2_pr_delta_process data.response_key=response

