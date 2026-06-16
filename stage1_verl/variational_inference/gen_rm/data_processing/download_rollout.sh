#!/bin/bash
#SBATCH --job-name=download_rollout
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
# #### rollout only
# python variational_inference/gen_rm/data_processing/continue_rollout.py \
# --data_source ${HF_NAMESPACE:-violetxi}/qwen4b-no-thinking-omni-l5-score-steps \
# --local_dir data/process_reward/qwen4b_no_thinking_omni_l5_steps \
# --which_prompt rollout

# python variational_inference/gen_rm/data_processing/continue_rollout.py \
# --data_source ${HF_NAMESPACE:-violetxi}/qwen4b-no-thinking-omni-l2-score-steps \
# --local_dir data/process_reward/qwen4b_no_thinking_omni_l2_steps \
# --which_prompt rollout

###### rollout on Qwen3-4B-Instruct-2507 ######
# # OMNI raw L7
# python variational_inference/gen_rm/data_processing/continue_rollout.py \
# --data_source ${HF_NAMESPACE:-violetxi}/qwen4b-instruct-2507-omni-l7-score-steps \
# --local_dir data/process_reward/qwen4b-instruct-2507-omni-l7-steps \
# --which_prompt rollout

# # OMNI rule L7
# python variational_inference/gen_rm/data_processing/continue_rollout.py \
# --data_source ${HF_NAMESPACE:-violetxi}/omni-rule-l7-above-gemini-pro-filtered_qwen3-4b-instruct-2507-steps \
# --local_dir data/process_reward/omni-rule-l7-above-gemini-pro-filtered_qwen3-4b-instruct-2507-steps \
# --which_prompt rollout

###### ICML Set1 ######
python variational_inference/gen_rm/data_processing/continue_rollout.py \
--data_source ${HF_NAMESPACE:-violetxi}/judge-calibration_set1_16k_qwen3-4b-score-steps \
--local_dir data/process_reward/judge-calibration_set1_16k_qwen3-4b-score-steps \
--which_prompt rollout