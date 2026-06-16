#!/bin/bash
#SBATCH --job-name=gen_rollout
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
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

# test_path=data/process_reward/qwen4b_no_thinking_omni_l5_steps/train.parquet
# python3 -m verl.trainer.main_generation \
# --config-path=config --config-name=generation \
# model.path=qwen/Qwen3-4B \
# model.revision=main \
# trainer.nnodes=1 \
# trainer.n_gpus_per_node=8 \
# data.path=$test_path \
# data.prompt_key=prompt \
# data.n_samples=64 \
# data.batch_size=2048 \
# data.use_chat_template=True \
# data.enable_thinking=False \
# data.output_path=${HF_NAMESPACE:-violetxi}/process_rollout-qwen4b-omni-l5 \
# rollout.prompt_length=16384 \
# rollout.response_length=16384 \
# rollout.temperature=0.6 \
# rollout.tensor_model_parallel_size=2 \
# rollout.gpu_memory_utilization=0.85 \
# rollout.max_num_batched_tokens=32768 \
# rollout.max_num_seqs=2048

# test_path=data/process_reward/qwen4b_no_thinking_omni_l2_steps/train.parquet
# python3 -m verl.trainer.main_generation \
# --config-path=config --config-name=generation \
# model.path=qwen/Qwen3-4B \
# model.revision=main \
# trainer.nnodes=1 \
# trainer.n_gpus_per_node=8 \
# data.path=$test_path \
# data.prompt_key=prompt \
# data.n_samples=16 \
# data.batch_size=2048 \
# data.use_chat_template=True \
# data.enable_thinking=False \
# data.output_path=${HF_NAMESPACE:-violetxi}/process_rollout-qwen4b-omni-l2 \
# rollout.prompt_length=16384 \
# rollout.response_length=16384 \
# rollout.temperature=0.6 \
# rollout.tensor_model_parallel_size=2 \
# rollout.gpu_memory_utilization=0.85 \
# rollout.max_num_batched_tokens=32768 \
# rollout.max_num_seqs=2048

# ###### rollout on ICML Set1 ######
test_path=data/process_reward/judge-calibration_set1_16k_qwen3-4b-score-steps/train.parquet
python3 -m verl.trainer.main_generation \
--config-path=config --config-name=generation \
model.path=qwen/Qwen3-4B \
model.revision=main \
trainer.nnodes=1 \
trainer.n_gpus_per_node=8 \
data.path=$test_path \
data.prompt_key=prompt \
data.n_samples=32 \
data.batch_size=2048 \
data.use_chat_template=True \
data.enable_thinking=False \
data.output_path=${HF_NAMESPACE:-violetxi}/process_rollout-judge-calibration_set1_16k_qwen3-4b \
rollout.prompt_length=16384 \
rollout.response_length=16384 \
rollout.temperature=0.8 \
rollout.tensor_model_parallel_size=1 \
rollout.gpu_memory_utilization=0.85 \
rollout.max_num_batched_tokens=32768 \
rollout.max_num_seqs=2048

# ###### rollout on Qwen3-4B-Instruct-2507 ######
# test_path=data/process_reward/qwen4b-instruct-2507-omni-l7-steps/train.parquet
# python3 -m verl.trainer.main_generation \
# --config-path=config --config-name=generation \
# model.path=qwen/Qwen3-4B-Instruct-2507 \
# model.revision=main \
# trainer.nnodes=1 \
# trainer.n_gpus_per_node=8 \
# data.path=$test_path \
# data.prompt_key=prompt \
# data.n_samples=32 \
# data.batch_size=2048 \
# data.use_chat_template=True \
# data.enable_thinking=False \
# data.output_path=${HF_NAMESPACE:-violetxi}/process_rollout-qwen4b-instruct-2507-omni-l7 \
# rollout.prompt_length=16384 \
# rollout.response_length=16384 \
# rollout.temperature=0.8 \
# rollout.tensor_model_parallel_size=1 \
# rollout.gpu_memory_utilization=0.9 \
# rollout.max_num_batched_tokens=40000 \
# rollout.max_num_seqs=2048