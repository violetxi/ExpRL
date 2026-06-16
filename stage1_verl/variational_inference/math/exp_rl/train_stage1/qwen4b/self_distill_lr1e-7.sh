#!/bin/bash
#SBATCH --job-name=self_distill_lr1e-7
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:3
#SBATCH --cpus-per-task=16
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
train_path_pope=data/instruct/pope_full/test.parquet
train_path_int=data/instruct/int_dataset/train.parquet
test_path_aime25=data/instruct/aime-2025/test.parquet
test_path_aime26=data/instruct/aime-2026/train.parquet

train_files="['$train_path_pope','$train_path_int']"
test_files="['$test_path_aime25','$test_path_aime26']"
local_dir=checkpoints/pope_int_qwen3-4b/onpolicy_distill_lr1e-7

set -x

python3 -m verl.trainer.main_onpolicy_distill \
    algorithm.adv_estimator=basic \
    data.train_files=$train_files \
    data.val_files=$test_files \
    data.train_batch_size=32 \
    data.max_prompt_length=4096 \
    data.max_response_length=16384 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.use_chat_template=True \
    actor_rollout_ref.actor.optim.lr=1e-7 \
    actor_rollout_ref.model.path=Qwen/Qwen3-4B-Instruct-2507 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=40000 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.26 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
    actor_rollout_ref.actor.policy_loss.loss_mode="distill" \
    actor_rollout_ref.rollout.max_model_len=40000 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.max_num_batched_tokens=40000 \
    actor_rollout_ref.rollout.max_num_seqs=2048 \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.8 \
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    +custom_reward_function.clip_length=False \
    +custom_reward_function.clip_by=full_sequence \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.log_val_generations=50 \
    trainer.project_name='verl_exp_rl_qwen3-4b_instruct' \
    trainer.experiment_name='self_distill_lr1e-7' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_local_dir=$local_dir \
    trainer.save_freq=30 \
    trainer.test_freq=10 \
    trainer.total_epochs=5 $@