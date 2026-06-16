#!/bin/bash
#SBATCH --job-name=dense_pr_qwen8b
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:7
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=96:00:00
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
local_dir=checkpoints/pope_int_qwen8b/pr_delta_process/

set -x

python3 -m verl.trainer.main_rl_genrm \
    algorithm.adv_estimator=delta_process \
    data.train_files=$train_files \
    data.val_files=$test_files \
    data.train_batch_size=36 \
    data.max_prompt_length=4096 \
    data.max_response_length=16384 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.use_chat_template=True \
    data.enable_thinking=False \
    actor_rollout_ref.model.path=Qwen/Qwen3-8B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=36 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.26 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.max_model_len=32768 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.max_num_seqs=2048 \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.8 \
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    genrm.model.path=Qwen/Qwen3-4B-Instruct-2507 \
    genrm.rollout.prompt_length=32768 \
    genrm.rollout.response_length=2048 \
    genrm.rollout.gpu_memory_utilization=0.8 \
    genrm.rollout.tensor_model_parallel_size=1 \
    genrm.n_gpus=1 \
    +custom_reward_function.clip_length=True \
    +custom_reward_function.clip_by=full_sequence \
    +custom_reward_function.which_last_step=genrm \
    +custom_reward_function.is_process=True \
    +custom_reward_function.norm_strategy=smooth \
    +custom_reward_function.prompt_template_path=verl/utils/prompt_templates/math_likelihood_process.py \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.log_val_generations=50 \
    trainer.project_name='verl_exp_rl_qwen8b_no-thinking' \
    trainer.experiment_name='pr_delta' \
    trainer.n_gpus_per_node=7 \
    trainer.nnodes=1 \
    trainer.default_local_dir=$local_dir \
    trainer.save_freq=30 \
    trainer.test_freq=10 \
    trainer.total_epochs=5 $@