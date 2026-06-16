#!/bin/bash
#SBATCH --job-name=dense_mixed_qwen8b
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:8
#SBATCH --cpus-per-task=32
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
# NOTE: this script uses the new compute_mixed_genrm_score path. Per-row routing:
#   - math (pope, int) + sciknow MCQ → process reward, ### step-splitting,
#     math_likelihood_process.py template
#   - code (lcb) + sciknow OE        → outcome reward (single GenRM call per
#     response) with per-domain templates (code_likelihood_judge.py,
#     sciknow_likelihood_judge.py)
# Rule-based outcome reward is also computed for every row (used by metrics +
# optionally as which_last_step=outcome). For OE physics rows the deterministic
# Gemini Flash Lite judge fires; export GEMINI_API_KEY before launching.

# All 5 training datasets — covers math (pope, int), code (lcb v6), science OE
# (physics) and science MCQ (5 L3 reasoning tasks).
train_pope=data/instruct/pope_full/test.parquet
train_int=data/instruct/int_dataset/train.parquet
train_lcb=data/instruct/livecodebench_v6_with_oracle/train.parquet
train_oe=data/instruct/sciknoweval_oe_v1_physics/train.parquet
train_mcq=data/instruct/sciknoweval_mcq_l3_loose/train.parquet

# Cross-domain val: aime-2025 (math, 30) + gpqa_diamond (science, 198)
test_aime25=data/instruct/aime-2025/test.parquet
test_gpqa=data/instruct/gpqa_diamond/test.parquet

train_files="['$train_pope','$train_int','$train_lcb','$train_oe','$train_mcq']"
test_files="['$test_aime25','$test_gpqa']"
local_dir=checkpoints/dense_mixed_qwen8b_all_domains/

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
    actor_rollout_ref.actor.ppo_mini_batch_size=18 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=2 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=16384 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.max_model_len=32768 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.8 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.75 \
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
    genrm.rollout.response_length=4096 \
    genrm.rollout.gpu_memory_utilization=0.8 \
    genrm.rollout.tensor_model_parallel_size=1 \
    genrm.n_gpus=2 \
    +custom_reward_function.is_mixed=True \
    +custom_reward_function.clip_length=True \
    +custom_reward_function.clip_by=full_sequence \
    +custom_reward_function.which_last_step=genrm \
    +custom_reward_function.norm_strategy=smooth \
    +custom_reward_function.prompt_template_path=verl/utils/prompt_templates/math_likelihood_process.py \
    +custom_reward_function.prompt_template_paths.code=verl/utils/prompt_templates/code_likelihood_judge.py \
    +custom_reward_function.prompt_template_paths.science=verl/utils/prompt_templates/sciknow_likelihood_judge.py \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.log_val_generations=50 \
    trainer.project_name='verl_exp_rl_qwen8b_no-thinking' \
    trainer.experiment_name='dense_mixed' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.default_local_dir=$local_dir \
    trainer.max_actor_ckpt_to_keep=5 \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.total_epochs=3 $@
