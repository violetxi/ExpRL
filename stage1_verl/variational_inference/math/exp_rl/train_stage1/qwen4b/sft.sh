#!/bin/bash
#SBATCH --job-name=stage1_sft
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=96:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

# Stage-1 baseline: supervised fine-tuning (SFT) on the reference solutions of
# the math mid-training mixture (InT + POPE). This produces the "SFT" priming
# init that Stage-2 RL is then run from. Single node, 8 GPUs.
#
# Usage: bash sft.sh [nproc_per_node] [save_path]

# envs
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate verl

# project dir (override EXPRL_ROOT to point at your stage1_verl checkout)
cd "${EXPRL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && while [ ! -f setup.py ] && [ "$PWD" != / ]; do cd ..; done && pwd)}"

set -x

nproc_per_node=${1:-8}
save_path=${2:-checkpoints/pope_int_qwen3-4b/sft}
shift 2 2>/dev/null || true

train_pope=data/instruct/pope_full/test.parquet
train_int=data/instruct/int_dataset/train.parquet

train_files="['$train_pope','$train_int']"
test_files="['$train_int']"

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$train_files \
    data.val_files=$test_files \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    data.max_length=20480 \
    data.train_batch_size=32 \
    data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['solution'] \
    data.micro_batch_size_per_gpu=4 \
    optim.lr=1e-6 \
    model.partial_pretrain=Qwen/Qwen3-4B-Instruct-2507 \
    model.use_liger=True \
    trainer.default_local_dir=$save_path \
    trainer.project_name=verl_exp_rl_qwen3-4b_instruct \
    trainer.experiment_name=sft-lr1e-6 \
    trainer.logger=['console','wandb'] \
    ulysses_sequence_parallel_size=2 \
    trainer.total_epochs=2 \
    trainer.total_training_steps=600 \
    trainer.save_freq=68 \
    trainer.test_freq=5 \
    use_remove_padding=true \
    $@
