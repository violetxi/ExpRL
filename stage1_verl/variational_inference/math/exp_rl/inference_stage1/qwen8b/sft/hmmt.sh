#!/bin/bash
#SBATCH --job-name=hmmt_qwen8b_sft_stage1
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=24:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

cd "${EXPRL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && while [ ! -f setup.py ] && [ "$PWD" != / ]; do cd ..; done && pwd)}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate verl
unset ROCR_VISIBLE_DEVICES

export VLLM_TORCH_COMPILE_LEVEL=0


###### HMMT Nov 2025 — SFT (exp_rl_stage1_qwen8b_sft, main = global_step_204)
test_path=data/instruct/hmmt-nov-2025/train.parquet
python3 -m verl.trainer.vllm_generation_ray_dp \
--config-path=config --config-name=vllm_generation \
model.path=${HF_NAMESPACE:-violetxi}/exp_rl_stage1_qwen8b_sft \
model.revision=main \
data.path=$test_path \
data.prompt_key=prompt \
data.n_samples=128 \
data.batch_size=2048 \
data.use_chat_template=True \
data.enable_thinking=False \
data.output_path=${HF_NAMESPACE:-violetxi}/hmmt-nov-2025_stage1_qwen8b_sft \
data.chunk_output_dir=./chunks_hmmt-nov-2025_stage1_qwen8b_sft \
rollout.prompt_length=2048 \
rollout.response_length=16384 \
rollout.temperature=0.8 \
rollout.data_parallel_size=4 \
rollout.tensor_model_parallel_size=1 \
rollout.gpu_memory_utilization=0.8 \
rollout.max_num_batched_tokens=40000 \
rollout.max_num_seqs=4096 \
rollout.enforce_eager=True \
scoring.enabled=true
