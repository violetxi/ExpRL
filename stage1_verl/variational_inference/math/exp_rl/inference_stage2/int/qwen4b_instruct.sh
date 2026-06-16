#!/bin/bash
#SBATCH --job-name=int_qwen4b_instruct_stage1
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_PARTITION
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=96
#SBATCH --mem=512G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_ray_cluster/%x_%j.out
#SBATCH --error=slurm_ray_cluster/%x_%j.err
# (cluster reservation directive removed for public release)


cd "${EXPRL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && while [ ! -f setup.py ] && [ "$PWD" != / ]; do cd ..; done && pwd)}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate verl
unset ROCR_VISIBLE_DEVICES
# set -x

# export TORCHDYNAMO_VERBOSE=1
# export TORCH_LOGS="+dynamo"
export VLLM_TORCH_COMPILE_LEVEL=0

###### POPE Hard eval 
test_path=data/instruct/int_dataset/train.parquet
python3 -m verl.trainer.vllm_generation_ray_dp \
--config-path=config --config-name=vllm_generation \
model.path=Qwen/Qwen3-4B-Instruct-2507 \
model.revision=checkpoint-500 \
data.path=$test_path \
data.prompt_key=prompt \
data.n_samples=128 \
data.batch_size=2048 \
data.use_chat_template=True \
data.enable_thinking=False \
data.output_path=${HF_NAMESPACE:-violetxi}/int_stage1_qwen4b_instruct_ckpt500 \
data.chunk_output_dir=./chunks_int_stage1_qwen4b_instruct_ckpt500 \
rollout.prompt_length=2048 \
rollout.response_length=16384 \
rollout.temperature=0.8 \
rollout.data_parallel_size=4 \
rollout.tensor_model_parallel_size=1 \
rollout.gpu_memory_utilization=0.8 \
rollout.max_num_batched_tokens=40000 \
rollout.max_num_seqs=4096 \
rollout.enforce_eager=True
