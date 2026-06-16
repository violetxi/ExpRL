#!/bin/bash
#SBATCH --job-name=lcb_v6_grpo_stage1
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:7
#SBATCH --cpus-per-task=64
#SBATCH --mem=900G
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
export VLLM_TORCH_COMPILE_LEVEL=0


###### LiveCodeBench v6
test_path=data/instruct/livecodebench_v6/test.parquet
python3 -m verl.trainer.vllm_generation_ray_dp \
--config-path=config --config-name=vllm_generation \
model.path=${HF_NAMESPACE:-violetxi}/exp_stage1_qwen3-4b_grpo \
model.revision=main \
data.path=$test_path \
data.prompt_key=prompt \
data.n_samples=128 \
data.batch_size=2048 \
data.use_chat_template=True \
data.enable_thinking=False \
data.output_path=${HF_NAMESPACE:-violetxi}/lcb_v6_stage1_grpo \
data.chunk_output_dir=./chunks_lcb_v6_stage1_grpo \
rollout.prompt_length=3072 \
rollout.response_length=16384 \
rollout.temperature=0.8 \
rollout.data_parallel_size=7 \
rollout.tensor_model_parallel_size=1 \
rollout.gpu_memory_utilization=0.8 \
rollout.max_num_batched_tokens=40000 \
rollout.max_num_seqs=4096 \
rollout.enforce_eager=True \
scoring.enabled=true
