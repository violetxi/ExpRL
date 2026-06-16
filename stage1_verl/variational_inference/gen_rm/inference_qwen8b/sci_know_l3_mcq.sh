#!/bin/bash
#SBATCH --job-name=judge_calibration_sciknoweval_mcq_l3_loose_qwen8b
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err

cd "${EXPRL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && while [ ! -f setup.py ] && [ "$PWD" != / ]; do cd ..; done && pwd)}"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate verl
unset ROCR_VISIBLE_DEVICES

export VLLM_TORCH_COMPILE_LEVEL=0


###### Smoke test: first 100 SciKnowEval MCQ L3 loose problems, end-to-end ######
# Validates the deterministic MCQ scorer (verl/utils/reward_score/sciknoweval_judge.py)
# end-to-end with real Qwen3-8B no-thinking outputs.
test_path=data/instruct/sciknoweval_mcq_l3_loose/train.parquet
python3 -m verl.trainer.vllm_generation_ray_dp \
--config-path=config --config-name=vllm_generation \
model.path=Qwen/Qwen3-8B \
model.revision=main \
data.path=$test_path \
data.prompt_key=prompt \
data.n_samples=1 \
data.batch_size=2048 \
data.use_chat_template=True \
data.enable_thinking=False \
data.output_path=${HF_NAMESPACE:-violetxi}/judge_calibration_sciknoweval_mcq_l3_loose_qwen8b \
data.chunk_output_dir=./chunks_judge_calibration_sciknoweval_mcq_l3_loose_qwen8b \
rollout.prompt_length=2048 \
rollout.response_length=16384 \
rollout.temperature=0.8 \
rollout.data_parallel_size=4 \
rollout.tensor_model_parallel_size=1 \
rollout.gpu_memory_utilization=0.8 \
rollout.max_num_batched_tokens=40000 \
rollout.max_num_seqs=2048 \
rollout.enforce_eager=True \
scoring.enabled=true
