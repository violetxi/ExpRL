# test_path=data/instruct/omni-math-difficulty-5/train.parquet
# python3 -m verl.trainer.main_generation \
# --config-path=config --config-name=generation \
# model.path=qwen/Qwen3-4B \
# model.revision=main \
# trainer.nnodes=1 \
# trainer.n_gpus_per_node=8 \
# data.path=$test_path \
# data.prompt_key=prompt \
# data.n_samples=1 \
# data.batch_size=2048 \
# data.use_chat_template=True \
# data.enable_thinking=False \
# data.output_path=${HF_NAMESPACE:-violetxi}/qwen4b-no-thinking-omni-l5 \
# rollout.prompt_length=2048 \
# rollout.response_length=16384 \
# rollout.temperature=0.6 \
# rollout.tensor_model_parallel_size=2 \
# rollout.gpu_memory_utilization=0.85 \
# rollout.max_num_batched_tokens=40000 \
# rollout.max_num_seqs=2048

###### inference on Qwen3-4B-Instruct-2507 ######
set -x


###### Genrate rollout for Omni Math L7 and above ######
test_path=data/instruct/omni-math-above-l7/train.parquet
python3 -m verl.trainer.main_generation \
--config-path=config --config-name=generation \
model.path=qwen/Qwen3-4B-Instruct-2507 \
model.revision=main \
trainer.nnodes=1 \
trainer.n_gpus_per_node=8 \
data.path=$test_path \
data.prompt_key=prompt \
data.n_samples=1 \
data.batch_size=1024 \
data.use_chat_template=True \
data.enable_thinking=False \
data.output_path=${HF_NAMESPACE:-violetxi}/qwen4b-instruct-2507-omni-l7 \
rollout.prompt_length=2048 \
rollout.response_length=16384 \
rollout.temperature=0.8 \
rollout.tensor_model_parallel_size=2 \
rollout.gpu_memory_utilization=0.85 \
rollout.max_num_batched_tokens=32768 \
rollout.max_num_seqs=2048