set -x

if [ "$#" -lt 2 ]; then
    echo "Usage: run_qwen_05_sp2.sh <nproc_per_node> <save_path> [other_configs...]"
    exit 1
fi

nproc_per_node=$1
save_path=$2

# Shift the arguments so $@ refers to the rest
shift 2

train_pope=data/instruct/pope_full/test.parquet
train_int=data/instruct/int_dataset/train.parquet
train_lcb=data/instruct/livecodebench_v6_with_oracle/train.parquet
train_oe=data/instruct/sciknoweval_oe_v1_physics/train.parquet
train_mcq=data/instruct/sciknoweval_mcq_l3_loose/train.parquet

train_files="['$train_pope','$train_int','$train_lcb','$train_oe','$train_mcq']"
test_files="['$train_oe']"


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
    trainer.project_name=exp_rl_stage1_qwen8b \
    trainer.experiment_name=sft-lr1e-6 \
    trainer.logger=['console','wandb'] \
    ulysses_sequence_parallel_size=2 \
    trainer.total_epochs=2 \
    trainer.total_training_steps=600 \
    trainer.save_freq=68 \
    trainer.test_freq=5 \
    use_remove_padding=true \
    $@
