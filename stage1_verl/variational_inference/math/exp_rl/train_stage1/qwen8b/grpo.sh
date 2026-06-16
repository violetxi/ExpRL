#!/bin/bash
#SBATCH --job-name=stage1_grpo_qwen8b
#SBATCH --account=YOUR_ACCOUNT
#SBATCH --partition=YOUR_PARTITION
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=64
#SBATCH --mem=720G
#SBATCH --time=48:00:00
#SBATCH --output=slurm_ray_cluster/%x_%j.out
#SBATCH --error=slurm_ray_cluster/%x_%j.err

set -euo pipefail

# --- Configuration ---
RAY_PORT=${RAY_PORT:-$((6300 + SLURM_JOB_ID % 1000))}
RAY_DASHBOARD_PORT=${RAY_DASHBOARD_PORT:-$((8300 + SLURM_JOB_ID % 1000))}
JOB_WORKING_DIR=${JOB_WORKING_DIR:-"${EXPRL_ROOT:-$PWD}"}
CONDA_ROOT=${CONDA_ROOT:-"$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")"}
CONDA_ENV=${CONDA_ENV:-"verl"}
CONDA_SH="$CONDA_ROOT/etc/profile.d/conda.sh"
PYTHON_BIN="$CONDA_ROOT/envs/$CONDA_ENV/bin/python3"
RAY_BIN="$CONDA_ROOT/envs/$CONDA_ENV/bin/ray"

export JOB_WORKING_DIR CONDA_ROOT CONDA_ENV CONDA_SH PYTHON_BIN RAY_BIN
export PATH="$CONDA_ROOT/envs/$CONDA_ENV/bin:$PATH"
export RAY_TMPDIR="/tmp/ray_${SLURM_JOB_ID}"
GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-1}
if [[ "$GPUS_PER_NODE" == *:* ]]; then
  GPUS_PER_NODE=${GPUS_PER_NODE##*:}
fi
export GPUS_PER_NODE

for required_path in "$JOB_WORKING_DIR" "$CONDA_SH" "$PYTHON_BIN" "$RAY_BIN"; do
  if [ ! -e "$required_path" ]; then
    echo "ERROR: required path does not exist: $required_path" >&2
    exit 1
  fi
done

# --- Environment variables (set BEFORE Ray start so they propagate to all Ray workers) ---
export HYDRA_FULL_ERROR=1
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# Network config for Slingshot interconnect on Delta-AI
# Use NCCL socket transport (TCP over Slingshot) - avoids OFI plugin compat issues
unset NCCL_NET
unset NCCL_NET_PLUGIN
export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=hsn
export NCCL_CROSS_NIC=1
export NCCL_COLLNET_ENABLE=0
export NCCL_DEBUG=WARN
export GLOO_SOCKET_IFNAME=hsn0
export TP_SOCKET_IFNAME=hsn0

head_pid=""
worker_pids=()

cleanup() {
  local rc=$?
  trap - EXIT INT TERM
  echo "Cleaning up Ray processes (script exit status: $rc)..."
  if [ -n "${head_pid:-}" ]; then
    kill "$head_pid" 2>/dev/null || true
  fi
  for p in "${worker_pids[@]}"; do
    kill "$p" 2>/dev/null || true
  done
  wait 2>/dev/null || true
  exit "$rc"
}
trap cleanup EXIT INT TERM

# --- Clean up stale Ray sessions on ALL nodes (with timeout) ---
echo "Cleaning up stale Ray sessions on allocated nodes..."
timeout 60 srun --export=ALL --nodes=$SLURM_NNODES --ntasks-per-node=1 bash -c '
  timeout 30 "$RAY_BIN" stop --force 2>/dev/null || true
  rm -rf "$RAY_TMPDIR" 2>/dev/null || true
  rm -f "$HOME/ray/node_ip_address.json" 2>/dev/null || true
' || echo "WARNING: Cleanup timed out, continuing anyway..."
sleep 5

echo "Using Ray temp directory: $RAY_TMPDIR (local to each node)"
echo "Using Ray ports: GCS=$RAY_PORT dashboard=$RAY_DASHBOARD_PORT"

# --- Setup ---
echo "Running on nodes: $SLURM_JOB_NODELIST"
echo "Job ID: $SLURM_JOB_ID"

nodes=$(scontrol show hostnames $SLURM_JOB_NODELIST)
nodes_array=($nodes)
head_node=${nodes_array[0]}
worker_nodes=("${nodes_array[@]:1}")

# Resolve head node IP via DNS (avoids IPv6 link-local scope ID issues with Ray)
head_node_ip=$(python3 -c "import socket; print(socket.gethostbyname('$head_node'))")
echo "Resolved $head_node via DNS to: $head_node_ip"

if [ -z "$head_node_ip" ]; then
   echo "ERROR: Failed to obtain head node IP address."
   exit 1
fi

echo "--------------------"
echo "Head Node: $head_node"
echo "Head Node IP: $head_node_ip"
echo "Worker Nodes: ${worker_nodes[@]}"
echo "GPUs per node: $GPUS_PER_NODE"
echo "--------------------"

# --- Start Ray Head Node ---
echo "Starting Ray head node on $head_node..."
srun --export=ALL --nodes=1 --ntasks=1 -w "$head_node" \
   "$RAY_BIN" start --head \
   --node-ip-address="$head_node_ip" \
   --port=$RAY_PORT \
   --include-dashboard=true \
   --dashboard-host=0.0.0.0 \
   --dashboard-port=$RAY_DASHBOARD_PORT \
   --temp-dir=$RAY_TMPDIR \
   --disable-usage-stats \
   --block &
head_pid=$!
echo "Ray head node PID: $head_pid"
echo "Waiting for head node to initialize..."
for i in $(seq 1 60); do
  if nc -z "$head_node_ip" "$RAY_PORT" 2>/dev/null; then
    echo "Ray GCS is reachable after $i checks."
    break
  fi
  sleep 2
done
if ! nc -z "$head_node_ip" "$RAY_PORT" 2>/dev/null; then
  echo "ERROR: Ray GCS port $RAY_PORT did not open on $head_node_ip" >&2
  cat "$RAY_TMPDIR"/session_*/logs/gcs_server.* 2>/dev/null | tail -80 || echo "No GCS logs found"
  exit 1
fi

# --- Start Ray Worker Nodes ---
echo "Starting Ray worker nodes..."
for worker_node in "${worker_nodes[@]}"; do
   worker_ip=$("$PYTHON_BIN" -c "import socket; print(socket.gethostbyname('$worker_node'))")
   echo "Starting worker on $worker_node ($worker_ip)"
   srun --export=ALL --nodes=1 --ntasks=1 -w "$worker_node" \
       "$RAY_BIN" start \
       --node-ip-address="$worker_ip" \
       --address="$head_node_ip:$RAY_PORT" \
       --temp-dir=$RAY_TMPDIR \
       --disable-usage-stats \
       --block &
   worker_pids+=($!)
   sleep 10
done
echo "Ray worker PIDs: ${worker_pids[@]}"

echo "Waiting for cluster to form completely..."
export RAY_ADDRESS="$head_node_ip:$RAY_PORT"
"$PYTHON_BIN" - <<'PY'
import os
import time

import ray

expected_nodes = int(os.environ["SLURM_NNODES"])
expected_gpus = expected_nodes * int(os.environ["GPUS_PER_NODE"])

ray.init(address=os.environ["RAY_ADDRESS"])
try:
    for i in range(60):
        alive_nodes = sum(1 for node in ray.nodes() if node["Alive"])
        resources = ray.cluster_resources()
        gpus = int(resources.get("GPU", 0))
        print(
            f"Ray resource check {i + 1}/60: alive_nodes={alive_nodes}, "
            f"GPUs={gpus}, resources={resources}",
            flush=True,
        )
        if alive_nodes >= expected_nodes and gpus >= expected_gpus:
            break
        time.sleep(5)
    else:
        raise SystemExit("ERROR: Ray cluster did not register expected nodes/GPUs")
finally:
    ray.shutdown()
PY

if curl -s -f --max-time 5 "http://$head_node_ip:$RAY_DASHBOARD_PORT/api/v0/jobs" > /dev/null 2>&1; then
  echo "Ray dashboard API is ready at $head_node_ip:$RAY_DASHBOARD_PORT"
else
  echo "WARNING: Ray dashboard API is not ready; training only requires GCS, so continuing."
fi

echo "Checking Ray cluster status..."
"$RAY_BIN" status --address="$RAY_ADDRESS" || echo "WARNING: Ray status check failed or cluster not fully ready yet."
sleep 5

# --- Run the Training Job ---
cd "$JOB_WORKING_DIR"
set +e
(
  source "$CONDA_SH"
  conda activate "$CONDA_ENV"
  unset ROCR_VISIBLE_DEVICES
  echo "Activated conda environment: $CONDA_DEFAULT_ENV"
  echo "Python path: $(which python3)"

  train_path_pope=data/instruct/pope_full/test.parquet
  train_path_int=data/instruct/int_dataset/train.parquet
  test_path_aime25=data/instruct/aime-2025/test.parquet
  test_path_aime26=data/instruct/aime-2026/train.parquet

  train_files="['$train_path_pope','$train_path_int']"
  test_files="['$test_path_aime25','$test_path_aime26']"
  local_dir=checkpoints/pope_int_qwen8b/grpo

  set -x

  "$PYTHON_BIN" -m verl.trainer.main_ppo \
      algorithm.adv_estimator=grpo \
      data.train_files=$train_files \
      data.val_files=$test_files \
      data.train_batch_size=32 \
      data.max_prompt_length=4096 \
      data.max_response_length=16384 \
      data.filter_overlong_prompts=True \
      data.truncation='error' \
      data.use_chat_template=True \
      data.enable_thinking=False \
      actor_rollout_ref.model.path=Qwen/Qwen3-8B \
      actor_rollout_ref.actor.optim.lr=1e-6 \
      actor_rollout_ref.model.use_remove_padding=True \
      actor_rollout_ref.actor.ppo_mini_batch_size=16 \
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
      actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
      actor_rollout_ref.rollout.name=vllm \
      actor_rollout_ref.rollout.temperature=0.8 \
      actor_rollout_ref.rollout.n=16 \
      actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
      actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
      actor_rollout_ref.rollout.max_num_seqs=2048 \
      actor_rollout_ref.rollout.disable_log_stats=False \
      actor_rollout_ref.rollout.val_kwargs.temperature=0.8 \
      actor_rollout_ref.rollout.val_kwargs.n=4 \
      actor_rollout_ref.rollout.val_kwargs.do_sample=True \
      actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
      actor_rollout_ref.ref.fsdp_config.param_offload=True \
      algorithm.use_kl_in_reward=False \
      trainer.critic_warmup=0 \
      trainer.logger='["console","wandb"]' \
      trainer.log_val_generations=50 \
      trainer.project_name='verl_exp_rl_qwen8b_no-thinking' \
      trainer.experiment_name='grpo' \
      trainer.n_gpus_per_node=$GPUS_PER_NODE \
      trainer.nnodes=$SLURM_NNODES \
      trainer.default_local_dir=$local_dir \
      trainer.save_freq=20 \
      trainer.test_freq=10 \
      trainer.total_epochs=5 "$@"
)
train_status=$?
set -e
echo "Training exited with status: $train_status"
exit "$train_status"
