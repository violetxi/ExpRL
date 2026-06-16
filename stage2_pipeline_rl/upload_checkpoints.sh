# # SFT (Done!)
# python upload_checkpoints.py \
# --checkpoint_dir results/prl-exp-stage2-sft-16k/finetune/ \
# --hf_repo_id ${HF_NAMESPACE:-violetxi}/exp_stage2_qwen3-4b_sft \
# --main_checkpoint current

# # Self-Distillation (Done!)
# python upload_checkpoints.py \
# --checkpoint_dir results/prl-exp-stage2-self-distill-16k/finetune/ \
# --hf_repo_id ${HF_NAMESPACE:-violetxi}/exp_stage2_qwen3-4b_self_distill \
# --main_checkpoint current

# # GRPO (Done!)
# python upload_checkpoints.py \
# --checkpoint_dir results/prl-exp-stage2-grpo-16k/finetune/ \
# --hf_repo_id ${HF_NAMESPACE:-violetxi}/exp_stage2_qwen3-4b_grpo \
# --main_checkpoint current

# # Dense Outcome (Done!)
# python upload_checkpoints.py \
# --checkpoint_dir results/dense-outcome-16k/finetune/ \
# --hf_repo_id ${HF_NAMESPACE:-violetxi}/exp_stage2_qwen3-4b_dense_outcome \
# --main_checkpoint current

# Dense Process (Done!)
python upload_checkpoints.py \
--checkpoint_dir results/dense-process-16k/finetune/ \
--hf_repo_id ${HF_NAMESPACE:-violetxi}/exp_stage2_qwen3-4b_dense_process \
--main_checkpoint current