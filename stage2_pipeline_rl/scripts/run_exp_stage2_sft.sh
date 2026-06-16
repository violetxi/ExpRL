#!/bin/bash

python -m pipelinerl.launch \
--config-name=exp_stage2_sft \
wandb.wandb_project_name=prl_exp_stage2_qwen4b_instruct \
actor.llm_max_rollouts=32 \
output_dir=./results/prl-exp-stage2-sft-16k \
"$@"
