#!/bin/bash

python -m pipelinerl.launch \
--config-name=exp_stage2_dense_process \
wandb.wandb_project_name=prl_exp_stage2_qwen4b_instruct \
output_dir=./results/dense-process-16k-1 \
"$@"
