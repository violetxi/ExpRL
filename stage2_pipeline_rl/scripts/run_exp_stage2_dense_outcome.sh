#!/bin/bash

python -m pipelinerl.launch \
--config-name=exp_stage2_dense_outcome \
wandb.wandb_project_name=prl_exp_stage2_qwen4b_instruct \
output_dir=./results/dense-outcome-16k \
"$@"
