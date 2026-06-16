#!/bin/bash
# Code-outcome judge smoke test (100 prompts, 1 sample each).
# Parallel to judge_qwen4b_outcome.sh (math) but for LCB v6 — uses
# verl/utils/prompt_templates/code_likelihood_judge.py as the prompt template.

python variational_inference/gen_rm/llm_likelihood_judge_code.py \
    --data ${HF_NAMESPACE:-violetxi}/lcb_v6_stage1_qwen4b_instruct \
    --oracle data/instruct/livecodebench_v6_with_oracle/train.parquet \
    --model qwen/qwen3_4b_instruct \
    --use_local_model \
    --output ${HF_NAMESPACE:-violetxi}/Qwen4b-Instruct-Judge-lcb-v6-stage1-outcome-smoke100 \
    --max-concurrency 256 \
    --temperature 0.0 \
    --max_prompts 100 \
    --max_samples_per_prompt 1
