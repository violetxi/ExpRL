#!/bin/bash
# Science-outcome judge for SciKnowEval OE v1 smoketest100.
# Parallel to judge_qwen4b_outcome.sh (math) and judge_qwen4b_outcome_code.sh
# but uses verl/utils/prompt_templates/science_likelihood_judge.py as the prompt template.

python variational_inference/gen_rm/llm_likelihood_judge_science.py \
    --data ${HF_NAMESPACE:-violetxi}/sciknoweval_oe_v1_smoketest100_qwen8b \
    --model qwen/qwen3_4b_instruct \
    --use_local_model \
    --output ${HF_NAMESPACE:-violetxi}/Qwen4b-Instruct-Judge-sciknow-oe-v1-smoketest100-outcome \
    --max-concurrency 256 \
    --temperature 0.0

# Larger judge variant
# python variational_inference/gen_rm/llm_likelihood_judge_science.py \
#     --data ${HF_NAMESPACE:-violetxi}/sciknoweval_oe_v1_smoketest100_qwen8b \
#     --model qwen/qwen3_30b_instruct \
#     --use_local_model \
#     --output ${HF_NAMESPACE:-violetxi}/Qwen30b-Instruct-Judge-sciknow-oe-v1-smoketest100-outcome \
#     --max-concurrency 256 \
#     --temperature 0.0
