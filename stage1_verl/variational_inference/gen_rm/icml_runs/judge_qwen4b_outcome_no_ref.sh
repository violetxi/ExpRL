#!/bin/bash
# Example script for running LLM judge WITHOUT reference solution (outcome level)
# This evaluates complete reasoning traces without seeing the ground truth answer

python variational_inference/gen_rm/llm_likelihood_judge_no_ref.py \
--data ${HF_NAMESPACE:-violetxi}/omni-rule-l7-above-gemini-pro-filtered_qwen3-4b-instruct-2507-score \
--model qwen/qwen3_4b_instruct \
--use_local_model \
--output ${HF_NAMESPACE:-violetxi}/Qwen4b-Instruct-Judge-omni-l7-gemini-pro-filtered-outcome-no-ref \
--max-concurrency 256 \
--temperature 0.0

# Example with larger model
# python variational_inference/gen_rm/llm_likelihood_judge_no_ref.py \
# --data ${HF_NAMESPACE:-violetxi}/omni-rule-l7-above-gemini-pro-filtered_qwen3-4b-instruct-2507-score \
# --model qwen/qwen3_30b_instruct \
# --use_local_model \
# --output ${HF_NAMESPACE:-violetxi}/Qwen30b-Instruct-Judge-omni-l7-gemini-pro-filtered-outcome-no-ref \
# --max-concurrency 256 \
# --temperature 0.0

# Example with OpenRouter API (e.g., GPT-4, Claude)
# export OPENROUTER_API_KEY=your_key_here
# python variational_inference/gen_rm/llm_likelihood_judge_no_ref.py \
# --data ${HF_NAMESPACE:-violetxi}/omni-rule-l7-above-gemini-pro-filtered_qwen3-4b-instruct-2507-score \
# --model anthropic/claude-3.5-sonnet \
# --output ${HF_NAMESPACE:-violetxi}/Claude-3.5-Judge-omni-l7-gemini-pro-filtered-outcome-no-ref \
# --max-concurrency 32 \
# --temperature 0.0
