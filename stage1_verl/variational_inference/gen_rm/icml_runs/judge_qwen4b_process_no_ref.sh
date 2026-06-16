#!/bin/bash
# Example script for running LLM judge WITHOUT reference solution (process/step level)
# This evaluates partial reasoning traces without seeing the ground truth answer

python variational_inference/gen_rm/llm_likelihood_judge_process_no_ref.py \
--data ${HF_NAMESPACE:-violetxi}/partial-qwen4b-instruct-2507-omni-l7-step-only \
--model qwen/qwen3_4b_instruct \
--use_local_model \
--output ${HF_NAMESPACE:-violetxi}/Qwen4b-Instruct-Judge-partial-omni-l7-step-no-ref \
--max-concurrency 256 \
--temperature 0.0

# Example with model launching (if vLLM server is not already running)
# python variational_inference/gen_rm/llm_likelihood_judge_process_no_ref.py \
# --data ${HF_NAMESPACE:-violetxi}/partial-qwen4b-instruct-2507-omni-l7-step-only \
# --model qwen/qwen3_4b_instruct \
# --use_local_model \
# --launch_model \
# --tensor_parallel_size 4 \
# --output ${HF_NAMESPACE:-violetxi}/Qwen4b-Instruct-Judge-partial-omni-l7-step-no-ref \
# --max-concurrency 256 \
# --temperature 0.0

# Example with larger model
# python variational_inference/gen_rm/llm_likelihood_judge_process_no_ref.py \
# --data ${HF_NAMESPACE:-violetxi}/partial-qwen4b-instruct-2507-omni-l7-step-only \
# --model qwen/qwen3_30b_instruct \
# --use_local_model \
# --output ${HF_NAMESPACE:-violetxi}/Qwen30b-Instruct-Judge-partial-omni-l7-step-no-ref \
# --max-concurrency 256 \
# --temperature 0.0

# Example with OpenRouter API (e.g., GPT-4, Claude)
# export OPENROUTER_API_KEY=your_key_here
# python variational_inference/gen_rm/llm_likelihood_judge_process_no_ref.py \
# --data ${HF_NAMESPACE:-violetxi}/partial-qwen4b-instruct-2507-omni-l7-step-only \
# --model anthropic/claude-3.5-sonnet \
# --output ${HF_NAMESPACE:-violetxi}/Claude-3.5-Judge-partial-omni-l7-step-no-ref \
# --max-concurrency 32 \
# --temperature 0.0
