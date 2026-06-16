# ###### LLM likelihood judge for partial reasoning trace ######
# python variational_inference/gen_rm/llm_likelihood_judge_process.py \
# --data ${HF_NAMESPACE:-violetxi}/test_pope_hard_qwen4b_instruct_2507-score-steps \
# --model qwen/qwen3_4b_instruct_2507 \
# --use_local_model \
# --output ${HF_NAMESPACE:-violetxi}/Qwen4b-2507-Judge-partial-pope-hard-qwen4b-instruct-2507 \
# --max-concurrency 256 \
# --temperature 0.0

# ###### LLM likelihood judge for partial reasoning trace ######
# python variational_inference/gen_rm/llm_likelihood_judge_process.py \
# --data ${HF_NAMESPACE:-violetxi}/test_pope_hard_qwen4b_instruct_2507-score-steps \
# --model qwen/qwen3_4b \
# --use_local_model \
# --output ${HF_NAMESPACE:-violetxi}/Qwen4b-Judge-partial-pope-hard-qwen4b \
# --max-concurrency 256 \
# --temperature 0.0

# python variational_inference/gen_rm/llm_likelihood_judge_process.py \
# --data ${HF_NAMESPACE:-violetxi}/qwen4b-instruct-2507-omni-l7-score-steps \
# --model qwen/qwen3_4b_instruct_2507 \
# --use_local_model \
# --output ${HF_NAMESPACE:-violetxi}/Qwen4b-2507-Judge-partial-qwen4b-instruct-2507-omni-l7 \
# --max-concurrency 128 \
# --temperature 0.0


python variational_inference/gen_rm/llm_likelihood_judge_process.py \
--data ${HF_NAMESPACE:-violetxi}/omni-rule-l7-above-gemini-pro-filtered_qwen3-4b-instruct-2507-steps \
--model qwen/qwen3_4b_instruct \
--use_local_model \
--output ${HF_NAMESPACE:-violetxi}/Qwen4b-Instruct-Judge-partial-qwen4b-instruct-omni-l7-step-score \
--max-concurrency 128 \
--temperature 0.0

python variational_inference/gen_rm/llm_likelihood_judge_process.py \
--data ${HF_NAMESPACE:-violetxi}/omni-rule-l7-above-gemini-pro-filtered_qwen3-4b-instruct-2507-steps \
--model qwen/qwen3_30b_instruct \
--use_local_model \
--output ${HF_NAMESPACE:-violetxi}/Qwen30b-Instruct-Judge-partial-qwen30b-instruct-omni-l7-step-score \
--max-concurrency 128 \
--temperature 0.0