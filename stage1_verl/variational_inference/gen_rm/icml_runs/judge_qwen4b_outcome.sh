python variational_inference/gen_rm/llm_likelihood_judge.py \
--data ${HF_NAMESPACE:-violetxi}/omni-rule-l7-above-gemini-pro-filtered_qwen3-4b-instruct-2507-score \
--model qwen/qwen3_4b_instruct \
--use_local_model \
--output ${HF_NAMESPACE:-violetxi}/Qwen4b-Instruct-Judge-omni-l7-gemini-pro-filtered-outcome \
--max-concurrency 256 \
--temperature 0.0

python variational_inference/gen_rm/llm_likelihood_judge.py \
--data ${HF_NAMESPACE:-violetxi}/omni-rule-l7-above-gemini-pro-filtered_qwen3-4b-instruct-2507-score \
--model qwen/qwen3_30b_instruct \
--use_local_model \
--output ${HF_NAMESPACE:-violetxi}/Qwen30b-Instruct-Judge-omni-l7-gemini-pro-filtered-outcome \
--max-concurrency 256 \
--temperature 0.0