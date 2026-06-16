# ###### LLM likelihood judge for partial reasoning trace ######

python variational_inference/gen_rm/llm_likelihood_judge_process_no_ref.py \
--data ${HF_NAMESPACE:-violetxi}/judge-calibration_set1_16k_qwen3-4b-score-steps \
--model qwen/qwen3_4b_instruct \
--use_local_model \
--output ${HF_NAMESPACE:-violetxi}/icml_judge_set1_process_no_ref \
--max-concurrency 128 \
--temperature 0.0

# OMNI-L7 No Ref
python variational_inference/gen_rm/llm_likelihood_judge_process_no_ref.py \
--data ${HF_NAMESPACE:-violetxi}/omni-rule-l7-above-gemini-pro-filtered_qwen3-4b-instruct-2507-steps \
--model qwen/qwen3_4b_instruct \
--use_local_model \
--output ${HF_NAMESPACE:-violetxi}/omni-l7-Judge-partial-qwen4b-instruct-omni-l7-step-score-no_ref \
--max-concurrency 128 \
--temperature 0.0