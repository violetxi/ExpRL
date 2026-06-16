# ###### LLM likelihood judge for partial reasoning trace ######

python variational_inference/gen_rm/llm_likelihood_judge_process.py \
--data ${HF_NAMESPACE:-violetxi}/process_rollout-judge-calibration_set1_16k_qwen3-4b-score \
--model qwen/qwen3_4b_instruct \
--use_local_model \
--output ${HF_NAMESPACE:-violetxi}/icml_judge_set1_process \
--max-concurrency 128 \
--temperature 0.0
