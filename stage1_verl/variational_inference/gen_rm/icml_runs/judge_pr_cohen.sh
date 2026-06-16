###### LLM likelihood judge for partial reasoning trace ######
python variational_inference/gen_rm/llm_likelihood_judge_process.py \
--data ${HF_NAMESPACE:-violetxi}/test_cohen_qwen4b_genrm-smooth-delta_30-score-steps \
--model qwen/qwen3_4b \
--use_local_model \
--output ${HF_NAMESPACE:-violetxi}/Qwen4b-Judge-partial_cohen_qwen4b_pr-smooth-delta_30 \
--max-concurrency 256 \
--temperature 0.0

python variational_inference/gen_rm/llm_likelihood_judge_process.py \
--data ${HF_NAMESPACE:-violetxi}/test_cohen_qwen4b_genrm-smooth-delta_60-score-steps \
--model qwen/qwen3_4b \
--use_local_model \
--output ${HF_NAMESPACE:-violetxi}/Qwen4b-Judge-partial_cohen_qwen4b_pr-smooth-delta_60 \
--max-concurrency 256 \
--temperature 0.0

python variational_inference/gen_rm/llm_likelihood_judge_process.py \
--data ${HF_NAMESPACE:-violetxi}/test_cohen_qwen4b_genrm-smooth-delta_90-score-steps \
--model qwen/qwen3_4b \
--use_local_model \
--output ${HF_NAMESPACE:-violetxi}/Qwen4b-Judge-partial_cohen_qwen4b_pr-smooth-delta_90 \
--max-concurrency 256 \
--temperature 0.0

python variational_inference/gen_rm/llm_likelihood_judge_process.py \
--data ${HF_NAMESPACE:-violetxi}/test_cohen_qwen4b_genrm-smooth-delta_120-score-steps \
--model qwen/qwen3_4b \
--use_local_model \
--output ${HF_NAMESPACE:-violetxi}/Qwen4b-Judge-partial_cohen_qwen4b_pr-smooth-delta_120 \
--max-concurrency 256 \
--temperature 0.0

python variational_inference/gen_rm/llm_likelihood_judge_process.py \
--data ${HF_NAMESPACE:-violetxi}/test_cohen_qwen4b_genrm-smooth-delta_main-score-steps \
--model qwen/qwen3_4b \
--use_local_model \
--output ${HF_NAMESPACE:-violetxi}/Qwen4b-Judge-partial_cohen_qwen4b_pr-smooth-delta_main \
--max-concurrency 256 \
--temperature 0.0