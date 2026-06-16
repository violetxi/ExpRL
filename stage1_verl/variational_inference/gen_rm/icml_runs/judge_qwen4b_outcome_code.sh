python variational_inference/gen_rm/llm_likelihood_judge_code.py \
--data ${HF_NAMESPACE:-violetxi}/lcb_v6_stage1_qwen4b_instruct \
--oracle data/instruct/livecodebench_v6_with_oracle/train.parquet \
--model qwen/qwen3_4b_instruct \
--use_local_model \
--output ${HF_NAMESPACE:-violetxi}/Qwen4b-Instruct-Judge-lcb-v6-stage1-outcome \
--max-concurrency 256 \
--temperature 0.0
