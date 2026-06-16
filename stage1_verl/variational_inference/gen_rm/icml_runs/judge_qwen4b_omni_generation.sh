###### LLM likelihood judge for partial reasoning trace ######
python variational_inference/gen_rm/llm_likelihood_judge_process.py \
--data ${HF_NAMESPACE:-violetxi}/qwen4b-no-thinking-omni-l5-score-steps \
--model qwen/qwen3_4b \
--use_local_model \
--output ${HF_NAMESPACE:-violetxi}/Qwen4b-Judge-partial-qwen4b-no-thinking-omni-l5 \
--max-concurrency 256 \
--temperature 0.0

python variational_inference/gen_rm/llm_likelihood_judge_process.py \
--data ${HF_NAMESPACE:-violetxi}/qwen4b-no-thinking-omni-l5-score-steps \
--model qwen/qwen3_8b \
--use_local_model \
--output ${HF_NAMESPACE:-violetxi}/Qwen8b-Judge-partial-qwen4b-no-thinking-omni-l5 \
--max-concurrency 256 \
--temperature 0.0

python variational_inference/gen_rm/llm_likelihood_judge_process.py \
--data ${HF_NAMESPACE:-violetxi}/qwen4b-no-thinking-omni-l5-score-steps \
--model qwen/qwen3_14b \
--use_local_model \
--output ${HF_NAMESPACE:-violetxi}/Qwen14b-Judge-partial-qwen4b-no-thinking-omni-l5 \
--max-concurrency 256 \
--temperature 0.0

###### LLM likelihood judge for full reasoning trace ######
# ### Qwen4b as judge
# python variational_inference/gen_rm/llm_likelihood_judge.py \
# --data ${HF_NAMESPACE:-violetxi}/qwen4b-thinking-omni-l8-score \
# --model qwen/qwen3_4b \
# --use_local_model \
# --output ${HF_NAMESPACE:-violetxi}/Qwen4b-Judge-qwen4b-thinking-omni-l8 \
# --max-concurrency 256 \
# --temperature 0.0

# python variational_inference/gen_rm/llm_likelihood_judge.py \
# --data ${HF_NAMESPACE:-violetxi}/qwen4b-no-thinking-omni-l8-score \
# --model qwen/qwen3_4b \
# --use_local_model \
# --output ${HF_NAMESPACE:-violetxi}/Qwen4b-Judge-qwen4b-no-thinking-omni-l8 \
# --max-concurrency 256 \
# --temperature 0.0

# ython variational_inference/gen_rm/llm_likelihood_judge.py \
# --data ${HF_NAMESPACE:-violetxi}/qwen4b-thinking-omni-l5-score \
# --model qwen/qwen3_4b \
# --use_local_model \
# --output ${HF_NAMESPACE:-violetxi}/Qwen4b-Judge-qwen4b-thinking-omni-l5 \
# --max-concurrency 256 \
# --temperature 0.0

# python variational_inference/gen_rm/llm_likelihood_judge.py \
# --data ${HF_NAMESPACE:-violetxi}/qwen4b-no-thinking-omni-l5-score \
# --model qwen/qwen3_4b \
# --use_local_model \
# --output ${HF_NAMESPACE:-violetxi}/Qwen4b-Judge-qwen4b-no-thinking-omni-l5 \
# --max-concurrency 256 \
# --temperature 0.0

# ## 32k
# python variational_inference/gen_rm/llm_likelihood_judge.py \
# --data ${HF_NAMESPACE:-violetxi}/qwen4b-thinking-omni-l5-32k-score \
# --model qwen/qwen3_4b \
# --use_local_model \
# --output ${HF_NAMESPACE:-violetxi}/Qwen4b-Judge-qwen4b-thinking-omni-l5-32k \
# --max-concurrency 256 \
# --temperature 0.0

# python variational_inference/gen_rm/llm_likelihood_judge.py \
# --data ${HF_NAMESPACE:-violetxi}/qwen4b-thinking-omni-l8-32k-score \
# --model qwen/qwen3_4b \
# --use_local_model \
# --output ${HF_NAMESPACE:-violetxi}/Qwen4b-Judge-qwen4b-thinking-omni-l8-32k \
# --max-concurrency 256 \
# --temperature 0.0

# ### GPT-4.1 as judge
# python variational_inference/gen_rm/llm_likelihood_judge.py \
# --data ${HF_NAMESPACE:-violetxi}/qwen4b-thinking-omni-l8-score \
# --model openai/gpt-4.1 \
# --output ${HF_NAMESPACE:-violetxi}/GPT4.1-Judge-qwen4b-thinking-omni-l8 \
# --max-concurrency 256 \
# --temperature 0.0

# ## 32k
# python variational_inference/gen_rm/llm_likelihood_judge.py \
# --data ${HF_NAMESPACE:-violetxi}/qwen4b-thinking-omni-l8-32k-score \
# --model openai/gpt-4.1 \
# --output ${HF_NAMESPACE:-violetxi}/GPT4.1-Judge-qwen4b-thinking-omni-l8-32k \
# --max-concurrency 256 \
# --temperature 0.0

# # ### Qwen8b as judge
# python variational_inference/gen_rm/llm_likelihood_judge.py \
# --data ${HF_NAMESPACE:-violetxi}/qwen4b-thinking-omni-l8-score \
# --model qwen/qwen3_8b \
# --use_local_model \
# --output ${HF_NAMESPACE:-violetxi}/Qwen8b-Judge-qwen4b-thinking-omni-l8 \
# --max-concurrency 256 \
# --temperature 0.0

# python variational_inference/gen_rm/llm_likelihood_judge.py \
# --data ${HF_NAMESPACE:-violetxi}/qwen4b-thinking-omni-l5-score \
# --model qwen/qwen3_8b \
# --use_local_model \
# --output ${HF_NAMESPACE:-violetxi}/Qwen8b-Judge-qwen4b-thinking-omni-l5 \
# --max-concurrency 256 \
# --temperature 0.0

## 32k
python variational_inference/gen_rm/llm_likelihood_judge.py \
--data ${HF_NAMESPACE:-violetxi}/qwen4b-thinking-omni-l5-32k-score \
--model qwen/qwen3_8b \
--use_local_model \
--output ${HF_NAMESPACE:-violetxi}/Qwen8b-Judge-qwen4b-thinking-omni-l5-32k \
--max-concurrency 256 \
--temperature 0.0

# python variational_inference/gen_rm/llm_likelihood_judge.py \
# --data ${HF_NAMESPACE:-violetxi}/qwen4b-thinking-omni-l8-32k-score \
# --model qwen/qwen3_8b \
# --use_local_model \
# --output ${HF_NAMESPACE:-violetxi}/Qwen8b-Judge-qwen4b-thinking-omni-l8-32k \
# --max-concurrency 256 \
# --temperature 0.0

# #### Qwen14b as judge
# python variational_inference/gen_rm/llm_likelihood_judge.py \
# --data ${HF_NAMESPACE:-violetxi}/qwen4b-thinking-omni-l8-score \
# --model qwen/qwen3_14b \
# --use_local_model \
# --output ${HF_NAMESPACE:-violetxi}/Qwen14b-Judge-qwen4b-thinking-omni-l8 \
# --max-concurrency 256 \
# --temperature 0.0

# python variational_inference/gen_rm/llm_likelihood_judge.py \
# --data ${HF_NAMESPACE:-violetxi}/qwen4b-thinking-omni-l5-32k-score \
# --model qwen/qwen3_14b \
# --use_local_model \
# --output ${HF_NAMESPACE:-violetxi}/Qwen14b-Judge-qwen4b-thinking-omni-l5-32k \
# --max-concurrency 256 \
# --temperature 0.0

# python variational_inference/gen_rm/llm_likelihood_judge.py \
# --data ${HF_NAMESPACE:-violetxi}/qwen4b-thinking-omni-l8-32k-score \
# --model qwen/qwen3_14b \
# --use_local_model \
# --output ${HF_NAMESPACE:-violetxi}/Qwen14b-Judge-qwen4b-thinking-omni-l8-32k \
# --max-concurrency 256 \
# --temperature 0.0


##### LLM likelihood judge summary ######
# # ### Qwen4b as judge (no enable thinking turned on)
# python variational_inference/gen_rm/llm_likelihood_judge_summary.py \
# --data ${HF_NAMESPACE:-violetxi}/qwen4b-thinking-omni-l5-32k-score \
# --model qwen/qwen3_4b \
# --use_local_model \
# --output ${HF_NAMESPACE:-violetxi}/Qwen4b-Judge-Summary-qwen4b-thinking-omni-l5-32k \
# --max-concurrency 256 \
# --temperature 0.0

# python variational_inference/gen_rm/llm_likelihood_judge_summary.py \
# --data ${HF_NAMESPACE:-violetxi}/qwen4b-thinking-omni-l8-32k-score \
# --model qwen/qwen3_4b \
# --use_local_model \
# --output ${HF_NAMESPACE:-violetxi}/Qwen4b-Judge-Summary-qwen4b-thinking-omni-l8 \
# --max-concurrency 256 \
# --temperature 0.0

# ### Qwen8b as judge (no enable thinking turned on)
# python variational_inference/gen_rm/llm_likelihood_judge_summary.py \
# --data ${HF_NAMESPACE:-violetxi}/qwen4b-thinking-omni-l5-32k-score \
# --model qwen/qwen3_8b \
# --use_local_model \
# --output ${HF_NAMESPACE:-violetxi}/Qwen8b-Judge-Summary-qwen4b-thinking-omni-l5-32k \
# --max-concurrency 256 \
# --temperature 0.0

# python variational_inference/gen_rm/llm_likelihood_judge_summary.py \
# --data ${HF_NAMESPACE:-violetxi}/qwen4b-thinking-omni-l8-32k-score \
# --model qwen/qwen3_8b \
# --use_local_model \
# --output ${HF_NAMESPACE:-violetxi}/Qwen8b-Judge-Summary-qwen4b-thinking-omni-l8-32k \
# --max-concurrency 256 \
# --temperature 0.0

# ### Qwen14b as judge (no enable thinking turned on)
# python variational_inference/gen_rm/llm_likelihood_judge_summary.py \
# --data ${HF_NAMESPACE:-violetxi}/qwen4b-thinking-omni-l5-32k-score \
# --model qwen/qwen3_14b \
# --use_local_model \
# --output ${HF_NAMESPACE:-violetxi}/Qwen14b-Judge-Summary-qwen4b-thinking-omni-l5-32k \
# --max-concurrency 256 \
# --temperature 0.0

# python variational_inference/gen_rm/llm_likelihood_judge_summary.py \
# --data ${HF_NAMESPACE:-violetxi}/qwen4b-thinking-omni-l8-32k-score \
# --model qwen/qwen3_14b \
# --use_local_model \
# --output ${HF_NAMESPACE:-violetxi}/Qwen14b-Judge-Summary-qwen4b-thinking-omni-l8-32k \
# --max-concurrency 256 \
# --temperature 0.0