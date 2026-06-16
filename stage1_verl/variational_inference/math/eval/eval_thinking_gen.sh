# echo "Evaluating Qwen8B-sft-thoughts-omni-l1_2"
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/qwen8b-sft-thoughts-omni-l1_2

# echo "Evaluating Qwen8B-sft-thoughts-omni-l2_3"
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/qwen8b-sft-thoughts-omni-l2_3

# echo "Evaluating Qwen4B-thinking-omni-l5"
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/qwen4b-thinking-omni-l5

# echo "Evaluating Qwen4B-thinking-omni-l8"
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/qwen4b-thinking-omni-l8

# echo "Evaluating Qwen4B-thinking-omni-l5"
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/qwen4b-thinking-omni-l5-32k

# echo "Evaluating Qwen4B-thinking-omni-l8"
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/qwen4b-thinking-omni-l8-32k

# echo "Evaluating Qwen4B-thinking-omni-l1_2"
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/qwen4b-thinking-omni-l1_2

# echo "Evaluating Qwen4B-thinking-omni-l2_3"
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/qwen4b-thinking-omni-l2_3

# echo "Evaluating ${HF_NAMESPACE:-violetxi}/test-qwen8b-rewrite-omni-l1_2"
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/qwen4b-thinking-omni-l1_2

# echo "Evaluating ${HF_NAMESPACE:-violetxi}/test-qwen8b-rewrite-omni-l2_3"
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/qwen4b-thinking-omni-l2_3

# echo "Evaluating qwen4b-no-thinking-omni-l8"
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/qwen4b-no-thinking-omni-l8

# echo "Evaluating qwen4b-thinking-omni-l5"
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/qwen4b-no-thinking-omni-l5

# echo "Evaluating qwen4b-thinking-omni-l4"
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/qwen4b-thinking-omni-l4

# echo "Evaluating qwen8b-rewrite-omni-l4"
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/test-qwen8b-rewrite-omni-l4

# echo "Evaluating qwen4b-thinking-omni-l1_2"
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/qwen4b-thinking-omni-l1_2

echo "Evaluating qwen8b-rewrite-omni-l1_2"
python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/qwen8b-rewrite-omni-l1_2

echo "Evaluating qwen8b-rewrite-omni-l2_3"
python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/qwen8b-rewrite-omni-l2_3

echo "Evaluating qwen4b-thinking-model_rewrite-omni-l4"
python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/qwen8b-rewrite-omni-l4

echo "Evaluating test-qwen8b-rewrite-omni-l1_2"
python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/test-qwen8b-rewrite-omni-l1_2

python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/qwen4b-no-thinking-omni-l2

#### gen_rm ####
echo "Evaluating ${HF_NAMESPACE:-violetxi}/process_rollout-qwen4b-omni-l5"
python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/process_rollout-qwen4b-omni-l5