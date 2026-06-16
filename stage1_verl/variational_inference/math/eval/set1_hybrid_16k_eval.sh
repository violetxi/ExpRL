###### Evaluation on Set1 ######

#### Qwen3-4B #####
# echo "Evaluating Qwen3-4B on Set1 16k"
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/set1_16k_qwen3-4b data.response_key=response

# echo "Evaluating Qwen3-4B on AIME 2024"
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime24_16k_qwen3-4b data.response_key=response

# echo "Evaluating Qwen3-4B on AIME 2025"
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime25_16k_qwen3-4b data.response_key=response

#### GRPO ####
# echo "Evaluating GRPO Qwen3-4B on Set1 16k"
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/set1_16k_grpo data.response_key=response

# echo "Evaluating GRPO Qwen3-4B on AIME 2024"
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime24_16k_grpo data.response_key=response

# echo "Evaluating GRPO Qwen3-4B on AIME 2025"
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime25_16k_grpo data.response_key=response

#### PR OnPolicy ####
# echo "Evaluating PR OnPolicy Qwen3-4B on Set1 16k"
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/set1_16k_pr_on-policy_ch0.26 data.response_key=response

# echo "Evaluating PR OnPolicy Qwen3-4B on AIME 2024"
python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime24_16k_pr_on-policy_ch0.26 data.response_key=response

# echo "Evaluating PR OnPolicy Qwen3-4B on AIME 2025"
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime25_16k_pr_on-policy_ch0.26 data.response_key=response

# #### GenRM ch0.28 ####
# echo "Evaluating GenRM Qwen3-4B on Set1 16k"
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/set1_16k_genrm_smooth_lp_ch0.28 data.response_key=response

# echo "Evaluating GenRM Qwen3-4B on AIME 2024"
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime24_16k_genrm_smooth_lp_ch0.28 data.response_key=response

# echo "Evaluating GenRM Qwen3-4B on AIME 2025"
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime25_16k_genrm_smooth_lp_ch0.28 data.response_key=response


#### PR-Delta ####
# echo "Evaluating PR-Delta Qwen3-4B on Set1 16k"
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/set1_16k_pr_delta data.response_key=response

# echo "Evaluating PR-Delta Qwen3-4B on AIME 2024"
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime24_16k_pr_delta data.response_key=response

echo "Evaluating PR-Delta Qwen3-4B on AIME 2025"
python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime25_16k_pr_delta data.response_key=response

#### PR-GPPO ####
# echo "Evaluating PR-GPPO Qwen3-4B on Set1 16k"
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/set1_16k_pr_gppo_fixed data.response_key=response

# echo "Evaluating PR-GPPO Qwen3-4B on AIME 2024"
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime24_16k_pr_gppo_fixed data.response_key=response

# echo "Evaluating PR-GPPO Qwen3-4B on AIME 2025"
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime25_16k_pr_gppo_fixed data.response_key=response