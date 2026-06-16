# echo "Evaluating Qwen4B-rloo-aime24"
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/test-qwen4b-rloo-aime24

# echo "Evaluating Qwen4B-rloo-vi-box-aime24"
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/test-qwen4b-rloo-vi-box-aime24

# echo "Evaluating Qwen4B-rloo-aime25"
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/test-qwen4b-rloo-aime25

# echo "Evaluating Qwen4B-rloo-vi-box-aime25"
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/test-qwen4b-rloo-vi-box-aime25

# echo "Evaluating Qwen4B-rloo-olympiadbench"
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/test-qwen4b-rloo-olympiadbench

# echo "Evaluating Qwen4B-rloo-vi-box-olympiadbench"
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/test-qwen4b-rloo-vi-box-olympiadbench

# echo "Evaluating Qwen4B-instruct-ace_rl_4b_inst_hard"
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/test-qwen4b-instruct-ace_rl_4b_inst_hard

# echo "Evaluating Qwen4B-rloo-vi-box-pope"
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/test-qwen4b-rloo-vi-box-pope

# # echo "Evaluating Qwen4B-rloo-vi-box-pope"
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/qwen4b-thinking-omni-l1_2

#### Cohen GRPO Qwen3-4B
# echo "Evaluating Cohen GRPO Qwen3-4B on POPE Hard"
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/test_pope_hard_cohen_qwen4b_grpo

# echo "Evaluating Cohen GRPO Qwen3-4B on AMC"
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/test_amc_cohen_qwen4b_grpo

# echo "Evaluating Cohen GRPO Qwen3-4B on AIME 2024"
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/test_aime24_cohen_qwen4b_grpo

# echo "Evaluating Cohen GRPO Qwen3-4B on AIME 2025"
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/test_aime25_cohen_qwen4b_grpo

# #### Cohen GemRM Qwen3-4B
# echo "Evaluating Cohen GemRM Qwen3-4B on POPE Hard"
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/test_pope_hard_cohen_qwen4b_genrm-smooth

# echo "Evaluating Cohen GemRM Qwen3-4B on AMC"
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/test_amc_cohen_qwen4b_genrm-smooth

# echo "Evaluating Cohen GemRM Qwen3-4B on AIME 2024"
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/test_aime24_cohen_qwen4b_genrm-smooth

# echo "Evaluating Cohen GemRM Qwen3-4B on AIME 2025"
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/test_aime25_cohen_qwen4b_genrm-smooth

# #### Cohen Rand Qwen3-4B
# echo "Evaluating Cohen Rand Qwen3-4B on AIME 2024"
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/test_aime24_cohen_qwen4b_rand

# echo "Evaluating Cohen Rand Qwen3-4B on AIME 2025"
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/test_aime25_cohen_qwen4b_rand

# #### Cohen PR Delta Qwen3-4B
# echo "Evaluating Cohen PR Delta Qwen3-4B on AIME 2024"
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/test_aime24_cohen_qwen4b_pr-smooth-delta

# echo "Evaluating Cohen PR Delta Qwen3-4B on AIME 2025"
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/test_aime25_cohen_qwen4b_pr-smooth-delta

# #### Qwen4B-Instruct-2507
# echo "Evaluating Qwen4B-Instruct-2507 on POPE Hard"
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/test_pope_hard_qwen4b_instruct_2507

# echo "Evaluating Qwen4B-Instruct-2507 with PR GPPO on POPE Hard"
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/test_pope_hard_qwen4b-instruct-2507_w_hint_pr_gppo

# echo "Evaluating Qwen4B-Instruct-2507 with PR GPPO on AIME 2024"
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/test_aime24_qwen4b-instruct-2507_w_hint_pr_gppo

# echo "Evaluating Qwen4B-Instruct-2507 with PR GPPO on AIME 2025"
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/test_aime25_qwen4b-instruct-2507_w_hint_pr_gppo

# echo "Evaluating Qwen4B-Instruct-2507 with GenRM Outcome on POPE Hard"
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/test_pope_hard_qwen4b-instruct-2507_w_hint_genrm_smooth

# echo "Evaluating Qwen4B-Instruct-2507 with GenRM Outcome on AIME 2024"
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/test_aime24_qwen4b-instruct-2507_w_hint_genrm_smooth

# echo "Evaluating Qwen4B-Instruct-2507 with GenRM Outcome on AIME 2025"
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/test_aime25_qwen4b-instruct-2507_w_hint_genrm_smooth

# echo "Evaluating Qwen4B-Instruct-2507 with GPPO on AIME 2024"
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/test_aime24_qwen4b-instruct-2507_w_hint_grpo

# echo "Evaluating Qwen4B-Instruct-2507 with GPPO on AIME 2025"
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/test_aime25_qwen4b-instruct-2507_w_hint_grpo

echo "Evaluating Qwen4B-Instruct-2507 with GPPO on POPE Hard"
python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/test_pope_hard_qwen4b-instruct-2507_w_hint_grpo