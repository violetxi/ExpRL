echo "Evaluating Cohen GemRM Qwen3-4B on Cohen Test (global_step_30)"
python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/test_cohen_qwen4b_genrm-smooth-delta_30

echo "Evaluating Cohen GemRM Qwen3-4B on Cohen Test (global_step_60)"
python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/test_cohen_qwen4b_genrm-smooth-delta_60

echo "Evaluating Cohen GemRM Qwen3-4B on Cohen Test (global_step_90)"
python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/test_cohen_qwen4b_genrm-smooth-delta_90

echo "Evaluating Cohen GemRM Qwen3-4B on Cohen Test (global_step_120)"
python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/test_cohen_qwen4b_genrm-smooth-delta_120

echo "Evaluating Cohen GemRM Qwen3-4B on Cohen Test (main)"
python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/test_cohen_qwen4b_genrm-smooth-delta_main

echo "Evaluating Omni L7 Above on Cohen Test"
python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/process_rollout-qwen4b-instruct-2507-omni-l7