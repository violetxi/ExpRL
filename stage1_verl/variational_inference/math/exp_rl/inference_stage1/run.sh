# echo "Running HMMT Self Distill inference"
# bash variational_inference/math/exp_rl/inference_stage1/hmmt/self_distill.sh
# echo "HMMT Self Distill inference completed"
# echo "--------------------------------"

# echo "Running Pope Self Distill inference"
# bash variational_inference/math/exp_rl/inference_stage1/pope/self_distill.sh
# echo "Pope Self Distill inference completed"
# echo "--------------------------------"

# echo "Running IMO Answer Self Distill inference"
# bash variational_inference/math/exp_rl/inference_stage1/imo_answer/self_distill.sh
# echo "IMO Answer Self Distill inference completed"
# echo "--------------------------------"

# Dense Process
echo "Running AIME 2025 Dense Process inference"
bash variational_inference/math/exp_rl/inference_stage1/aime_25/pr_delta_process.sh
echo "AIME 2025 Dense Process inference completed"
echo "--------------------------------"

echo "Running AIME 2026 Dense Process inference"
bash variational_inference/math/exp_rl/inference_stage1/aime_26/pr_delta_process.sh
echo "AIME 2026 Dense Process inference completed"
echo "--------------------------------"

echo "Running HMMT PR Delta Process inference"
bash variational_inference/math/exp_rl/inference_stage1/hmmt/pr_delta_process.sh
echo "HMMT PR Delta Process inference completed"
echo "--------------------------------"

echo "Running Pope PR Delta Process inference"
bash variational_inference/math/exp_rl/inference_stage1/pope/pr_delta_process.sh
echo "Pope PR Delta Process inference completed"
echo "--------------------------------"

# echo "Running IMO Answer PR Delta Process inference"
# bash variational_inference/math/exp_rl/inference_stage1/imo_answer/pr_delta_process.sh
# echo "IMO Answer PR Delta Process inference completed"
# echo "--------------------------------"