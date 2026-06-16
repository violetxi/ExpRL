python variational_inference/math/data_processing/gen_sft_thoughts.py \
    --data_source ${HF_NAMESPACE:-violetxi}/omni-math-difficulty-1_2\
    --local_dir data/sft_thoughts/omni-math-difficulty-1_2 \
    --problem_key=problem \
    --answer_key=answer \
    --solution_key=solution \
    --which_prompt=basic

python variational_inference/math/data_processing/gen_sft_thoughts.py \
    --data_source ${HF_NAMESPACE:-violetxi}/omni-math-difficulty-2_3 \
    --local_dir data/sft_thoughts/omni-math-difficulty-2_3 \
    --problem_key=problem \
    --answer_key=answer \
    --solution_key=solution \
    --which_prompt=basic