# AceReason dataset
python variational_inference/math/data_processing/base_curriculum.py \
    --data_source CohenQu/AceReason-Math-Qwen3-4B \
    --local_dir data/base/ace_reason_curriculum \
    --problem_key=problem \
    --answer_key=answer \
    --which_prompt=qwen3