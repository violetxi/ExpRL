import argparse
import datasets
import pandas as pd


""" usage: 
python variational_inference/gen_rm/compute_step_score.py --rollout_score_path violetxi/process_rollout-qwen4b-omni-l5-score --judge_score_path violetxi/Qwen4b-Judge-partial-qwen4b-no-thinking-omni-l5
python variational_inference/gen_rm/compute_step_score.py --rollout_score_path violetxi/process_rollout-qwen4b-omni-l5-score --judge_score_path violetxi/Qwen8b-Judge-partial-qwen4b-no-thinking-omni-l5
python variational_inference/gen_rm/compute_step_score.py --rollout_score_path violetxi/process_rollout-qwen4b-omni-l5-score --judge_score_path violetxi/Qwen14b-Judge-partial-qwen4b-no-thinking-omni-l5
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout_score_path", type=str, required=True)
    parser.add_argument("--judge_score_path", type=str, required=True)    
    args = parser.parse_args()

    # get rollout score
    ds_rollout = datasets.load_dataset(args.rollout_score_path, split="train")        
    df_rollout = ds_rollout.to_pandas()    
    grouped_rollout = df_rollout.groupby(['problem_index', 'step_index'], as_index=False)['score'].mean()
    
    ds_judge = datasets.load_dataset(args.judge_score_path, split="train")
    df_judge = ds_judge.to_pandas()    
        
    df_judge_merged = df_judge.merge(
        grouped_rollout.rename(columns={'score': 'rollout_score@16'}),
        on=['problem_index', 'step_index'],
        how='left'
    )
        
    new_ds_judge = datasets.Dataset.from_pandas(df_judge_merged, preserve_index=False)
    output_path = f"{args.judge_score_path}-step-score"
    new_ds_judge.push_to_hub(output_path)
    print(f"Saved {len(new_ds_judge)} rows to {output_path}")