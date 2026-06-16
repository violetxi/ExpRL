import datasets
import pandas as pd

def compute_solve_rate(data_path):
    dataset = datasets.load_dataset(data_path, split="train")
    # add a `index` column by extract the `extra_info` column
    df = dataset.to_pandas()
    df["problem_index"] = df["extra_info"].apply(lambda x: x["index"])
    # compute the solve rate, return a df with: 
    # `data_source`, `prompt`, `reward_model`, `problem_index`, `extra_info` columns
    # and `solve_rate` column + responses columns (list of all response for the same problem_index)
    solve_rate_df = df.groupby("problem_index").agg(
        data_source=("data_source", "first"),
        prompt=("prompt", "first"),
        reward_model=("reward_model", "first"),
        extra_info=("extra_info", "first"),
        responses=("responses", list),
        scores=("score", list),
        solve_rate=("score", "mean"),  # assuming score is 0/1 binary, mean gives solve rate
    ).reset_index()
    
    # back to dataset
    ds = datasets.Dataset.from_pandas(solve_rate_df)
    ds.push_to_hub(f"{data_path}-grouped")
    return ds

if __name__ == "__main__":
    data_path = "violetxi/omni-rule-l7-above-gemini-pro-filtered_qwen3-4b-instruct-k32-score"
    ds = compute_solve_rate(data_path)
    print(f"Solve rate: {ds}")