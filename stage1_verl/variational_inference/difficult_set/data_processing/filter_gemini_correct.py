import datasets

def filter_gemini_correct(data_path):
    ds = datasets.load_dataset(data_path)
    ds = ds.filter(lambda x: x["score"])
    # remove the `score` column
    ds = ds.remove_columns("score")
    upload_path = data_path.replace("-sol-score", "-filtered")
    ds.push_to_hub(upload_path, private=False)

if __name__ == "__main__":
    data_path = "violetxi/omni-math-above-l7-rule-gemini-pro-sol-score"
    filter_gemini_correct(data_path)
