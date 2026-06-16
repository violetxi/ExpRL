
"""
For RFT data filtering
"""
import argparse
import datasets


# args = argparse.ArgumentParser()
# args.add_argument("--data_source", type=str, required=True)
# args = args.parse_args()

def filter_data_correct(data_path):    
    ds = datasets.load_dataset(data_path)
    ds = ds.filter(lambda x: x["score"])
    ds.push_to_hub(data_path, private=False)

if __name__ == "__main__":
    # data_path = "violetxi/qwen4b-thinking-omni-l1_2-score"
    # filter_data_correct(data_path)
    # data_path = "violetxi/qwen4b-thinking-omni-l2_3-score"
    # filter_data_correct(data_path)
    # data_path = "violetxi/qwen4b-thinking-omni-l1_2-score"
    # filter_data_correct(data_path)
    # data_path = "violetxi/qwen4b-thinking-omni-l2_3-score"
    # filter_data_correct(data_path)
    # data_path = "violetxi/qwen4b-thinking-omni-l4-score"
    # filter_data_correct(data_path)
    data_path = "violetxi/test-qwen8b-rewrite-omni-l4-score"
    filter_data_correct(data_path)
