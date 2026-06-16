import os
import argparse
from datasets import load_dataset


def save_hf_split_to_parquet(hf_dataset: str, save_dir: str, split: str):
    # Extract dataset name (after "/")
    dataset_name = hf_dataset.split("/")[-1]
    
    # Prepare save path
    output_path = os.path.join(save_dir, dataset_name)
    os.makedirs(output_path, exist_ok=True)

    # Load HF dataset
    print(f"📥 Loading dataset: {hf_dataset}, split: {split}")
    ds = load_dataset(hf_dataset, split=split)

    # Save parquet
    parquet_path = os.path.join(output_path, f"{split}.parquet")
    ds.to_parquet(parquet_path)

    print(f"✅ Saved parquet file to: {parquet_path}")


def main():
    parser = argparse.ArgumentParser(description="Save HF dataset split to Parquet")
    parser.add_argument("--hf_dataset", required=True, type=str,
                        help="HuggingFace dataset name, e.g., CohenQu/my_dataset")
    parser.add_argument("--save_dir", required=True, type=str,
                        help="Directory to store parquet output files")
    parser.add_argument("--split", required=True, type=str,
                        help="Dataset split to download, e.g., train, test, validation")

    args = parser.parse_args()

    save_hf_split_to_parquet(
        hf_dataset=args.hf_dataset,
        save_dir=args.save_dir,
        split=args.split
    )


if __name__ == "__main__":
    main()
