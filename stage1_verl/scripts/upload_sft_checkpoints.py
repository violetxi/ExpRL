import re
import os
import sys
import argparse
from huggingface_hub import HfApi

def upload_local_checkpoint(checkpoint_folder, repo_id, main_checkpoint):
    api = HfApi()
    checkpoint_paths = []
    checkpoint_folders = [f for f in os.listdir(checkpoint_folder) if os.path.isdir(os.path.join(checkpoint_folder, f))]    

    for folder in checkpoint_folders:
        path = os.path.join(checkpoint_folder, folder)
        # path = os.path.join(checkpoint_folder, folder)
        checkpoint_paths.append(path)
        
    if not main_checkpoint:
        # If no main checkpoint specified, use the first checkpoint
        def extract_step_number(path):
            # Extract step number from path like "global_step_xxx"
            parent_dir = os.path.basename(os.path.dirname(path))
            match = re.search(r'global_step_(\d+)', parent_dir)
            return int(match.group(1)) if match else 0

        main_checkpoint = max(checkpoint_paths, key=extract_step_number)
        print(f"No main checkpoint specified. Using latest checkpoint: {main_checkpoint}")
    
    for path in checkpoint_paths:
        branch = path.split("checkpoints/")[1].split("/")[-1]        
        api.create_repo(repo_id, exist_ok=True, private=False)
        # ignore data.pt file
        api.create_branch(repo_id=repo_id, branch=branch, exist_ok=True)
        
        if path == main_checkpoint or path == main_checkpoint[0]:            
            api.upload_folder(folder_path=path, repo_id=repo_id, revision="main", ignore_patterns=["data.pt"])
            print(f"Uploaded {path} as main branch to {repo_id}/main")
        else:
             api.upload_folder(folder_path=path, repo_id=repo_id, revision=branch, ignore_patterns=["data.pt"])
        

if __name__ == "__main__":
    # Get the command-line arguments
    parser = argparse.ArgumentParser(description="Process and upload multiple checkpoints to HuggingFace Hub")
    parser.add_argument('--parent_dir', required=True, type=str, 
                        help="Parent directory containing checkpoint directories")
    parser.add_argument('--hf_repo_id', required=True, type=str, 
                        help="HuggingFace repository ID (e.g., 'organization/model-name')")
    parser.add_argument('--main_checkpoint', required=False, type=str, 
                        help="Path to the checkpoint that should be set as the main branch")
    parser.add_argument('--temp_dir', type=str, default="./tmp_hf_models",
                        help="Temporary directory for processed models")
    args = parser.parse_args()
    upload_local_checkpoint(args.parent_dir, args.hf_repo_id, args.main_checkpoint)
