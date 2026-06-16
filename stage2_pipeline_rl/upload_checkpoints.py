#!/usr/bin/env python3
"""
Script to upload model checkpoints to HuggingFace Hub.

Usage:
    python upload_checkpoints.py \
        --checkpoint_dir results/prl-exp-stage2-grpo-16k/finetune/ \
        --hf_repo_id your-username/model-name \
        --main_checkpoint current
"""

import argparse
import os
from pathlib import Path
from typing import Optional
from huggingface_hub import HfApi, create_repo


def upload_checkpoints(
    checkpoint_dir: str,
    hf_repo_id: str,
    main_checkpoint: str = "current",
    repo_type: str = "model",
    private: bool = True,
    commit_message: Optional[str] = None,
):
    """
    Upload all checkpoints from a training run to HuggingFace Hub.

    Args:
        checkpoint_dir: Path to the finetune directory containing current/ and intermediate/ folders
        hf_repo_id: HuggingFace repository ID (e.g., "username/model-name")
        main_checkpoint: Path to the main checkpoint subfolder or "current" (default: "current")
        repo_type: Type of repository (default: "model")
        private: Whether to create a private repository (default: True)
        commit_message: Optional commit message for uploads
    """
    api = HfApi()
    checkpoint_path = Path(checkpoint_dir)

    if not checkpoint_path.exists():
        raise ValueError(f"Checkpoint directory not found: {checkpoint_dir}")

    # Create or get the repository
    print(f"Creating/accessing repository: {hf_repo_id}")
    try:
        create_repo(
            repo_id=hf_repo_id,
            repo_type=repo_type,
            private=private,
            exist_ok=True
        )
        print(f"✓ Repository ready: {hf_repo_id}")
    except Exception as e:
        print(f"Warning: Could not create repo (it may already exist): {e}")

    # Determine the main checkpoint path
    if main_checkpoint == "current":
        main_checkpoint_path = checkpoint_path / "current"
    else:
        main_checkpoint_path = Path(main_checkpoint)
        if not main_checkpoint_path.is_absolute():
            main_checkpoint_path = checkpoint_path / main_checkpoint_path

    if not main_checkpoint_path.exists():
        raise ValueError(f"Main checkpoint not found: {main_checkpoint_path}")

    # Upload main checkpoint to the default branch
    print(f"\n{'='*60}")
    print(f"Uploading main checkpoint from: {main_checkpoint_path}")
    print(f"{'='*60}")

    main_commit_msg = commit_message or f"Upload main checkpoint from {main_checkpoint_path.name}"
    api.upload_folder(
        folder_path=str(main_checkpoint_path),
        repo_id=hf_repo_id,
        repo_type=repo_type,
        commit_message=main_commit_msg,
    )
    print(f"✓ Main checkpoint uploaded to main branch")

    # Upload intermediate checkpoints as revisions
    intermediate_dir = checkpoint_path / "intermediate"
    if intermediate_dir.exists() and intermediate_dir.is_dir():
        print(f"\n{'='*60}")
        print(f"Uploading intermediate checkpoints from: {intermediate_dir}")
        print(f"{'='*60}")

        # Get all checkpoint folders sorted by step number
        checkpoint_folders = sorted(
            [d for d in intermediate_dir.iterdir() if d.is_dir()],
            key=lambda x: int(x.name) if x.name.isdigit() else 0
        )

        for checkpoint_folder in checkpoint_folders:
            step_name = checkpoint_folder.name
            revision_name = f"checkpoint-{step_name}"

            print(f"\nUploading checkpoint {step_name} to revision '{revision_name}'...")

            try:
                # Create branch first if it doesn't exist
                try:
                    api.create_branch(
                        repo_id=hf_repo_id,
                        branch=revision_name,
                        repo_type=repo_type,
                        exist_ok=True,
                    )
                except Exception as branch_err:
                    print(f"  Note: Branch creation returned: {branch_err}")

                checkpoint_commit_msg = commit_message or f"Upload checkpoint at step {step_name}"
                api.upload_folder(
                    folder_path=str(checkpoint_folder),
                    repo_id=hf_repo_id,
                    repo_type=repo_type,
                    revision=revision_name,
                    commit_message=checkpoint_commit_msg,
                    create_pr=False,
                )
                print(f"✓ Checkpoint {step_name} uploaded to revision '{revision_name}'")
            except Exception as e:
                print(f"✗ Failed to upload checkpoint {step_name}: {e}")
    else:
        print(f"\nNo intermediate checkpoints found at {intermediate_dir}")

    # Also upload current checkpoint as a revision if it's not already the main
    current_dir = checkpoint_path / "current"
    if current_dir.exists() and main_checkpoint_path != current_dir:
        print(f"\nUploading current checkpoint to revision 'checkpoint-current'...")
        try:
            # Create branch first if it doesn't exist
            try:
                api.create_branch(
                    repo_id=hf_repo_id,
                    branch="checkpoint-current",
                    repo_type=repo_type,
                    exist_ok=True,
                )
            except Exception as branch_err:
                print(f"  Note: Branch creation returned: {branch_err}")

            current_commit_msg = commit_message or "Upload current checkpoint"
            api.upload_folder(
                folder_path=str(current_dir),
                repo_id=hf_repo_id,
                repo_type=repo_type,
                revision="checkpoint-current",
                commit_message=current_commit_msg,
                create_pr=False,
            )
            print(f"✓ Current checkpoint uploaded to revision 'checkpoint-current'")
        except Exception as e:
            print(f"✗ Failed to upload current checkpoint: {e}")

    print(f"\n{'='*60}")
    print(f"✓ All uploads complete!")
    print(f"{'='*60}")
    print(f"\nRepository URL: https://huggingface.co/{hf_repo_id}")
    print(f"Main checkpoint: main branch")
    if intermediate_dir.exists():
        print(f"Intermediate checkpoints: Available as revisions (checkpoint-50, checkpoint-100, etc.)")
    print(f"\nTo load the main model:")
    print(f"  from transformers import AutoModelForCausalLM")
    print(f"  model = AutoModelForCausalLM.from_pretrained('{hf_repo_id}')")
    print(f"\nTo load a specific checkpoint:")
    print(f"  model = AutoModelForCausalLM.from_pretrained('{hf_repo_id}', revision='checkpoint-100')")


def main():
    parser = argparse.ArgumentParser(
        description="Upload model checkpoints to HuggingFace Hub",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload with current as main checkpoint
  python upload_checkpoints.py \\
      --checkpoint_dir results/prl-exp-stage2-grpo-16k/finetune/ \\
      --hf_repo_id username/my-model

  # Upload with specific checkpoint as main
  python upload_checkpoints.py \\
      --checkpoint_dir results/prl-exp-stage2-grpo-16k/finetune/ \\
      --hf_repo_id username/my-model \\
      --main_checkpoint intermediate/300
        """
    )

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="Path to the finetune directory containing current/ and intermediate/ folders"
    )
    parser.add_argument(
        "--hf_repo_id",
        type=str,
        required=True,
        help="HuggingFace repository ID (e.g., 'username/model-name')"
    )
    parser.add_argument(
        "--main_checkpoint",
        type=str,
        default="current",
        help="Path to main checkpoint subfolder or 'current' (default: current)"
    )
    parser.add_argument(
        "--commit_message",
        type=str,
        default=None,
        help="Custom commit message for uploads"
    )

    args = parser.parse_args()

    upload_checkpoints(
        checkpoint_dir=args.checkpoint_dir,
        hf_repo_id=args.hf_repo_id,
        main_checkpoint=args.main_checkpoint,
        commit_message=args.commit_message,
    )


if __name__ == "__main__":
    main()
