# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import List, Tuple, Dict
import re
import os
import torch
import argparse
import numpy as np
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForTokenClassification, AutoModelForVision2Seq
from concurrent.futures import ThreadPoolExecutor
from safetensors.torch import load_file
from torch.distributed._tensor import Shard, Placement
try:
    # for torch 2.5+
    from torch.distributed.tensor import DTensor
except ImportError:
    from torch.distributed._tensor import DTensor

parser = argparse.ArgumentParser()
parser.add_argument('--hf_upload_path', type=str, required=True, help="The path for the huggingface model")
parser.add_argument(
    '--local_dir',
    type=str,
    required=True,
    help=
    "The path for your saved model. For megatron, point to the base dir of model, rng, optimizer checkpoints, commonly be `config.default_local_dir/global_step_\{global_step\}`."
)
parser.add_argument('--revision_tag', default="main", type=str, help="Revision tag")

args = parser.parse_args()

def upload_model_to_huggingface(hf_path, hf_upload_path, revision_tag):
    # Push to hugging face
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo_id=hf_upload_path, private=False, exist_ok=True)
    commit_info = api.upload_folder(folder_path=hf_path, repo_id=hf_upload_path, repo_type="model")
    api.create_tag(
            repo_id=hf_upload_path,
            tag=revision_tag,
            revision=commit_info.oid,  # fallback to branch if oid missing
        )

if __name__ == '__main__':

    upload_model_to_huggingface(args.local_dir, args.hf_upload_path, args.revision_tag)