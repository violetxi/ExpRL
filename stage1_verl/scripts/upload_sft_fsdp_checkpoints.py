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

from typing import List, Tuple, Dict, Optional
import re
import os
import torch
import argparse
from copy import deepcopy
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForTokenClassification, AutoModelForVision2Seq
from concurrent.futures import ThreadPoolExecutor
from torch.distributed._tensor import DTensor, Shard, Placement
from huggingface_hub import HfApi
import shutil
from torch.distributed._tensor import DTensor  
# …and attach it under the name torch.distributed.tensor.DTensor
import torch.distributed.tensor  
setattr(torch.distributed.tensor, "DTensor", DTensor)  


def merge_by_placement(tensors: List[torch.Tensor], placement: Placement):
    if placement.is_replicate():
        return tensors[0]
    elif placement.is_partial():
        raise NotImplementedError("Partial placement is not supported yet")
    elif placement.is_shard():
        return torch.cat(tensors, dim=placement.dim).contiguous()
    else:
        raise ValueError(f"Unsupported placement: {placement}")


def process_checkpoint(local_dir: str, output_dir: str) -> None:
    """Process a single checkpoint directory and save the merged model to output_dir."""
    # Find the world size and rank 0 file
    world_size = 0
    for filename in os.listdir(local_dir):
        match = re.match(r"model_world_size_(\d+)_rank_0\.pt", filename)
        if match:
            world_size = match.group(1)
            break
    
    if not world_size:
        raise ValueError(f"No model file with the proper format found in {local_dir}")
    
    rank = 0    
    state_dict = torch.load(os.path.join(local_dir, f'model_world_size_{world_size}_rank_{rank}.pt'), map_location='cpu')
    pivot_key = sorted(list(state_dict.keys()))[0]
    weight = state_dict[pivot_key]
    
    if not isinstance(weight, torch.distributed._tensor.DTensor):
        raise TypeError("Expected DTensor, but found different type")
    
    # Get sharding info
    device_mesh = weight.device_mesh
    mesh = device_mesh.mesh
    mesh_dim_names = device_mesh.mesh_dim_names
    original_mesh_dim_names = deepcopy(mesh_dim_names)

    # allow for a leading DDP axis by stripping it off
    if mesh_dim_names[0] in ("ddp", "dp"):
        mesh_dim_names = mesh_dim_names[1:]     # now ('fsdp',)
    

    print(f'Got device mesh {mesh}, mesh_dim_names {mesh_dim_names}')

    # assert mesh_dim_names in (
    #     ('fsdp',),
    # ), f'Unsupported mesh_dim_names {mesh_dim_names}'

    # we now allow either plain fsdp or a leading ddp/dp axis
    if mesh_dim_names not in (('fsdp',),):
        raise ValueError(f'Unsupported mesh_dim_names {mesh_dim_names}')
    

    if 'tp' in mesh_dim_names:
        # fsdp * tp
        total_shards = mesh.shape[-1] * mesh.shape[-2]
        mesh_shape = (mesh.shape[-2], mesh.shape[-1])
    else:
        # fsdp
        total_shards = mesh.shape[-1]
        mesh_shape = (mesh.shape[-1],)

    print(f'Processing model shards with {total_shards} {mesh_shape} in total')

    # Load all model shards
    model_state_dict_lst = []
    model_state_dict_lst.append(state_dict)
    model_state_dict_lst.extend([""] * (total_shards - 1))

    def process_one_shard(rank):
        model_path = os.path.join(local_dir, f'model_world_size_{world_size}_rank_{rank}.pt')
        state_dict = torch.load(model_path, map_location='cpu', weights_only=False)
        model_state_dict_lst[rank] = state_dict
        return state_dict

    with ThreadPoolExecutor(max_workers=min(32, os.cpu_count())) as executor:
        for rank in range(1, total_shards):
            executor.submit(process_one_shard, rank)
    
    # Merge state dictionary
    state_dict = {}
    param_placements: Dict[str, List[Placement]] = {}
    keys = set(model_state_dict_lst[0].keys())
    
    for key in keys:
        state_dict[key] = []
        for model_state_dict in model_state_dict_lst:
            try:
                tensor = model_state_dict.pop(key)
            except:
                print("-"*30)
                print(model_state_dict)
            
            if isinstance(tensor, DTensor):
                state_dict[key].append(tensor._local_tensor.bfloat16())
                placements = tuple(tensor.placements)
                # # replicated placement at dp dimension can be discarded
                # if mesh_dim_names[0] == 'dp':
                #   placements = placements[1:]
                # replicated placement at dp or ddp dimension can be discarded
                if original_mesh_dim_names[0] in ('dp','ddp'):
                    placements = placements[1:]
                if key not in param_placements:
                    param_placements[key] = placements
                else:
                    assert param_placements[key] == placements
            else:
                state_dict[key] = tensor.bfloat16()

    del model_state_dict_lst

    # Merge shards
    for key in sorted(state_dict):
        if not isinstance(state_dict[key], list):
            print(f"No need to merge key {key}")
            continue
        
        placements: Tuple[Shard] = param_placements[key]
        if len(mesh_shape) == 1:
            # 1-D list, FSDP without TP
            assert len(placements) == 1
            shards = state_dict[key]
            state_dict[key] = merge_by_placement(shards, placements[0])
        else:
            # 2-D list, FSDP + TP
            raise NotImplementedError("FSDP + TP is not supported yet")

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find and copy the HuggingFace config directory
    hf_path = os.path.join(local_dir, 'huggingface')
    if not os.path.exists(hf_path):
        # Look for the HuggingFace config in the parent directory
        parent_hf_path = os.path.join(os.path.dirname(local_dir), 'huggingface')
        if os.path.exists(parent_hf_path):
            hf_path = parent_hf_path
        else:
            sibling_hf_path = None
            for sibling_name in sorted(os.listdir(os.path.dirname(local_dir))):
                candidate = os.path.join(os.path.dirname(local_dir), sibling_name, 'huggingface')
                if os.path.isdir(candidate):
                    sibling_hf_path = candidate
                    break
            if sibling_hf_path:
                hf_path = sibling_hf_path
            else:
                raise FileNotFoundError(f"HuggingFace config not found in {hf_path}, parent directory, or sibling checkpoints")
    
    # Copy HuggingFace config files to output directory
    for item in os.listdir(hf_path):
        src = os.path.join(hf_path, item)
        dst = os.path.join(output_dir, item)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
        elif os.path.isdir(src):
            shutil.copytree(src, dst, dirs_exist_ok=True)
    
    # Load config and create model
    config = AutoConfig.from_pretrained(output_dir)
    
    if 'ForTokenClassification' in config.architectures[0]:
        auto_model = AutoModelForTokenClassification
    elif 'ForCausalLM' in config.architectures[0]:
        auto_model = AutoModelForCausalLM
    elif 'ForConditionalGeneration' in config.architectures[0]:
        auto_model = AutoModelForVision2Seq
    else:
        raise NotImplementedError(f'Unknown architecture {config.architectures}')

    with torch.device('meta'):
        model = auto_model.from_config(config, torch_dtype=torch.bfloat16)
    model.to_empty(device='cpu')

    print(f'Saving model to {output_dir}')
    model.save_pretrained(output_dir, state_dict=state_dict)
    return output_dir


def upload_to_huggingface(repo_id: str, folder_path: str, branch: str = "main") -> None:
    """Upload a folder to HuggingFace Hub."""
    api = HfApi()
    api.create_repo(repo_id=repo_id, private=False, exist_ok=True)
    
    if branch != "main":
        api.create_branch(repo_id=repo_id, branch=branch, exist_ok=True)
        
    print(f"Uploading {folder_path} to {repo_id}:{branch}")
    api.upload_folder(
        folder_path=folder_path,
        repo_id=repo_id,
        revision=branch)


def get_branch_name(checkpoint_path: str) -> str:
    """Extract branch name from checkpoint path."""
    # Get the directory name as the branch name    
    return os.path.normpath(checkpoint_path).split(os.sep)[-1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process and upload multiple checkpoints to HuggingFace Hub")
    parser.add_argument('--parent_dir', required=True, type=str, 
                        help="Parent directory containing checkpoint directories")
    parser.add_argument('--hf_repo_id', required=True, type=str, 
                        help="HuggingFace repository ID (e.g., 'organization/model-name')")
    parser.add_argument('--main_checkpoint', required=True, type=str, 
                        help="Path to the checkpoint that should be set as the main branch")
    parser.add_argument('--temp_dir', type=str, default="./tmp_hf_models",
                        help="Temporary directory for processed models")    
    
    args = parser.parse_args()
    
    # Create temp directory if it doesn't exist
    os.makedirs(args.temp_dir, exist_ok=True)            
    checkpoint_paths = [os.path.join(args.parent_dir, checkpoint_dir) 
                        for checkpoint_dir in os.listdir(args.parent_dir)]
    # remove paths that are not directories
    checkpoint_paths = [path for path in checkpoint_paths if os.path.isdir(path)]

    if not checkpoint_paths:
        print(f"No checkpoint directories found in {args.parent_dir} ")
        exit(1)
    
    print(f"Found {len(checkpoint_paths)} checkpoint directories")    

    # If no main checkpoint specified, use the latest checkpoint
    if not args.main_checkpoint:
        args.main_checkpoint = max(checkpoint_paths, key=os.path.getmtime)        
        print(f"No main checkpoint specified. Using latest checkpoint: {args.main_checkpoint}")
    elif not os.path.exists(args.main_checkpoint):
        print(f"Specified main checkpoint {args.main_checkpoint} does not exist")
        exit(1)

    if args.main_checkpoint.endswith('/'):
        args.main_checkpoint = args.main_checkpoint[:-1]
        
    # Process and upload each checkpoint
    for checkpoint_path in checkpoint_paths:
        branch_name = get_branch_name(checkpoint_path)
        temp_output_dir = os.path.join(args.temp_dir, branch_name)        
        try:
            print(f"Processing checkpoint {checkpoint_path}")
            processed_dir = process_checkpoint(checkpoint_path, temp_output_dir)
            
            # Determine branch (main or checkpoint name)
            branch = "main" if checkpoint_path == args.main_checkpoint else branch_name            
            # Upload to HuggingFace
            upload_to_huggingface(args.hf_repo_id, processed_dir, branch)
            print(f"Successfully uploaded {checkpoint_path} to {args.hf_repo_id}:{branch}")
            
        except Exception as e:
            raise Exception(f"Error processing checkpoint {checkpoint_path}: {e}")            
            
        finally:
            # Clean up temp directory to save space
            if os.path.exists(temp_output_dir):
                shutil.rmtree(temp_output_dir)
    
    print("All checkpoints processed and uploaded successfully!")
