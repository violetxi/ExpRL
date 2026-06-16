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

import datasets
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial


def load_dataset_subset(data_source, subset_name):
    """Helper function to load a single dataset subset."""
    try:
        return datasets.load_dataset(data_source, subset_name, split='train')
    except Exception as e:
        print(f"Failed to load subset {subset_name}: {e}")
        return None


def load_subsets_parallel(data_source, subset_names, max_workers=32):
    """
    Load multiple dataset subsets in parallel for faster downloading.
    
    Args:
        data_source: The dataset source identifier
        subset_names: List of subset names to load
        max_workers: Maximum number of parallel workers (default: 32)
    
    Returns:
        List of loaded datasets (excluding any that failed to load)
    """
    print(f"Loading {len(subset_names)} subsets in parallel with {max_workers} workers...")
    
    all_subsets = []
    load_fn = partial(load_dataset_subset, data_source)
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_subset = {executor.submit(load_fn, subset_name): subset_name 
                           for subset_name in subset_names}
                
        for i, future in enumerate(as_completed(future_to_subset)):
            subset_name = future_to_subset[future]
            try:
                result = future.result()
                if result is not None:
                    all_subsets.append(result)
                    print(f"Loaded subset {i+1}/{len(subset_names)}: {subset_name}")
                else:
                    print(f"Skipped failed subset: {subset_name}")
            except Exception as e:
                print(f"Exception loading subset {subset_name}: {e}")
    
    print(f"Successfully loaded {len(all_subsets)} out of {len(subset_names)} subsets")
    return all_subsets

