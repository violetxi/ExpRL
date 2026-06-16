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

from collections import defaultdict

import torch

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register


@register("math_format")
class MathFormatRewardManager:
    """The reward manager."""

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", **reward_kwargs) -> None:
        """
        Initialize the MathFormatRewardManager instance. Check the format_type in reward_kwargs.

        Args:
            tokenizer: The tokenizer used to decode token IDs into text.
            num_examine: The number of batches of decoded responses to print to the console for debugging purpose.
            compute_score: A function to compute the reward score. If None, `default_compute_score` will be used.
            reward_fn_key: The key used to access the data source in the non-tensor batch data. Defaults to
                "data_source".
        """
        self.tokenizer = tokenizer  # Store the tokenizer for decoding token IDs
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.reward_fn_key = reward_fn_key  # Store the key for accessing the data source
        self.format_type = reward_kwargs.get("format_type", "box")
        self.outcome_reward_weight = reward_kwargs.get("outcome_reward_weight", 1)
        self.weight_format = reward_kwargs.get("weight_format", 1)        

    def _check_format(self, response_str, format_type):
        """
        Check if response has the correct format        
        """

        if format_type == "think":
            begin_tag = "<think>"
            end_tag = "</think>"

            response_str = response_str.strip().lstrip()        
            begin_with = response_str.startswith(begin_tag)
            count_begin = response_str.count(begin_tag)
            count_end = response_str.count(end_tag)

            if begin_with and count_begin == count_end == 1:
                return 1
            else:
                return 0

        elif format_type == "response":
            if response_str.strip().endswith("<R>"):
                return 1
            else:
                return 0
        elif format_type == "box":
            if "\\boxed{" in response_str[-100:]:
                return 1
            else:
                return 0
        elif format_type == "answer":
            if "<answer>" in response_str[-100:] and "</answer>" in response_str[-100:]:
                return 1
            else:
                return 0
        else:
            raise ValueError(f"Invalid format type: {format_type}")

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)

        already_print_data_sources = {}
        format_score_lst = []

        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem

            prompt_ids = data_item.batch["prompts"]

            prompt_length = prompt_ids.shape[-1]

            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]

            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            # decode
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = data_item.non_tensor_batch.get("extra_info", {})
            num_turns = data_item.non_tensor_batch.get("__num_turns__", None)
            extra_info["num_turns"] = num_turns

            score = self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
            )
            # check if response is formatted correctly, if not, set format_score to 0
            format_score = self._check_format(response_str, format_type=self.format_type)
            if format_score:
                format_score_lst.append(True)
            else:
                format_score_lst.append(False)
                        
            # both format and answer need to be correct to get reward
            if isinstance(score, dict):
                reward_extra_info["outcome_score"].append(score["score"])
                reward = score["score"]
                # Store the information including original reward
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward_extra_info["outcome_score"].append(score)
                reward = score

            reward_extra_info["format_score"].append(format_score)
            reward_tensor[i, valid_response_length - 1] = reward

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(score, dict):
                    for key, value in score.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)        

        reward_extra_info["format_score"] = format_score_lst
        if return_dict:            
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }            
        else:
            return reward_tensor
