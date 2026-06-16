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

import torch
from torch import nn
from copy import deepcopy

from transformers import LlamaModel, LlamaPreTrainedModel


class AtheneForSequenceClassification(LlamaPreTrainedModel):
    """Based on doc: https://huggingface.co/Nexusflow/Athene-RM-8B#usage"""
    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.v_head = nn.Linear(config.hidden_size, 1, bias=False)
        self.CLS_ID = 128003
        # Initialize weights and apply final processing
        self.post_init()

    def get_device(self):
        return self.model.device    


    def forward(
        self,
        input_ids=None,
        past_key_values=None,
        attention_mask=None,
        position_ids=None,
    ):        
        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True,
        )
        hidden_states = transformer_outputs.hidden_states[-1]        
        scores = []
        rewards = self.v_head(hidden_states).squeeze(-1)        

        bs = int(input_ids.shape[0])

        for i in range(bs):
            c_inds = (input_ids[i] == self.CLS_ID).nonzero()
            if len(c_inds) == 0:
                print(f"No CLS ID found in input_ids[i]: {input_ids[i]} of shape {input_ids.shape}")
                print("Defaulting to last valid token, which is the last token of the response")                
                scores.append(rewards[i, -1])
            else:
                c_ind = c_inds[-1].item()
                scores.append(rewards[i, c_ind])
        
        scores = torch.stack(scores)
        return {"scores": scores}

import torch
import numpy as np

from verl import DataProto
from verl.utils.reward_score import _default_compute_score
from verl.workers.reward_manager import register
@register("hybrid_reason")
class HybridReasonRewardManager:
    """The reward manager for the hybrid reasoning model. It will return a new DataProto with chosen and rejected response IDs,
    and whether a thinking penalty (0 or 1) is applied to the response. Each uid must have more than one response, and the response
    with the highest rm score will be chosen, and the one with the lowest rm score will be rejected.
    
    Args:
        tokenizer: The tokenizer to use for decoding the responses.
        num_examine: The number of responses to examine.
        compute_score: The function to compute the rewards.
        reward_fn_key: The key to use for the reward function.
    """

    def __init__(self, tokenizer, num_examine, compute_score=None, reward_fn_key="data_source", **reward_kwargs) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self.reward_fn_key = reward_fn_key
        self.reward_kwargs = reward_kwargs

    def __call__(self, data: DataProto, return_dict: bool = False):
        """Takes the data and returns "rejected" and "chosen" rewards for each uid during training (return_dict=False).
        And just return the rm score during validation (return_dict=True)."""
        # first get the rm scores
        reward_tensor = data.batch['rm_scores']
        if return_dict:
            return {"reward_tensor": reward_tensor}

        else:            
            already_print_data_sources = {}
            unique_uids = np.unique(data.non_tensor_batch['uid'])
            reasoning_penalty_coef = self.reward_kwargs.get("reasoning_penalty_coef", 1e-7)

            new_data_dict = {
                "input_ids": [],
                "attention_mask": [],
                "position_ids": [],                
                "chosen_labels": [],
                "chosen_attention_mask": [],
                "rejected_labels": [],
                "rejected_attention_mask": [],
                "chosen_position_ids": [],
                "rejected_position_ids": [],
                "chosen_reasoning_penalty": [],
                "rejected_reasoning_penalty": []
            }
            new_non_tensor_batch = {
                'uid': [],
                'domain': []
                }
            for uid in unique_uids:                
                # select the data with the same uid
                uid_data = data[data.non_tensor_batch['uid'] == uid]
                data_source = uid_data.non_tensor_batch['data_source'][0]                                
                new_non_tensor_batch['uid'].append(uid)
                new_non_tensor_batch['domain'].append(uid_data.non_tensor_batch['extra_info'][0]['domain'])

                # all prompts should be the same
                prompt_ids = uid_data.batch['prompts'][0]
                prompt_len = prompt_ids.shape[-1]
                attention_mask = uid_data.batch['attention_mask'][0, :prompt_len]
                position_ids = uid_data.batch['position_ids'][0, :prompt_len]
                prompt_str = self.tokenizer.decode(torch.masked_select(prompt_ids[0], attention_mask[0].bool()), skip_special_tokens=True)
                
                # get index of the data with the highest and lowest rm score
                chosen_idx = uid_data.batch['rm_scores'].argmax().item()
                rejected_idx = uid_data.batch['rm_scores'].argmin().item()

                # make new data with the chosen and rejected data
                chosen_data = uid_data[chosen_idx].batch                
                chosen_labels = chosen_data['responses']
                chosen_mask = chosen_data['attention_mask'][prompt_len:]
                chosen_position_ids = chosen_data['position_ids'][prompt_len:]
                chosen_str = self.tokenizer.decode(torch.masked_select(chosen_labels, chosen_mask.bool()), skip_special_tokens=True).lstrip()
                chosen_has_reasoning = int(chosen_str.lstrip().startswith("<think>"))
                chosen_reasoning_penalty = torch.tensor(reasoning_penalty_coef * chosen_has_reasoning, dtype=torch.float32)

                rejected_data = uid_data[rejected_idx].batch
                rejected_labels = rejected_data['responses']
                rejected_mask = rejected_data['attention_mask'][prompt_len:]
                rejected_position_ids = rejected_data['position_ids'][prompt_len:]
                rejected_str = self.tokenizer.decode(torch.masked_select(rejected_labels, rejected_mask.bool()), skip_special_tokens=True).lstrip()
                rejected_has_reasoning = int(rejected_str.lstrip().startswith("<think>"))
                rejected_reasoning_penalty = torch.tensor(reasoning_penalty_coef * rejected_has_reasoning, dtype=torch.float32)

                if data_source not in already_print_data_sources:
                    already_print_data_sources[data_source] = 0

                if already_print_data_sources[data_source] < self.num_examine:
                    already_print_data_sources[data_source] += 1
                    print("[prompt]", prompt_str)
                    print("[chosen]", chosen_str)
                    print("[rejected]", rejected_str)

                new_data_dict["input_ids"].append(prompt_ids)
                new_data_dict["attention_mask"].append(attention_mask)
                new_data_dict["position_ids"].append(position_ids)
                new_data_dict["chosen_labels"].append(chosen_labels)
                new_data_dict["rejected_labels"].append(rejected_labels)
                new_data_dict["chosen_attention_mask"].append(chosen_mask)
                new_data_dict["rejected_attention_mask"].append(rejected_mask)
                new_data_dict["chosen_position_ids"].append(chosen_position_ids)
                new_data_dict["rejected_position_ids"].append(rejected_position_ids)                
                new_data_dict["chosen_reasoning_penalty"].append(chosen_reasoning_penalty)
                new_data_dict["rejected_reasoning_penalty"].append(rejected_reasoning_penalty)                

            for k, v in new_data_dict.items():                
                new_data_dict[k] = torch.stack(v, dim=0)

            new_data = DataProto.from_single_dict(data=new_data_dict, meta_info=data.meta_info)
            for k, v in new_non_tensor_batch.items():
                new_non_tensor_batch[k] = np.array(v)
                
            new_data.non_tensor_batch = new_non_tensor_batch
            new_data.meta_info = data.meta_info
            return new_data