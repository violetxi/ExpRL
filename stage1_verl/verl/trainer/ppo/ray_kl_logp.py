"""
KL Analysis: Compute student/teacher logprobs using verl's distributed compute_log_prob.

Copied from ray_logp.py, stripped to only compute logprobs on pre-existing
filtered data (from prepare_kl_data.py) rather than generating new rollouts.
"""

import json
import os
import pickle
from copy import deepcopy
from typing import Optional

import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Sampler
from tqdm import tqdm

from verl import DataProto
from verl.single_controller.ray import RayClassWithInitArgs, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo.ray_trainer import Role, ResourcePoolManager
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.model import compute_position_id_with_mask

import pandas as pd


class RayKLLogp:
    """Compute student/teacher logprobs for KL analysis using verl's distributed infrastructure."""

    def __init__(self, config, tokenizer, role_worker_mapping, resource_pool_manager,
                 ray_worker_group_cls=RayWorkerGroup, data_path=None, output_path=None,
                 solution_format="assistant_prefix", batch_size=32):
        self.config = config
        self.tokenizer = tokenizer
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.ray_worker_group_cls = ray_worker_group_cls
        self.data_path = data_path
        self.output_path = output_path
        self.solution_format = solution_format
        self.batch_size = batch_size
        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine

    def init_workers(self):
        """Initialize actor_rollout worker group (no critic, no RM needed)."""
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
        actor_rollout_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.ActorRollout],
            config=self.config.actor_rollout_ref,
            role="actor_rollout",
        )
        self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls

        all_wg = {}
        wg_kwargs = {}
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        wg_kwargs["device_name"] = "cuda"

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

    def _build_student_prompt(self, row) -> str:
        """Build student prompt: just the problem."""
        messages = row["prompt"]
        if isinstance(messages, str):
            messages = json.loads(messages)
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def _build_teacher_prompt(self, row) -> str:
        """Build teacher prompt: problem + solution."""
        messages = row["prompt"]
        if isinstance(messages, str):
            messages = json.loads(messages)

        solution = row["solution"]

        if self.solution_format == "user_append":
            teacher_messages = []
            for msg in messages:
                if msg["role"] == "user":
                    teacher_messages.append({
                        "role": "user",
                        "content": msg["content"] + f"\n\nReference solution:\n{solution}"
                    })
                else:
                    teacher_messages.append(msg)
            return self.tokenizer.apply_chat_template(teacher_messages, tokenize=False,
                                                       add_generation_prompt=True)
        else:  # assistant_prefix
            text = self.tokenizer.apply_chat_template(messages, tokenize=False,
                                                       add_generation_prompt=True)
            solution_prefix = f"Here is a reference approach:\n{solution}\n\nNow here is my solution:\n"
            return text + solution_prefix

    def _compute_logprobs(self, prompt_texts, response_texts):
        """
        Compute per-token logprobs of response given prompt, using verl's compute_log_prob.

        Args:
            prompt_texts: list of prompt strings
            response_texts: list of response strings

        Returns:
            list of numpy arrays with per-token logprobs for the response portion
        """
        all_logprobs = []
        pad_token_id = self.tokenizer.pad_token_id or self.tokenizer.eos_token_id

        for batch_start in tqdm(range(0, len(prompt_texts), self.batch_size), desc="Computing logprobs"):
            batch_prompts = prompt_texts[batch_start:batch_start + self.batch_size]
            batch_responses = response_texts[batch_start:batch_start + self.batch_size]

            # Tokenize prompts and responses separately
            prompt_ids_lst = []
            prompt_masks_lst = []
            response_ids_lst = []
            response_masks_lst = []
            max_prompt_len = 0
            max_response_len = 0

            for prompt_text, response_text in zip(batch_prompts, batch_responses):
                p_tokens = self.tokenizer(prompt_text, return_tensors="pt")
                r_tokens = self.tokenizer(response_text, return_tensors="pt")

                p_ids = p_tokens["input_ids"][0]
                r_ids = r_tokens["input_ids"][0]

                prompt_ids_lst.append(p_ids)
                prompt_masks_lst.append(torch.ones_like(p_ids))
                response_ids_lst.append(r_ids)
                response_masks_lst.append(torch.ones_like(r_ids))

                max_prompt_len = max(max_prompt_len, len(p_ids))
                max_response_len = max(max_response_len, len(r_ids))

            # Pad: left-pad prompt, right-pad response
            padded_input_ids_lst = []
            padded_attention_masks_lst = []
            padded_response_ids_lst = []
            padded_response_masks_lst = []

            for i in range(len(batch_prompts)):
                p_ids = prompt_ids_lst[i]
                p_mask = prompt_masks_lst[i]
                r_ids = response_ids_lst[i]
                r_mask = response_masks_lst[i]

                # Left-pad prompt
                prompt_pad_len = max_prompt_len - len(p_ids)
                padded_p_ids = torch.cat([torch.full((prompt_pad_len,), pad_token_id), p_ids])
                padded_p_mask = torch.cat([torch.zeros(prompt_pad_len, dtype=torch.long), p_mask])

                # Right-pad response
                response_pad_len = max_response_len - len(r_ids)
                padded_r_ids = torch.cat([r_ids, torch.full((response_pad_len,), pad_token_id)])
                padded_r_mask = torch.cat([r_mask, torch.zeros(response_pad_len, dtype=torch.long)])

                # Full input = prompt + response
                padded_input_ids_lst.append(torch.cat([padded_p_ids, padded_r_ids]))
                padded_attention_masks_lst.append(torch.cat([padded_p_mask, padded_r_mask]))
                padded_response_ids_lst.append(padded_r_ids)
                padded_response_masks_lst.append(padded_r_mask)

            input_ids = torch.stack(padded_input_ids_lst)
            attention_mask = torch.stack(padded_attention_masks_lst)
            position_ids = compute_position_id_with_mask(attention_mask)
            responses = torch.stack(padded_response_ids_lst)
            response_mask = torch.stack(padded_response_masks_lst)

            batch = DataProto.from_single_dict({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "responses": responses,
                "response_mask": response_mask,
            })

            logprob_output = self.actor_rollout_wg.compute_log_prob(batch)

            for i in range(len(batch_prompts)):
                logp = logprob_output.batch['old_log_probs'][i]
                effective_logp = torch.masked_select(logp, response_mask[i].bool())
                all_logprobs.append(effective_logp.cpu().numpy().astype(np.float32))

        return all_logprobs

    def fit(self):
        """Load filtered data, compute student/teacher logprobs, save results."""
        print(f"Loading filtered data from {self.data_path}")
        df = pd.read_parquet(self.data_path)
        print(f"Total responses: {len(df)}")

        # Build prompt texts
        student_prompts = []
        teacher_prompts = []
        responses = []

        for idx, row in tqdm(df.iterrows(), total=len(df), desc="Building prompts"):
            student_prompts.append(self._build_student_prompt(row))
            teacher_prompts.append(self._build_teacher_prompt(row))
            responses.append(row["response"])

        # Compute student logprobs: logp(response | prompt)
        print("\n=== Computing STUDENT logprobs (no solution) ===")
        student_logprobs = self._compute_logprobs(student_prompts, responses)

        # Compute teacher logprobs: logp(response | prompt + solution)
        print("\n=== Computing TEACHER logprobs (with solution) ===")
        teacher_logprobs = self._compute_logprobs(teacher_prompts, responses)

        # Save
        results = {
            "student_logprobs": student_logprobs,
            "teacher_logprobs": teacher_logprobs,
            "problem_indices": df["question"].tolist(),
            "scores": df["score"].tolist(),
            "is_learnable": df["is_learnable"].tolist(),
            "config": {
                "model_path": self.config.actor_rollout_ref.model.path,
                "solution_format": self.solution_format,
            }
        }

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        with open(self.output_path, "wb") as f:
            pickle.dump(results, f)
        print(f"\nSaved logprobs to {self.output_path}")

        # Sanity check
        n_valid = sum(1 for s, t in zip(student_logprobs, teacher_logprobs)
                      if len(s) > 0 and len(t) > 0)
        print(f"Valid pairs: {n_valid}/{len(df)}")
        for i in range(min(3, n_valid)):
            s, t = student_logprobs[i], teacher_logprobs[i]
            min_len = min(len(s), len(t))
            if min_len > 0:
                kl = s[:min_len] - t[:min_len]
                print(f"  Example {i}: student={s.mean():.4f}, teacher={t.mean():.4f}, KL/tok={kl.mean():.4f}")
