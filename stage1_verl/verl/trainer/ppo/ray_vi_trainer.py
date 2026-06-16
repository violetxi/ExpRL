# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import uuid
import warnings
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Any, Optional
from jinja2 import Template

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.model import compute_position_id_with_mask
from verl.utils.tracking import ValidationGenerationsLogger
from verl.trainer.ppo.ray_trainer import (
    Role, 
    ResourcePoolManager,
    apply_kl_penalty, 
    compute_response_mask,
)

WorkerType = type[Worker]


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayVITrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, and vLLM integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get("lora_rank", 0) > 0

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        if config.critic.enable is not None:
            self.use_critic = bool(config.critic.enable)
        elif self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        else:
            warnings.warn(
                "Disabled critic as algorithm.adv_estimator != gae. "
                "If it is not intended, please set critic.enable=True",
                stacklevel=2,
            )
            self.use_critic = False

        self._validate_config()
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes
        if config.actor_rollout_ref.actor.strategy == "megatron":
            model_parallel_size = (
                config.actor_rollout_ref.actor.megatron.tensor_model_parallel_size
                * config.actor_rollout_ref.actor.megatron.pipeline_model_parallel_size
            )
            assert (
                n_gpus % (model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size) == 0
            ), (
                f"n_gpus ({n_gpus}) must be divisible by model_parallel_size ({model_parallel_size}) times "
                f"context_parallel_size ({config.actor_rollout_ref.actor.megatron.context_parallel_size})"
            )
            megatron_dp = n_gpus // (
                model_parallel_size * config.actor_rollout_ref.actor.megatron.context_parallel_size
            )
            minimal_bsz = megatron_dp * config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu
        else:
            minimal_bsz = n_gpus

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % minimal_bsz == 0, (
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by minimal possible batch size "
            f"({minimal_bsz})"
        )

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            """Validate mutually exclusive micro batch size configuration options.

            Ensures that users don't set both deprecated micro_batch_size and
            the new micro_batch_size_per_gpu parameters simultaneously.

            Args:
                mbs: Deprecated micro batch size parameter value.
                mbs_per_gpu: New micro batch size per GPU parameter value.
                name (str): Configuration section name for error messages.

            Raises:
                ValueError: If both parameters are set or neither is set.
            """
            settings = {
                "reward_model": "micro_batch_size",
                "actor_rollout_ref.ref": "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }

            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"

                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(
                        f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'."
                    )

                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(
                        f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove "
                        f"'{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
                    )

        # Actor validation done in ActorConfig.__post_init__ and validate()
        actor_config = omega_conf_to_dataclass(config.actor_rollout_ref.actor)
        actor_config.validate(n_gpus, config.data.train_batch_size, config.actor_rollout_ref.model)

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(
                config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model"
            )

        if self.config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")

        # critic
        if self.use_critic:
            critic_config = omega_conf_to_dataclass(config.critic)
            critic_config.validate(n_gpus, config.data.train_batch_size)

        if config.data.get("val_batch_size", None) is not None:
            print(
                "WARNING: val_batch_size is deprecated."
                + " Validation datasets are sent to inference engines as a whole batch,"
                + " which will schedule the memory themselves."
            )

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, (
                "validation gen temperature should be greater than 0 when enabling do_sample"
            )

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files, self.config.data, self.tokenizer, self.processor
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files, self.config.data, self.tokenizer, self.processor
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []
        sample_turns = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            if "interaction_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("interaction_kwargs")
            if "agent_name" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("agent_name")
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True            

            # evaluate using reward_function
            if self.val_reward_fn is None:
                raise ValueError("val_reward_fn must be provided for validation.")
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            print(f"len reward_extra_infos_dict['reward']: {len(reward_extra_infos_dict['reward'])}")
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)
                    print(f"len reward_extra_infos_dict['{key}']: {len(reward_extra_infos_dict[key])}")

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))
        
        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_inputs, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
                profile_option=self.config.trainer.npu_profile.options,
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cfg = omega_conf_to_dataclass(self.config.critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role="ref",
                profile_option=self.config.trainer.npu_profile.options,
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.trainer, "profile_steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.trainer, "profile_steps")
            assert OmegaConf.select(self.config.trainer, "worker_nsight_options") is not None, (
                "worker_nsight_options must be set when profile_steps is set"
            )
            wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                OmegaConf.select(self.config.trainer, "worker_nsight_options")
            )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            from verl.experimental.agent_loop import AgentLoopManager

            self.async_rollout_mode = True
            self.async_rollout_manager = AgentLoopManager(
                config=self.config,
                worker_group=self.actor_rollout_wg,
            )

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile()
            if self.use_critic:
                self.critic_wg.start_profile()
            if self.use_rm:
                self.rm_wg.start_profile()

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm:
                self.rm_wg.stop_profile()

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def _check_format(self, response_str, format_type):
        """
        Return 1 if the format is correct, 0 otherwise.
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
            if response_str.strip().startswith("<R>"):
                return 1
            else:
                return 0
        elif format_type == "box":
            if "\\boxed{" in response_str[-100:]:
                return 1
            else:
                return 0
        elif format_type == "answer":
            if "<answer" in response_str[-100:]:
                return 1
            else:
                return 0
        else:
            raise ValueError(f"Invalid format type: {format_type}")

    def _compute_reward_vi_vanilla(self, batch: DataProto):
        outcome_reward_weight = self.config.reward_model.reward_kwargs.get("outcome_reward_weight", 1)
        format_type = self.config.reward_model.reward_kwargs.get("format_type", "answer")
        """Compute the reward for the batch, and output full_batch with proper
        pg_loss_mask and sft_loss_mask """
        prompt_ids_lst = []
        prompt_masks_lst = []        
        response_ids_lst = []
        response_masks_lst = []
        reference_ids_lst = []
        reference_masks_lst = []
        ### for tracking purpose: whether format is correct and if incorrect is due to length truncation
        has_stop_lst = []
        no_stop_max_length_lst = []
        # track for padding
        max_prompt_len = 0
        max_response_len = 0
        max_reference_len = 0
        eos_token_id = self.tokenizer.eos_token_id
        response_length = batch.batch['responses'].shape[-1]
        for i, batch_item in enumerate(batch):
            # get effective prompt_ids, prompt_mask, response_ids, response_masks
            prompt_ids = torch.masked_select(
                batch_item.batch['input_ids'][:-response_length], 
                batch_item.batch['attention_mask'][:-response_length].bool())
            prompt_masks = torch.ones_like(prompt_ids)
            response_ids = torch.masked_select(
                batch_item.batch['responses'],  batch_item.batch['response_mask'].bool())
            response_no_eos_mask = response_ids != eos_token_id
            response_ids = response_ids[response_no_eos_mask]
            response_masks = torch.ones_like(response_ids)
            
            response_str = self.tokenizer.decode(response_ids)
            response_masks = torch.ones_like(response_ids)
            # get reference ids and reference masks
            reference = batch_item.non_tensor_batch['extra_info'][self.config.data.reference_key]
            if self.config.data.reference_key == "solution":                
                # if thinking trace did not end with </think>, then prepend it
                # we assuming when using full solution as reference, the format_type must be think
                assert format_type == "think", "format_type must be think to use solution as reference"
                if "</think>" not in response_str:
                    reference_str = f"Considering the limited time by the user, I have to give the solution based on the thinking directly now \n</think>\n\n{reference}"
                    has_stop = True
                else:
                    reference_str = reference
                    has_stop = False
            elif self.config.data.reference_key == "answer":
                assert format_type == "answer", "format_type must be answer to use answer as reference"
                if "<answer" in response_str:
                    reference_str =  Template("> \\boxed{ {{-reference-}} } </answer>").render(reference=reference)
                    has_stop = True
                else:
                    reference_str =  reference
                    has_stop = False
            else:
                raise ValueError(f"Invalid reference key: {self.config.data.reference_key}")
            reference_token_dict = self.tokenizer(reference_str, return_tensors="pt")
            reference_ids = reference_token_dict["input_ids"][0]            
            reference_masks = reference_token_dict["attention_mask"][0]
            # track max length
            max_prompt_len = max(max_prompt_len, prompt_ids.shape[0])
            max_response_len = max(max_response_len, response_ids.shape[0])
            max_reference_len = max(max_reference_len, reference_ids.shape[0])
            # adding to list for padding
            prompt_ids_lst.append(prompt_ids)
            prompt_masks_lst.append(prompt_masks)
            response_ids_lst.append(response_ids)
            response_masks_lst.append(response_masks)            
            reference_ids_lst.append(reference_ids)
            reference_masks_lst.append(reference_masks)
            # for tracking purpose
            has_stop_lst.append(has_stop)
            no_stop_max_length = response_masks.sum().item() == self.config.data.max_response_length if not has_stop else False
            no_stop_max_length_lst.append(no_stop_max_length)
            
        # re-do padding across all tensors
        padded_prompt_ids_lst = []
        padded_prompt_masks_lst = []
        padded_full_response_ids_lst = []
        padded_full_response_masks_lst = []
        pad_token_id = self.tokenizer.pad_token_id
        for i in range(len(prompt_ids_lst)):
            prompt_id = prompt_ids_lst[i]
            prompt_mask = prompt_masks_lst[i]
            response_id = response_ids_lst[i]
            response_mask = response_masks_lst[i]
            reference_id = reference_ids_lst[i]
            reference_mask = reference_masks_lst[i]
            # left padding for prompt
            padded_prompt_ids_lst.append(torch.cat([
                torch.full((max_prompt_len - prompt_id.shape[0],), 
                pad_token_id), prompt_id]))
            padded_prompt_masks_lst.append(torch.cat([
                torch.full((max_prompt_len - prompt_mask.shape[0],), 
                0), prompt_mask]))
            # right padding for full_response (response + reference) to max_response_len + max_reference_len
            full_response_padding_len = max_response_len - response_id.shape[0] + max_reference_len - reference_id.shape[0]
            padded_full_response_id = torch.cat([
                response_id,
                reference_id,
                torch.full((full_response_padding_len,), 
                pad_token_id)])
            padded_full_response_mask = torch.cat([
                response_mask,
                reference_mask,
                torch.full((full_response_padding_len,), 0)])
            padded_full_response_ids_lst.append(padded_full_response_id)
            padded_full_response_masks_lst.append(padded_full_response_mask)
        
        padded_prompt_ids_tensor = torch.stack(padded_prompt_ids_lst)
        padded_prompt_masks_tensor = torch.stack(padded_prompt_masks_lst)        
        padded_full_response_ids_tensor = torch.stack(padded_full_response_ids_lst)
        padded_full_response_masks_tensor = torch.stack(padded_full_response_masks_lst)

        ##### for compute reward: 
        # - input_ids -> prompt_id + response_id + reference_id
        # - response_ids -> reference_id
        # - response_masks -> reference_mask
        logp_input_ids = torch.cat([padded_prompt_ids_tensor, padded_full_response_ids_tensor], dim=1)
        logp_attention_mask = torch.cat([padded_prompt_masks_tensor, padded_full_response_masks_tensor], dim=1)
        logp_position_ids = compute_position_id_with_mask(logp_attention_mask)
        logp_responses = padded_full_response_ids_tensor
        logp_response_mask = padded_full_response_masks_tensor
        logp_batch = DataProto.from_single_dict({
            "input_ids": logp_input_ids,
            "attention_mask": logp_attention_mask,
            "position_ids": logp_position_ids,
            "responses": logp_responses,
            "response_mask": logp_response_mask
        })
        ##### for compute loss: 
        # - input_ids -> prompt_id + response_id + reference_id
        # - response_ids -> response_id + reference_id
        # - pg_loss_mask -> response_mask
        # - sft_loss_mask -> reference_mask
        pg_loss_mask_lst = []
        sft_loss_mask_lst = []
        reward_no_stop_token_lst = []
        reward_with_stop_token_lst = []
        logprobs =self.actor_rollout_wg.compute_log_prob(logp_batch)
        reward_tensor = torch.zeros_like(logp_response_mask, dtype=torch.float32)
        agg_func = self.config.reward_model.reward_kwargs.get("agg_func", "sum")
        for i, logprob in enumerate(logprobs.batch['old_log_probs']):
            full_response_mask = logp_response_mask[i]
            full_response_len = full_response_mask.shape[0]
            unpadded_response_mask = response_masks_lst[i]
            unpadded_response_len = unpadded_response_mask.shape[0]
            unpadded_reference_mask = reference_masks_lst[i]
            unpadded_reference_len = unpadded_reference_mask.shape[0]                        
            # right pad unpadded_response_mask to full_response_len, 
            # pg_loss_mask is always on regardless of whether has_stop
            pg_loss_mask = torch.cat([
                unpadded_response_mask, 
                torch.full((full_response_len - unpadded_response_len,), 0)])
            # left pad reference_mask to unpadded_response_len, and right pad to 
            # full_response_len - unpadded_response_len - unpadded_reference_len 
            # sft_loss_mask is only on if has_stop
            if has_stop_lst[i]:
                sft_loss_mask = torch.cat([
                    torch.full((unpadded_response_len,), 0), 
                    unpadded_reference_mask, 
                    torch.full((full_response_len - unpadded_response_len - unpadded_reference_len,), 0)])
                logprob = torch.masked_select(logprob, sft_loss_mask.bool())
            else:
                sft_loss_mask = torch.full((full_response_len,), 0)
                logprob = torch.tensor(float('-inf'))

            if agg_func == "sum":
                logprob_agg = logprob.sum()
            elif agg_func == "mean":
                logprob_agg = logprob.mean()
            else:
                raise ValueError(f"Invalid agg_func: {agg_func}")
            reward = torch.exp(logprob_agg) * outcome_reward_weight
            reward_tensor[i, unpadded_response_mask.sum().item() - 1] = reward
            pg_loss_mask_lst.append(pg_loss_mask)
            sft_loss_mask_lst.append(sft_loss_mask)
            if has_stop_lst[i]:
                reward_with_stop_token_lst.append(reward)
            else:
                reward_no_stop_token_lst.append(reward)
        
        pg_loss_mask_tensor = torch.stack(pg_loss_mask_lst)
        sft_loss_mask_tensor = torch.stack(sft_loss_mask_lst)
        full_batch = DataProto.from_single_dict({
            "input_ids": logp_input_ids,
            "attention_mask": logp_attention_mask,
            "position_ids": logp_position_ids,
            "responses": logp_responses,
            "response_mask": logp_response_mask,
            "pg_loss_mask": pg_loss_mask_tensor,
            "sft_loss_mask": sft_loss_mask_tensor,
        })
        full_batch.non_tensor_batch = batch.non_tensor_batch.copy()
        full_batch.meta_info = batch.meta_info.copy()

        metrics = {
            "vi_tracking/reward_no_stop_token": np.mean(reward_no_stop_token_lst),
            "vi_tracking/reward_with_stop_token": np.mean(reward_with_stop_token_lst),
            "vi_tracking/has_stop": np.mean(has_stop_lst),
            "vi_tracking/no_stop_max_length": np.mean(no_stop_max_length_lst)
        }
        return reward_tensor, full_batch, metrics

    def _compute_reward_vi_scaled_p_answer(self, batch: DataProto):
        """Compute the reward for the batch. We are assuming instruct model here bc we will prepend 
        a prefix string to the response, which is used to ONLY compute logp(y* |x, z) reward."""
        outcome_reward_weight = self.config.reward_model.reward_kwargs.get("outcome_reward_weight", 1)
        logp_input_ids_lst = []
        logp_attention_mask_lst = []        
        logp_response_ids_lst = []
        logp_response_mask_lst = []
        max_input_length = 0
        max_response_length = 0
        # create a new batch with prefix string + solution string
        prefix_str = "\n\nHence, the final answer is \\boxed{"
        prefix_token_dict = self.tokenizer(prefix_str, return_tensors="pt")
        prefix_token_id = prefix_token_dict["input_ids"][0]
        prefix_attention_mask = prefix_token_dict["attention_mask"][0]
        eos_token_id = self.tokenizer.eos_token_id
        pad_token_id = self.tokenizer.pad_token_id
        for i, batch_item in enumerate(batch):
            # get effective input_ids with eos masked out for logp(y* |x, z) computation
            input_id = torch.masked_select(batch_item.batch['input_ids'], batch_item.batch['attention_mask'].bool())
            eos_mask = input_id == eos_token_id
            input_id = input_id[~eos_mask]
            attention_mask = torch.ones_like(input_id)
            # get reference solution
            reference = batch_item.non_tensor_batch['extra_info'][self.config.data.reference_key]
            if self.config.data.reference_key == "solution":
                # we assuming when using full solution as reference, the format_type must be think
                reference_str = reference
            elif self.config.data.reference_key == "answer":
                reference_str = reference + "}"
            else:
                raise ValueError(f"Invalid reference key: {self.config.data.reference_key}")
            reference_token_dict = self.tokenizer(reference_str, return_tensors="pt")
            reference_token_id = reference_token_dict["input_ids"][0]
            reference_attention_mask = reference_token_dict["attention_mask"][0]
            max_response_length = max(max_response_length, reference_token_id.shape[0])

            logp_input_ids = torch.cat([input_id, prefix_token_id], dim=0)
            logp_attention_mask = torch.cat([attention_mask, prefix_attention_mask], dim=0)
            max_input_length = max(max_input_length, logp_input_ids.shape[0])
            logp_input_ids_lst.append(logp_input_ids)
            logp_attention_mask_lst.append(logp_attention_mask)
            logp_response_ids_lst.append(reference_token_id)
            logp_response_mask_lst.append(reference_attention_mask)

        padded_logp_input_ids_lst = []
        padded_logp_attention_mask_lst = []        
        padded_logp_response_ids_lst = []
        padded_logp_response_mask_lst = []
        for i in range(len(logp_input_ids_lst)):
            logp_input_ids = logp_input_ids_lst[i]
            logp_attention_mask = logp_attention_mask_lst[i]
            logp_response_ids = logp_response_ids_lst[i]
            logp_response_mask = logp_response_mask_lst[i]
            input_padding_len = max_input_length - logp_input_ids.shape[0]
            response_padding_len = max_response_length - logp_response_ids.shape[0]
            # right padding for response
            padded_logp_response_ids = torch.cat([logp_response_ids, torch.full((response_padding_len,), pad_token_id)])
            padded_logp_response_mask = torch.cat([logp_response_mask, torch.full((response_padding_len,), 0)])
            # for input_ids, we need to left pad for logp_input_ids then concatenate the padded_logp_response_ids
            padded_logp_input_ids = torch.cat([torch.full((input_padding_len,), pad_token_id), logp_input_ids, padded_logp_response_ids])
            padded_logp_attention_mask = torch.cat([torch.full((input_padding_len,), 0), logp_attention_mask, padded_logp_response_mask])            

            padded_logp_input_ids_lst.append(padded_logp_input_ids)
            padded_logp_attention_mask_lst.append(padded_logp_attention_mask)
            padded_logp_response_ids_lst.append(padded_logp_response_ids)
            padded_logp_response_mask_lst.append(padded_logp_response_mask)
            # self.tokenizer.decode(torch.masked_select(padded_logp_input_ids_lst[0][-padded_logp_response_mask_lst[0].shape[0]:], padded_logp_response_mask_lst[0].bool()))
    
        padded_logp_input_ids = torch.stack(padded_logp_input_ids_lst)
        padded_logp_attention_mask = torch.stack(padded_logp_attention_mask_lst)
        padded_logp_position_ids = compute_position_id_with_mask(padded_logp_attention_mask)
        padded_logp_response_ids = torch.stack(padded_logp_response_ids_lst)
        padded_logp_response_mask = torch.stack(padded_logp_response_mask_lst)
        logp_batch = DataProto.from_single_dict({
            "input_ids": padded_logp_input_ids,
            "attention_mask": padded_logp_attention_mask,
            "position_ids": padded_logp_position_ids,
            "responses": padded_logp_response_ids,
            "response_mask": padded_logp_response_mask,
        })        
        logp_info = self.actor_rollout_wg.compute_log_prob(logp_batch)
        reward_tensor = torch.zeros_like(batch.batch['response_mask'], dtype=torch.float32)
        agg_func = self.config.reward_model.reward_kwargs.get("agg_func", "sum")
        for i, logp in enumerate(logp_info.batch['old_log_probs']):
            reward_mask = logp_batch.batch['response_mask'][i]
            effective_logp = torch.masked_select(logp, reward_mask.bool())
            if agg_func == "sum":
                logp_agg = effective_logp.sum()
            elif agg_func == "mean":
                logp_agg = effective_logp.mean()
            else:
                raise ValueError(f"Invalid agg_func: {agg_func}")
            reward_tensor[i, batch.batch['response_mask'][i].sum().item() - 1] = torch.exp(logp_agg) * outcome_reward_weight
        
        self.tokenizer.decode(torch.masked_select(padded_logp_input_ids[4], padded_logp_attention_mask[4].bool()))
        # @TODO we don't worry about format here just yet, also the policy will be updated the same way as vanilla loss mode        
        return reward_tensor, batch, {}

    def _compute_reward_vi_scaled_p_solution(self, batch: DataProto):
        """Compute the reward for the batch. We are assuming instruct model here bc we will prepend 
        a prefix string to the response, which is used to ONLY compute logp(y* |x, z) reward."""
        outcome_reward_weight = self.config.reward_model.reward_kwargs.get("outcome_reward_weight", 1)
        reward_scale = self.config.reward_model.reward_kwargs.get("reward_scale", 1)
        prompt_ids_lst = []
        prompt_masks_lst = []
        think_ids_lst = []
        think_masks_lst = []
        reference_ids_lst = []
        reference_masks_lst = []
        max_prompt_length = 0
        max_think_length = 0
        max_reference_length = 0
        ## split z from response based on </think> token
        end_of_think_token_id = self.tokenizer.encode("</think>", add_special_tokens=False)[0]
        response_length = batch.batch['responses'].shape[-1]
        for i, batch_item in enumerate(batch):            
            # reference - y*
            reference = batch_item.non_tensor_batch['extra_info'][self.config.data.reference_key]
            response_str = self.tokenizer.decode(
                torch.masked_select(batch_item.batch['responses'], batch_item.batch['response_mask'].bool()))
            has_end_of_think = response_str.find("</think>")
            if has_end_of_think == -1:    # did not finish thinking
                reference_str = Template("Considering the limited time by the user, I have to give the solution based on the thinking directly now \n</think>\n\n{{reference}}").render(reference=reference)
            else:
                reference_str = f"\n\n{reference}"
            reference_token_dict = self.tokenizer(reference_str, return_tensors="pt")
            reference_ids = reference_token_dict["input_ids"][0]
            reference_masks = reference_token_dict["attention_mask"][0]
            max_reference_length = max([max_reference_length, reference_masks.sum().item()])
            # prompt - x
            prompt_ids = torch.masked_select(
                batch_item.batch['input_ids'][:-response_length], 
                batch_item.batch['attention_mask'][:-response_length].bool())
            prompt_masks = torch.ones_like(prompt_ids)
            max_prompt_length = max([max_prompt_length, prompt_masks.sum().item()])
            # thinking - z
            response_ids = torch.masked_select(
                batch_item.batch['responses'],  batch_item.batch['response_mask'].bool())

            if has_end_of_think != -1:
                end_of_think_idx = (response_ids == end_of_think_token_id).nonzero()[0][0]
                think_ids = response_ids[: end_of_think_idx + 1]
                think_masks = torch.ones_like(think_ids)
            else:
                think_ids = response_ids
                think_masks = torch.ones_like(think_ids)
            max_think_length = max(max_think_length, think_masks.sum().item())

            prompt_ids_lst.append(prompt_ids)
            prompt_masks_lst.append(prompt_masks)
            think_ids_lst.append(think_ids)
            think_masks_lst.append(think_masks)
            reference_ids_lst.append(reference_ids)
            reference_masks_lst.append(reference_masks)

        ### pad prompt, think, reference
        padded_input_ids_lst = []
        padded_attention_masks_lst = []
        padded_response_ids_lst = []
        padded_response_masks_lst = []
        pad_token_id = self.tokenizer.pad_token_id
        for i in range(len(prompt_ids_lst)):
            prompt_ids = prompt_ids_lst[i]
            prompt_masks = prompt_masks_lst[i]
            think_ids = think_ids_lst[i]
            think_masks = think_masks_lst[i]
            reference_ids = reference_ids_lst[i]
            reference_masks = reference_masks_lst[i]
            # left pad prompt + think to max prompt + think length
            prompt_padding_len = max_prompt_length - prompt_ids.shape[0] + max_think_length - think_ids.shape[0]
            padded_prompt_ids = torch.cat([torch.full((prompt_padding_len,), pad_token_id), prompt_ids])
            padded_prompt_masks = torch.cat([torch.full((prompt_padding_len,), 0), prompt_masks])
            # right pad reference so the total of think + reference = max reference length + max response length
            reference_padding_len = max_reference_length - reference_ids.shape[0]
            padded_reference_ids = torch.cat([reference_ids, torch.full((reference_padding_len,), pad_token_id)])
            padded_reference_masks = torch.cat([reference_masks, torch.full((reference_padding_len,), 0)])
            # now create input_ids and responses
            padded_input_ids = torch.cat([padded_prompt_ids, think_ids, padded_reference_ids])
            padded_attention_masks = torch.cat([padded_prompt_masks, think_masks, padded_reference_masks])
            padded_input_ids_lst.append(padded_input_ids)
            padded_attention_masks_lst.append(padded_attention_masks)
            padded_response_ids_lst.append(padded_reference_ids)
            padded_response_masks_lst.append(padded_reference_masks)
    
        padded_input_ids = torch.stack(padded_input_ids_lst)
        padded_attention_masks = torch.stack(padded_attention_masks_lst)
        padded_position_ids = compute_position_id_with_mask(padded_attention_masks)
        padded_response_ids = torch.stack(padded_response_ids_lst)
        padded_response_masks = torch.stack(padded_response_masks_lst)
        logp_batch = DataProto.from_single_dict({
            "input_ids": padded_input_ids,
            "attention_mask": padded_attention_masks,
            "position_ids": padded_position_ids,
            "responses": padded_response_ids,
            "response_mask": padded_response_masks,
        })
        logp_info = self.actor_rollout_wg.compute_log_prob(logp_batch)
        reward_tensor = torch.zeros_like(batch.batch['response_mask'], dtype=torch.float32)
        agg_func = self.config.reward_model.reward_kwargs.get("agg_func", "sum")
        for i, logp in enumerate(logp_info.batch['old_log_probs']):
            reward_mask = logp_batch.batch['response_mask'][i]
            effective_logp = torch.masked_select(logp, reward_mask.bool())
            if agg_func == "sum":
                logp_agg = effective_logp.sum()
            elif agg_func == "mean":
                logp_agg = effective_logp.mean()
            else:
                raise ValueError(f"Invalid agg_func: {agg_func}")
            reward_tensor[i, batch.batch['response_mask'][i].sum().item() - 1] = torch.exp(logp_agg * reward_scale) * outcome_reward_weight
        
        return reward_tensor, batch, {}

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.trainer.profile_steps
            if self.config.trainer.profile_steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.trainer.profile_continuous_steps
                        else curr_step_profile
                    )

                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # pop those keys for generation
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                if "interaction_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("interaction_kwargs")
                if "index" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("index")
                if "agent_name" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("agent_name")

                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                is_last_step = self.global_steps >= self.total_training_steps
                loss_mode = self.config.actor_rollout_ref.actor.policy_loss.loss_mode

                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                    )
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    # TODO: Decouple the DP balancing and mini-batching.
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with marked_timer("reward", timing_raw, color="yellow"):
                        # vi_vanilla compute p(y* |x, z) as the reward
                        if loss_mode == "vi_vanilla":
                            reward_tensor, batch, vi_reward_metrics = self._compute_reward_vi_vanilla(batch)
                        elif loss_mode == "vanilla":
                            if self.config.data.reference_key == "answer":
                                reward_tensor, batch, vi_reward_metrics = self._compute_reward_vi_scaled_p_answer(batch)
                            elif self.config.data.reference_key == "solution":
                                reward_tensor, batch, vi_reward_metrics = self._compute_reward_vi_scaled_p_solution(batch)
                        else:
                            raise ValueError(f"Invalid loss_mode: {loss_mode}")
                        metrics.update(vi_reward_metrics)
                        reward_extra_infos_dict = {}
                        # if self.config.reward_model.launch_reward_fn_async:
                        #     future_reward = compute_reward_async.remote(data=batch, reward_fn=self.reward_fn)
                        # else:
                        #     # ground truth reward for logging and analysis
                        #     gt_reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, color="blue"):                        
                        # batch without reference
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        # get the max of logprob for each response
                        max_logprob = (old_log_prob.batch["old_log_probs"] * response_masks).max().item()
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        entropy_max = entropys.max().detach().item()
                        entropy_min = entropys.min().detach().item()  
                        old_log_prob_metrics = {
                            "actor/entropy": entropy_agg.detach().item(), 
                            "actor/entropy_max": entropy_max, 
                            "actor/entropy_min": entropy_min,
                            "actor/max_logprob": max_logprob
                        }                        

                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            rollout_old_log_probs = batch.batch["rollout_log_probs"]
                            actor_old_log_probs = batch.batch["old_log_probs"]
                            attention_mask = batch.batch["attention_mask"]
                            responses = batch.batch["responses"]
                            response_length = responses.size(1)
                            response_mask = attention_mask[:, -response_length:]

                            rollout_probs = torch.exp(rollout_old_log_probs)
                            actor_probs = torch.exp(actor_old_log_probs)
                            rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                            rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                            rollout_probs_diff_max = torch.max(rollout_probs_diff)
                            rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                            rollout_probs_diff_std = torch.std(rollout_probs_diff)
                            metrics.update(
                                {
                                    "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                                    "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                                    "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                                }
                            )                        

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer("ref", timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            gt_reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process
                        # GRPO adv normalization factor
                        norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)                        
                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):                            
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                            if "request_id" in batch.non_tensor_batch:
                                reward_extra_infos_dict.setdefault(
                                    "request_id",
                                    batch.non_tensor_batch["request_id"].tolist(),
                                )
                            self._dump_generations(
                                inputs=inputs,
                                outputs=outputs,
                                scores=scores,
                                reward_extra_infos_dict=reward_extra_infos_dict,
                                dump_path=rollout_data_dir,
                            )

                    # validate
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with marked_timer("testing", timing_raw, color="green"):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                    esi_close_to_expiration = should_save_ckpt_esi(
                        max_steps_duration=self.max_steps_duration,
                        redundant_time=self.config.trainer.esi_redundant_time,
                    )
                    # Check if the conditions for saving a checkpoint are met.
                    # The conditions include a mandatory condition (1) and
                    # one of the following optional conditions (2/3/4):
                    # 1. The save frequency is set to a positive value.
                    # 2. It's the last training step.
                    # 3. The current step number is a multiple of the save frequency.
                    # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                    if self.config.trainer.save_freq > 0 and (
                        is_last_step
                        or self.global_steps % self.config.trainer.save_freq == 0
                        or esi_close_to_expiration
                    ):
                        if esi_close_to_expiration:
                            print("Force saving checkpoint: ESI instance expiration approaching.")
                        with marked_timer("save_checkpoint", timing_raw, color="green"):
                            self._save_checkpoint()

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.trainer.profile_steps
                        if self.config.trainer.profile_steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.trainer.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
