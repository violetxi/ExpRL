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
from pprint import pprint
from typing import Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
import verl.utils.torch_functional as verl_F
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
from verl.utils.model import compute_position_id_with_mask
from verl.trainer.ppo.reward import compute_reward
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.trainer.ppo.ray_trainer import Role, ResourcePoolManager, apply_kl_penalty, compute_response_mask

WorkerType = type[Worker]



def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
    last_step_reward: Optional[torch.Tensor] = None,
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
    elif adv_estimator == AdvantageEstimator.GPPO:
        gppo_calculation_mask = data.batch["response_mask"]
        advantages, returns = core_algos.compute_gppo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=gppo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GPR_E:
        gpr_e_calculation_mask = data.batch["response_mask"]
        advantages, returns = core_algos.compute_gpr_e_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=gpr_e_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
            last_step_reward=last_step_reward,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.PR_BN:
        pr_bn_calculation_mask = data.batch["response_mask"]
        advantages, returns = core_algos.compute_rn_bn_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=pr_bn_calculation_mask,
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


class RayGenRMTrainer:
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
        self.use_genrm = Role.GenRM in role_worker_mapping
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

        # Pre-compute multi-turn BOT/EOT tokens if multi-turn GenRM is enabled
        self.max_turns = self.config.custom_reward_function.get("max_turns", 1)
        self.feedback_user_template = None
        if self.max_turns > 1:
            self._extract_turn_tokens()
            from jinja2 import Template
            self.feedback_user_template = Template(
                "Task: Write a revised second attempt that fixes the issues below "
                "based on the feedback for your first solution.\n"
                "Do NOT mention the feedback explicitly.\n"
                "Do NOT debate whether t exists.\n"
                "Produce a complete solution and a final answer.\n\n"
                "Feedback: {{ feedback }}"
            )

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

        # create GenRM worker (separate LLM-as-judge for reward scoring)
        resource_pool = self.resource_pool_manager.get_resource_pool(Role.GenRM)
        # Use actor model path if genrm model path is not specified
        genrm_config = deepcopy(self.config.genrm)
        if genrm_config.model.get("path", None) is None:
            genrm_config.model.path = self.config.actor_rollout_ref.model.path
        genrm_cls = RayClassWithInitArgs(
            cls=self.role_worker_mapping[Role.GenRM],
            config=genrm_config,
            role="genrm",
            profile_option=self.config.trainer.npu_profile.options,
        )
        self.resource_pool_to_cls[resource_pool]["genrm"] = genrm_cls

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
                
        if self.use_genrm:
            self.genrm_wg = all_wg["genrm"]
            self.genrm_wg.init_model()

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

    def _load_prompt_template(self, prompt_template_path: str) -> str:
        import importlib.util
        spec = importlib.util.spec_from_file_location("prompt_module", prompt_template_path)
        prompt_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(prompt_module)

        system_prompt = prompt_module.SYSTEM_PROMPT
        user_prompt = prompt_module.USER_PROMPT
        return system_prompt, user_prompt

    def _score_normalization(self, score: int, norm_strategy: str) -> int:        
        if norm_strategy == "extreme_only_clipped":
            if score <= 3:
                return 0
            else:
                return 1
        elif norm_strategy == "extreme_only":
            if score == 1:
                return 0
            elif score == 5:
                return 1
            else:
                return None
        elif norm_strategy == "semi_smooth":
            if score == 1:
                return 0
            elif score == 5:
                return 1
            elif score == 4:
                return 0.8
            else:
                return None
        elif norm_strategy == "smooth":
            return (score - 1) / 4

    # ================================================================
    # Multi-turn GenRM utilities
    # ================================================================

    @staticmethod
    def _find_subseq(full_ids: list[int], sub_ids: list[int]) -> int:
        """Find the starting index of sub_ids within full_ids."""
        n, m = len(full_ids), len(sub_ids)
        for i in range(n - m + 1):
            if full_ids[i:i + m] == sub_ids:
                return i
        raise ValueError(f"Subsequence not found: {sub_ids} in sequence of length {n}")

    def _extract_turn_tokens(self):
        """Probe the chat template to pre-compute BOT/EOT token ID sequences.

        Extracts two token-ID tensors by probing apply_chat_template(tokenize=True):
          - AFTER_RESPONSE: assistant EOT + user BOT  (placed after actor response, before feedback)
          - AFTER_FEEDBACK: user EOT + assistant BOT   (placed after feedback, before next generation)
        """
        PROBE_A = "ZZZZPROBEASSISTANT12345ZZZZ"
        PROBE_B = "ZZZZPROBEUSER67890ZZZZ"

        probe = [
            {"role": "user", "content": "x"},
            {"role": "assistant", "content": PROBE_A},
            {"role": "user", "content": PROBE_B},
        ]
        enable_thinking = self.config.data.get("enable_thinking", None)
        chat_template_kwargs = dict(add_generation_prompt=True, tokenize=True)
        if enable_thinking is not None:
            chat_template_kwargs["enable_thinking"] = enable_thinking
        full_ids = self.tokenizer.apply_chat_template(
            probe, **chat_template_kwargs)

        a_ids = self.tokenizer.encode(PROBE_A, add_special_tokens=False)
        b_ids = self.tokenizer.encode(PROBE_B, add_special_tokens=False)

        a_end = self._find_subseq(full_ids, a_ids) + len(a_ids)
        b_start = self._find_subseq(full_ids, b_ids)
        b_end = b_start + len(b_ids)

        # AFTER_RESPONSE = tokens between assistant content end and user content start
        #   e.g. [<|im_end|>, \n, <|im_start|>, user, \n] for ChatML
        self.AFTER_RESPONSE = torch.tensor(full_ids[a_end:b_start], dtype=torch.long)

        # AFTER_FEEDBACK = tokens after user content end through generation prompt
        #   e.g. [<|im_end|>, \n, <|im_start|>, assistant, \n] for ChatML
        self.AFTER_FEEDBACK = torch.tensor(full_ids[b_end:], dtype=torch.long)

        # Compute number of leading special tokens in AFTER_RESPONSE that correspond
        # to the assistant EOT (the same tokens _strip_trailing_eos removes from responses).
        # These should be included in response_mask during training.
        # e.g. for ChatML, AFTER_RESPONSE starts with <|im_end|> which is eos_token_id → 1 token.
        special = {self.tokenizer.eos_token_id, self.tokenizer.pad_token_id}
        if hasattr(self.tokenizer, 'eot_token_id') and self.tokenizer.eot_token_id is not None:
            special.add(self.tokenizer.eot_token_id)
        special.discard(None)
        eot_len = 0
        for tok_id in self.AFTER_RESPONSE.tolist():
            if tok_id in special:
                eot_len += 1
            else:
                break
        self.RESPONSE_EOT_LEN = eot_len

        print(f"[multi-turn] AFTER_RESPONSE token ids ({len(self.AFTER_RESPONSE)}): {self.AFTER_RESPONSE.tolist()}")
        print(f"[multi-turn] AFTER_FEEDBACK token ids ({len(self.AFTER_FEEDBACK)}): {self.AFTER_FEEDBACK.tolist()}")
        print(f"[multi-turn] RESPONSE_EOT_LEN (included in response_mask): {self.RESPONSE_EOT_LEN}")

    def _strip_trailing_eos(self, response_ids: torch.Tensor) -> torch.Tensor:
        """Strip trailing EOS/pad/EOT tokens from a response tensor.

        The bridge (AFTER_RESPONSE) already includes the assistant EOT,
        so we strip it from the response to avoid duplication.
        """
        special = {self.tokenizer.eos_token_id, self.tokenizer.pad_token_id}
        if hasattr(self.tokenizer, 'eot_token_id') and self.tokenizer.eot_token_id is not None:
            special.add(self.tokenizer.eot_token_id)
        special.discard(None)

        end = response_ids.numel()
        while end > 0 and response_ids[end - 1].item() in special:
            end -= 1
        return response_ids[:end]

    def _build_next_turn_prompt(self, prev_prompt_ids: torch.Tensor,
                                 response_ids: torch.Tensor,
                                 feedback_text: str) -> torch.Tensor:
        """Build the next turn's prompt via pure token concatenation.

        Layout: [prev_prompt | response (eos-stripped) | AFTER_RESPONSE | user_content_ids | AFTER_FEEDBACK]

        Only the user content (feedback, optionally wrapped by template) gets tokenized.
        prev_prompt_ids and response_ids stay as raw token tensors.
        """
        resp = self._strip_trailing_eos(response_ids)

        # Optionally wrap feedback with a user message template
        # e.g. "Task: Write a revised second attempt...\n\nFeedback: {{ feedback }}"
        if self.feedback_user_template is not None:
            user_content = self.feedback_user_template.render(feedback=feedback_text)
        else:
            user_content = feedback_text

        user_content_ids = torch.tensor(
            self.tokenizer.encode(user_content, add_special_tokens=False),
            dtype=torch.long)
        return torch.cat([
            prev_prompt_ids,
            resp,
            self.AFTER_RESPONSE,
            user_content_ids,
            self.AFTER_FEEDBACK,
        ])

    def _pad_prompts_to_gen_batch(self, prompt_ids_list: list[torch.Tensor],
                                   meta_info: Optional[dict] = None) -> DataProto:
        """Left-pad variable-length prompt tensors into a DataProto for generate_sequences.

        Args:
            prompt_ids_list: List of 1-D token ID tensors (variable length).
            meta_info: Optional meta_info dict to attach.

        Returns:
            DataProto with input_ids, attention_mask, position_ids (all left-padded).
        """
        pad_id = self.tokenizer.pad_token_id
        max_len = max(p.numel() for p in prompt_ids_list)

        input_ids_lst = []
        attention_mask_lst = []
        for p in prompt_ids_list:
            pad_len = max_len - p.numel()
            padded = torch.cat([torch.full((pad_len,), pad_id, dtype=torch.long), p])
            mask = torch.cat([torch.zeros(pad_len, dtype=torch.long), torch.ones(p.numel(), dtype=torch.long)])
            input_ids_lst.append(padded.unsqueeze(0))
            attention_mask_lst.append(mask.unsqueeze(0))

        input_ids = torch.cat(input_ids_lst, dim=0)
        attention_mask = torch.cat(attention_mask_lst, dim=0)
        position_ids = compute_position_id_with_mask(attention_mask)

        gen_batch = DataProto.from_dict({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        })
        if meta_info is not None:
            gen_batch.meta_info = meta_info
        return gen_batch

    def _genrm_feedback_and_score(self, batch: DataProto, turn: int = 0):
        """Call GenRM and parse output into (feedback_text, score) per sample.

        Args:
            batch: The current batch with responses merged in.
            turn: Current turn index (0-based). Used to select prompt template.

        Returns:
            feedbacks: list[str | None] — extracted feedback text per sample.
            scores: list[float | None] — normalized score per sample.
            extra_info: dict with score distributions and parse stats.
        """
        prompt_max_length = self.config.genrm.rollout.get("prompt_length", 32768)
        norm_strategy = self.config.custom_reward_function.get("norm_strategy", "extreme_only_clipped")
        do_think_strip = self.config.custom_reward_function.get("think_strip", False)

        # Choose prompt template — optionally different for turn 2+
        feedback_template_path = self.config.custom_reward_function.get(
            "feedback_prompt_template_path", None)
        if turn > 0 and feedback_template_path is not None:
            system_prompt, user_prompt = self._load_prompt_template(feedback_template_path)
        else:
            system_prompt, user_prompt = self._load_prompt_template(
                self.config.custom_reward_function.prompt_template_path)

        input_ids_lst = []
        attention_mask_lst = []
        position_ids_lst = []

        for i, batch_item in enumerate(batch):
            response_str = self.tokenizer.decode(batch_item.batch['responses'], skip_special_tokens=True)            
            if do_think_strip:
                if "</think>" in response_str:
                    response_str = response_str.split("</think>")[-1].strip().lstrip()
                else:
                    response_str = " "

            extra_info = batch_item.non_tensor_batch['extra_info']
            question = extra_info['question']
            reference_solution = extra_info['solution']

            user_prompt_formatted = user_prompt.format(
                problem=question,
                reasoning_trace=response_str,
                reference_solution=reference_solution)
            prompts = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt_formatted}]

            prompt_with_chat_template = self.tokenizer.apply_chat_template(
                prompts, add_generation_prompt=True, tokenize=False)
            input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
                prompt=prompt_with_chat_template,
                tokenizer=self.tokenizer,
                max_length=prompt_max_length,
                pad_token_id=self.tokenizer.pad_token_id,
                left_pad=True,
                truncation="right")
            position_ids = compute_position_id_with_mask(attention_mask)
            input_ids_lst.append(input_ids)
            attention_mask_lst.append(attention_mask)
            position_ids_lst.append(position_ids)

        genrm_prompts = DataProto.from_dict({
            "input_ids": torch.cat(input_ids_lst, dim=0),
            "attention_mask": torch.cat(attention_mask_lst, dim=0),
            "position_ids": torch.cat(position_ids_lst, dim=0)})
        genrm_prompts.meta_info["do_sample"] = False

        genrm_output = self.genrm_wg.generate_sequences(genrm_prompts)
        responses = genrm_output.batch["responses"]

        feedbacks = []
        scores = []
        reward_dist = {}
        raw_score_dist = {}
        parse_failures = 0

        for i, response in enumerate(responses):
            response_str = self.tokenizer.decode(response, skip_special_tokens=True)
            feedback = None
            score = None
            raw_score = 'NA'

            try:
                # Extract feedback (text between "Feedback:" and "Score:")
                if "Feedback:" in response_str and "Score:" in response_str:
                    feedback = response_str.split("Feedback:")[1].split("Score:")[0].strip()
                elif "Feedback:" in response_str:
                    feedback = response_str.split("Feedback:")[1].strip()

                # Extract score
                if "Score:" in response_str:
                    score_str = response_str.split("Score:")[1].lstrip().split("\n")[0]
                elif "Score" in response_str:
                    score_str = response_str.split("Score")[1].lstrip().split("\n")[0]
                else:
                    raise ValueError(f"Score not found in response: {response_str}")

                score_str = score_str.replace("*", "").lstrip()[0]
                raw_score = int(score_str[0])
                score = self._score_normalization(raw_score, norm_strategy)
            except Exception as e:
                parse_failures += 1
                feedback = None
                score = None
                raw_score = 'NA'
            
            if raw_score not in raw_score_dist:
                raw_score_dist[raw_score] = 0
            raw_score_dist[raw_score] += 1

            if score is not None:
                if score not in reward_dist:
                    reward_dist[score] = 0
                reward_dist[score] += 1

            feedbacks.append(feedback)
            scores.append(score)

        extra_info = {
            "reward_dist": reward_dist,
            "raw_score_dist": raw_score_dist,
            "parse_failures": parse_failures,
        }
        if parse_failures > 0:
            print(f"[multi-turn] GenRM parse failures at turn {turn}: {parse_failures}/{len(batch)}")

        return feedbacks, scores, extra_info

    def _build_multi_turn_training_batch(
        self,
        original_batch: DataProto,
        prompt_ids_list: list[torch.Tensor],
        response_ids_per_turn: list[list[torch.Tensor]],
        bridge_ids_per_turn: list[list[torch.Tensor]],
    ) -> DataProto:
        """Assemble the final PPO-ready batch from multi-turn segments.

        Builds per sample:
            input_ids     = [prompt | R1 | bridge1 | R2 | bridge2 | ... | RT | pad]
            attention_mask = 1 for real tokens, 0 for padding
            response_mask  = 1 for actor response tokens, 0 for everything else
        """
        batch_size = len(prompt_ids_list)

        # Compute max total sequence length across samples
        total_lengths = []
        for i in range(batch_size):
            total_len = prompt_ids_list[i].numel()
            for t in range(len(response_ids_per_turn[i])):
                total_len += response_ids_per_turn[i][t].numel()
                if t < len(bridge_ids_per_turn[i]):
                    total_len += bridge_ids_per_turn[i][t].numel()
            total_lengths.append(total_len)

        max_seq_len = max(total_lengths)
        pad_id = self.tokenizer.pad_token_id

        all_input_ids = []
        all_attention_mask = []
        all_response_mask = []

        for i in range(batch_size):
            segments = []
            mask_segments = []

            # Original prompt (mask=0)
            prompt = prompt_ids_list[i]
            segments.append(prompt)
            mask_segments.append(torch.zeros(prompt.numel(), dtype=torch.long))

            for t in range(len(response_ids_per_turn[i])):
                # Actor response (mask=1)
                resp = response_ids_per_turn[i][t]
                segments.append(resp)
                mask_segments.append(torch.ones(resp.numel(), dtype=torch.long))

                # Bridge: AFTER_RESPONSE + feedback_ids + AFTER_FEEDBACK
                # The leading EOT tokens of the bridge (stripped from the response by
                # _strip_trailing_eos) should have mask=1 so the model learns to produce them.
                if t < len(bridge_ids_per_turn[i]):
                    bridge = bridge_ids_per_turn[i][t]
                    eot_len = min(self.RESPONSE_EOT_LEN, bridge.numel())
                    segments.append(bridge)
                    mask_segments.append(torch.cat([
                        torch.ones(eot_len, dtype=torch.long),
                        torch.zeros(bridge.numel() - eot_len, dtype=torch.long),
                    ]))

            full_ids = torch.cat(segments)
            full_response_mask = torch.cat(mask_segments)

            # Right-pad to max_seq_len
            seq_len = full_ids.numel()
            pad_len = max_seq_len - seq_len
            if pad_len > 0:
                full_ids = torch.cat([full_ids, torch.full((pad_len,), pad_id, dtype=torch.long)])
                full_response_mask = torch.cat([full_response_mask, torch.zeros(pad_len, dtype=torch.long)])

            attn_mask = (full_ids != pad_id).long()
            # Ensure padding positions have mask 0 even if pad_id appears in content
            attn_mask[seq_len:] = 0

            all_input_ids.append(full_ids.unsqueeze(0))
            all_attention_mask.append(attn_mask.unsqueeze(0))
            all_response_mask.append(full_response_mask.unsqueeze(0))

        input_ids = torch.cat(all_input_ids, dim=0)
        attention_mask = torch.cat(all_attention_mask, dim=0)
        response_mask = torch.cat(all_response_mask, dim=0)
        position_ids = compute_position_id_with_mask(attention_mask)

        # Split into prompts and responses tensors for DataProto compatibility
        # "prompts" = the original prompt portion, "responses" = everything after
        max_prompt_len = max(p.numel() for p in prompt_ids_list)
        # max_response_len must be the actual max response portion across samples,
        # not max_seq_len - max_prompt_len (which underestimates for short-prompt samples)
        max_response_len = max(total_lengths[i] - prompt_ids_list[i].numel() for i in range(batch_size))

        # Rebuild with left-padded prompt + right-padded response layout
        all_prompts = []
        all_responses = []
        all_input_ids_aligned = []
        all_attention_mask_aligned = []
        all_response_mask_aligned = []

        for i in range(batch_size):
            prompt_len = prompt_ids_list[i].numel()
            prompt_pad = max_prompt_len - prompt_len

            # Left-pad the prompt portion
            prompt_padded = torch.cat([
                torch.full((prompt_pad,), pad_id, dtype=torch.long),
                prompt_ids_list[i]
            ])
            prompt_attn = torch.cat([
                torch.zeros(prompt_pad, dtype=torch.long),
                torch.ones(prompt_len, dtype=torch.long)
            ])

            # The response portion = everything after the original prompt
            resp_start = prompt_len  # index in the un-padded full sequence
            full_ids_unpadded_len = total_lengths[i]
            resp_portion = all_input_ids[i][0, prompt_len:full_ids_unpadded_len]
            resp_mask_portion = all_response_mask[i][0, prompt_len:full_ids_unpadded_len]
            resp_attn_portion = torch.ones(resp_portion.numel(), dtype=torch.long)

            # Right-pad response portion
            resp_pad = max_response_len - resp_portion.numel()
            if resp_pad > 0:
                resp_portion = torch.cat([resp_portion, torch.full((resp_pad,), pad_id, dtype=torch.long)])
                resp_mask_portion = torch.cat([resp_mask_portion, torch.zeros(resp_pad, dtype=torch.long)])
                resp_attn_portion = torch.cat([resp_attn_portion, torch.zeros(resp_pad, dtype=torch.long)])

            full_input_ids = torch.cat([prompt_padded, resp_portion])
            full_attn_mask = torch.cat([prompt_attn, resp_attn_portion])
            all_prompts.append(prompt_padded.unsqueeze(0))
            all_responses.append(resp_portion.unsqueeze(0))
            all_input_ids_aligned.append(full_input_ids.unsqueeze(0))
            all_attention_mask_aligned.append(full_attn_mask.unsqueeze(0))
            # response_mask covers only the response portion (not prompt), matching verl convention
            all_response_mask_aligned.append(resp_mask_portion.unsqueeze(0))

        batch_dict = {
            "input_ids": torch.cat(all_input_ids_aligned, dim=0),
            "attention_mask": torch.cat(all_attention_mask_aligned, dim=0),
            "position_ids": compute_position_id_with_mask(torch.cat(all_attention_mask_aligned, dim=0)),
            "response_mask": torch.cat(all_response_mask_aligned, dim=0),
            "prompts": torch.cat(all_prompts, dim=0),
            "responses": torch.cat(all_responses, dim=0),
        }
        result = DataProto.from_dict(batch_dict)

        # Carry over non_tensor_batch from original batch
        for key in original_batch.non_tensor_batch:
            result.non_tensor_batch[key] = original_batch.non_tensor_batch[key]

        return result

    def _build_multi_turn_training_batch_last_turn_only(
        self,
        original_batch: DataProto,
        prompt_ids_list: list[torch.Tensor],
        response_ids_per_turn: list[list[torch.Tensor]],
        bridge_ids_per_turn: list[list[torch.Tensor]],
    ) -> DataProto:
        """Assemble training batch where only the last turn is trained on.

        Same token layout as vanilla:
            input_ids = [prompt | R1 | bridge1 | R2 | ... | RT | pad]
        But masks differ:
            attention_mask = 1 for original prompt + last response only
            response_mask  = 1 for last response tokens only

        This trains the model to directly produce the final (feedback-improved)
        response given only the original prompt.
        """
        batch_size = len(prompt_ids_list)
        pad_id = self.tokenizer.pad_token_id

        # Compute total lengths per sample
        total_lengths = []
        for i in range(batch_size):
            total_len = prompt_ids_list[i].numel()
            for t in range(len(response_ids_per_turn[i])):
                total_len += response_ids_per_turn[i][t].numel()
                if t < len(bridge_ids_per_turn[i]):
                    total_len += bridge_ids_per_turn[i][t].numel()
            total_lengths.append(total_len)

        max_seq_len = max(total_lengths)

        all_input_ids = []
        all_response_mask = []
        all_attn_mask = []

        for i in range(batch_size):
            prompt = prompt_ids_list[i]
            prompt_len = prompt.numel()

            # Build token sequence (same segments as vanilla)
            segments = [prompt]
            for t in range(len(response_ids_per_turn[i])):
                segments.append(response_ids_per_turn[i][t])
                if t < len(bridge_ids_per_turn[i]):
                    segments.append(bridge_ids_per_turn[i][t])

            full_ids = torch.cat(segments)
            seq_len = full_ids.numel()

            # Find where the last response starts in the full sequence
            offset = prompt_len
            for t in range(len(response_ids_per_turn[i])):
                last_resp_start = offset
                offset += response_ids_per_turn[i][t].numel()
                if t < len(bridge_ids_per_turn[i]):
                    offset += bridge_ids_per_turn[i][t].numel()
            last_resp_end = last_resp_start + response_ids_per_turn[i][-1].numel()

            # response_mask: 1 only for last response
            resp_mask = torch.zeros(seq_len, dtype=torch.long)
            resp_mask[last_resp_start:last_resp_end] = 1

            # attention_mask: 1 for prompt + last response only
            attn_mask = torch.zeros(seq_len, dtype=torch.long)
            attn_mask[:prompt_len] = 1
            attn_mask[last_resp_start:last_resp_end] = 1

            # Right-pad
            pad_len = max_seq_len - seq_len
            if pad_len > 0:
                full_ids = torch.cat([full_ids, torch.full((pad_len,), pad_id, dtype=torch.long)])
                resp_mask = torch.cat([resp_mask, torch.zeros(pad_len, dtype=torch.long)])
                attn_mask = torch.cat([attn_mask, torch.zeros(pad_len, dtype=torch.long)])

            all_input_ids.append(full_ids.unsqueeze(0))
            all_response_mask.append(resp_mask.unsqueeze(0))
            all_attn_mask.append(attn_mask.unsqueeze(0))

        # Second pass: left-padded prompt + right-padded response alignment
        max_prompt_len = max(p.numel() for p in prompt_ids_list)
        max_response_len = max(total_lengths[i] - prompt_ids_list[i].numel() for i in range(batch_size))

        all_prompts = []
        all_responses = []
        all_input_ids_aligned = []
        all_attention_mask_aligned = []
        all_response_mask_aligned = []

        for i in range(batch_size):
            prompt_len = prompt_ids_list[i].numel()
            prompt_pad = max_prompt_len - prompt_len

            prompt_padded = torch.cat([
                torch.full((prompt_pad,), pad_id, dtype=torch.long),
                prompt_ids_list[i]
            ])
            prompt_attn = torch.cat([
                torch.zeros(prompt_pad, dtype=torch.long),
                torch.ones(prompt_len, dtype=torch.long)
            ])

            # Extract response portion and its masks from first pass
            resp_portion = all_input_ids[i][0, prompt_len:total_lengths[i]]
            resp_mask_portion = all_response_mask[i][0, prompt_len:total_lengths[i]]
            resp_attn_portion = all_attn_mask[i][0, prompt_len:total_lengths[i]]

            # Right-pad response portion
            resp_pad = max_response_len - resp_portion.numel()
            if resp_pad > 0:
                resp_portion = torch.cat([resp_portion, torch.full((resp_pad,), pad_id, dtype=torch.long)])
                resp_mask_portion = torch.cat([resp_mask_portion, torch.zeros(resp_pad, dtype=torch.long)])
                resp_attn_portion = torch.cat([resp_attn_portion, torch.zeros(resp_pad, dtype=torch.long)])

            full_input_ids = torch.cat([prompt_padded, resp_portion])
            full_attn_mask = torch.cat([prompt_attn, resp_attn_portion])
            all_prompts.append(prompt_padded.unsqueeze(0))
            all_responses.append(resp_portion.unsqueeze(0))
            all_input_ids_aligned.append(full_input_ids.unsqueeze(0))
            all_attention_mask_aligned.append(full_attn_mask.unsqueeze(0))
            all_response_mask_aligned.append(resp_mask_portion.unsqueeze(0))

        batch_dict = {
            "input_ids": torch.cat(all_input_ids_aligned, dim=0),
            "attention_mask": torch.cat(all_attention_mask_aligned, dim=0),
            "position_ids": compute_position_id_with_mask(torch.cat(all_attention_mask_aligned, dim=0)),
            "response_mask": torch.cat(all_response_mask_aligned, dim=0),
            "prompts": torch.cat(all_prompts, dim=0),
            "responses": torch.cat(all_responses, dim=0),
        }

        result = DataProto.from_dict(batch_dict)

        for key in original_batch.non_tensor_batch:
            result.non_tensor_batch[key] = original_batch.non_tensor_batch[key]

        return result

    def _build_multi_turn_training_batch_expanded(
        self,
        original_batch: DataProto,
        prompt_ids_list: list[torch.Tensor],
        response_ids_per_turn: list[list[torch.Tensor]],
        bridge_ids_per_turn: list[list[torch.Tensor]],
    ) -> DataProto:
        """Expanded batch: vanilla + last-turn-only for each sample (2x batch size).

        First half uses vanilla masks (loss on all actor responses across all turns).
        Second half uses last-turn-only masks (loss only on the last response,
        attending to only the original prompt).
        """
        vanilla_batch = self._build_multi_turn_training_batch(
            original_batch, prompt_ids_list, response_ids_per_turn, bridge_ids_per_turn)

        last_turn_batch = self._build_multi_turn_training_batch_last_turn_only(
            original_batch, prompt_ids_list, response_ids_per_turn, bridge_ids_per_turn)

        # Convert to plain dicts to avoid TensorDict dispatch issues
        v_dict = dict(vanilla_batch.batch.items())
        l_dict = dict(last_turn_batch.batch.items())
        batch_dict = {key: torch.cat([v_dict[key], l_dict[key]], dim=0) for key in v_dict}
        result = DataProto.from_dict(batch_dict)

        for key in vanilla_batch.non_tensor_batch:
            v = vanilla_batch.non_tensor_batch[key]
            l = last_turn_batch.non_tensor_batch[key]
            if isinstance(v, np.ndarray):
                result.non_tensor_batch[key] = np.concatenate([v, l])
            elif isinstance(v, list):
                result.non_tensor_batch[key] = v + l
            else:
                result.non_tensor_batch[key] = v

        return result

    def _aggregate_turn_scores(
        self,
        final_batch: DataProto,
        scores_per_turn: list[list[float | None]],
        response_ids_per_turn: list[list[torch.Tensor]],
        prompt_ids_list: list[torch.Tensor],
        bridge_ids_per_turn: list[list[torch.Tensor]],
    ):
        """Aggregate per-turn GenRM scores into a reward_tensor.

        Args:
            final_batch: The assembled multi-turn training batch.
            scores_per_turn: scores_per_turn[i][t] = score for sample i, turn t.
            response_ids_per_turn: token IDs per turn (for computing segment offsets).
            prompt_ids_list: original prompt token IDs (for computing offsets).
            bridge_ids_per_turn: bridge token IDs per turn (for computing offsets).

        Returns:
            reward_tensor: Shape matches final_batch response_mask.
            extra_info: dict with aggregation metrics.
        """
        aggregation = self.config.custom_reward_function.get(
            "multi_turn_score_aggregation", "last")
        batch_size = len(scores_per_turn)
        response_mask = final_batch.batch["response_mask"]
        reward_tensor = torch.zeros_like(response_mask, dtype=torch.float32)

        use_reward = [True] * batch_size
        reward_dist = {}
        per_turn_avg_scores = defaultdict(list)

        for i in range(batch_size):
            num_turns = len(scores_per_turn[i])
            valid_scores = [(t, s) for t, s in enumerate(scores_per_turn[i]) if s is not None]

            if len(valid_scores) == 0:
                use_reward[i] = False
                # reward_tensor is already zeros for this sample;
                # keep response_mask intact so GRPO normalization still works
                continue

            # Track per-turn score averages
            for t, s in valid_scores:
                per_turn_avg_scores[t].append(s)

            # Compute the offset of the last response token for each turn
            # response_mask and reward_tensor cover only the response portion (0-based)
            # Structure: R1 | bridge1 | R2 | bridge2 | ... | RT
            turn_end_offsets = []
            resp_offset = 0
            for t in range(num_turns):
                resp_len = response_ids_per_turn[i][t].numel()
                turn_end = resp_offset + resp_len - 1
                turn_end_offsets.append(turn_end)
                resp_offset += resp_len
                if t < len(bridge_ids_per_turn[i]):
                    resp_offset += bridge_ids_per_turn[i][t].numel()

            if aggregation == "last":
                final_score = valid_scores[-1][1]
                last_turn_idx = valid_scores[-1][0]
                reward_tensor[i, turn_end_offsets[last_turn_idx]] = final_score
            elif aggregation == "sum":
                total_score = sum(s for _, s in valid_scores)
                last_turn_idx = valid_scores[-1][0]
                reward_tensor[i, turn_end_offsets[last_turn_idx]] = total_score
            elif aggregation == "per_turn":
                for t, s in valid_scores:
                    reward_tensor[i, turn_end_offsets[t]] = s
            elif aggregation == "delta":
                if len(valid_scores) >= 2:
                    delta = valid_scores[-1][1] - valid_scores[0][1]
                else:
                    delta = valid_scores[0][1]
                last_turn_idx = valid_scores[-1][0]
                reward_tensor[i, turn_end_offsets[last_turn_idx]] = delta
            else:
                raise ValueError(f"Unknown score aggregation: {aggregation}")

            # Track reward distribution
            final_score_val = reward_tensor[i].sum().item()
            if final_score_val not in reward_dist:
                reward_dist[final_score_val] = 0
            reward_dist[final_score_val] += 1

        extra_info = {
            "use_reward": use_reward,
            "reward_dist": reward_dist,
            "raw_score_dist": {},
            "clip_reward_for_length_lst": np.array([False] * batch_size),
        }

        # Per-turn metrics
        for t, score_list in per_turn_avg_scores.items():
            extra_info[f"multi_turn/turn_{t}_score_mean"] = np.mean(score_list)
            extra_info[f"multi_turn/turn_{t}_score_std"] = np.std(score_list)

        extra_info["multi_turn/use_reward_rate"] = np.mean(use_reward)

        return reward_tensor, extra_info

    def _multi_turn_generate_and_score(self, gen_batch: DataProto, batch: DataProto,
                                        max_turns: int, timing_raw: dict):
        """Multi-turn generation loop with GenRM feedback.

        For each turn:
          1. Generate response via actor
          2. Call GenRM -> parse feedback + score
          3. Build next turn's prompt via token concatenation

        After all turns, assemble the final PPO training batch.
        """
        batch_size = len(gen_batch.batch)

        # Extract un-padded prompt ids from the initial gen_batch
        prompt_ids_list = []
        for i in range(batch_size):
            ids = gen_batch.batch['input_ids'][i]
            mask = gen_batch.batch['attention_mask'][i]
            prompt_ids_list.append(ids[mask.bool()])

        response_ids_per_turn = [[] for _ in range(batch_size)]
        bridge_ids_per_turn = [[] for _ in range(batch_size)]
        scores_per_turn = [[] for _ in range(batch_size)]

        # Track current prompt ids per sample (grows each turn)
        current_prompt_ids = [p.clone() for p in prompt_ids_list]

        gen_meta_info = {
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
            "recompute_log_prob": False,
            "do_sample": True,
        }

        # All samples get exactly max_turns assistant generations.
        # GenRM is called on the first max_turns-1 turns (to produce feedback for the next turn).
        # The last turn is generation-only (no feedback / bridge needed).
        for turn in range(max_turns):
            is_last_turn = (turn == max_turns - 1)

            with marked_timer(f"mt_gen_turn_{turn}", timing_raw):
                # Build gen_batch for this turn
                if turn > 0:
                    active_prompts = []
                    for i in range(batch_size):
                        active_prompts.append(current_prompt_ids[i])
                    gen_batch = self._pad_prompts_to_gen_batch(active_prompts, meta_info=gen_meta_info)
                else:
                    gen_batch.meta_info.update(gen_meta_info)

                # Generate
                gen_output = self.actor_rollout_wg.generate_sequences(gen_batch)

            with marked_timer(f"mt_genrm_turn_{turn}", timing_raw):
                # Record response ids (raw tokens, never decoded for reuse)
                for i in range(batch_size):
                    resp = gen_output.batch['responses'][i]
                    # Strip padding from response
                    resp_mask = gen_output.batch.get('attention_mask', None)
                    if resp_mask is not None:
                        resp_len = gen_output.batch['attention_mask'][i, -resp.shape[0]:].sum().item()
                        resp = resp[:int(resp_len)]
                    response_ids_per_turn[i].append(resp)

                # Build a temporary batch for GenRM evaluation
                temp_batch = batch.union(gen_output) if turn == 0 else self._build_temp_genrm_batch(
                    batch, gen_batch, gen_output)

                # Call GenRM on every turn for scoring
                feedbacks, turn_scores, genrm_extra = self._genrm_feedback_and_score(temp_batch, turn=turn)
                for i in range(batch_size):
                    scores_per_turn[i].append(turn_scores[i])

            # Build bridges and next-turn prompts only on non-last turns
            # (last turn's response is NOT appended to input_ids — only scored)
            if not is_last_turn:
                for i in range(batch_size):
                    feedback = feedbacks[i]

                    # Build bridge ids for this turn
                    if self.feedback_user_template is not None:
                        user_content = self.feedback_user_template.render(feedback=feedback)
                    else:
                        user_content = feedback
                    user_content_ids = torch.tensor(
                        self.tokenizer.encode(user_content, add_special_tokens=False),
                        dtype=torch.long)
                    bridge = torch.cat([self.AFTER_RESPONSE, user_content_ids, self.AFTER_FEEDBACK])
                    bridge_ids_per_turn[i].append(bridge)

                    # Build next turn's prompt via token concatenation
                    new_prompt = self._build_next_turn_prompt(
                        current_prompt_ids[i], response_ids_per_turn[i][-1], feedback)

                    current_prompt_ids[i] = new_prompt
                    print(self.tokenizer.decode(new_prompt))

        # Strip EOS from response ids for training batch assembly
        # (bridge already includes EOT, and final response keeps its natural ending)
        response_ids_for_batch = []
        for i in range(batch_size):
            sample_resps = []
            for t in range(len(response_ids_per_turn[i])):
                resp = response_ids_per_turn[i][t]
                if resp.numel() > 0 and t < len(bridge_ids_per_turn[i]):
                    # Intermediate response: strip EOS (bridge has EOT)
                    resp = self._strip_trailing_eos(resp)
                # Final response: keep as-is (model's natural stopping point)
                sample_resps.append(resp)
            response_ids_for_batch.append(sample_resps)

        # Build final training batch based on batch_mode
        batch_mode = self.config.custom_reward_function.get("multi_turn_batch_mode", "vanilla")
        build_args = (batch, prompt_ids_list, response_ids_for_batch, bridge_ids_per_turn)

        if batch_mode == "vanilla":
            final_batch = self._build_multi_turn_training_batch(*build_args)
        elif batch_mode == "last_turn_only":
            final_batch = self._build_multi_turn_training_batch_last_turn_only(*build_args)
        elif batch_mode == "expanded":
            final_batch = self._build_multi_turn_training_batch_expanded(*build_args)
        else:
            raise ValueError(f"Unknown multi_turn_batch_mode: {batch_mode}")
        
        # Aggregate scores (on original batch_size, then duplicate for expanded)
        reward_tensor, extra_info = self._aggregate_turn_scores(
            final_batch.slice(end=batch_size) if batch_mode == "expanded" else final_batch,
            scores_per_turn, response_ids_for_batch,
            prompt_ids_list, bridge_ids_per_turn)

        if batch_mode == "expanded":
            reward_tensor = torch.cat([reward_tensor, reward_tensor], dim=0)

        return final_batch, reward_tensor, extra_info

    def _build_temp_genrm_batch(self, original_batch: DataProto,
                                 gen_batch: DataProto,
                                 gen_output: DataProto) -> DataProto:
        """Build a temporary batch for GenRM evaluation on turn > 0.

        Merges the generation output with necessary fields from the original batch
        so that _genrm_feedback_and_score can access extra_info, responses, etc.
        """
        # Start from gen_output (has full input_ids, attention_mask, responses, etc.)
        # Don't union with gen_batch — they share keys (input_ids) with different values
        result = gen_output

        # Carry over non_tensor_batch from original (extra_info, data_source, etc.)
        for key in original_batch.non_tensor_batch:
            if key not in result.non_tensor_batch:
                result.non_tensor_batch[key] = original_batch.non_tensor_batch[key]

        # Ensure response_mask is present
        if "response_mask" not in result.batch.keys():
            result.batch["response_mask"] = compute_response_mask(result)

        return result

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
                with marked_timer("step", timing_raw):

                    # Assign UIDs before repeat
                    batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                    )
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                    if self.use_genrm:
                        # ============================================================
                        # Multi-turn GenRM path: generate → GenRM feedback → ... → score
                        # ============================================================
                        with marked_timer("multi_turn_gen_and_score", timing_raw, color="red"):
                            batch, reward_tensor, reward_extra_infos_dict = \
                                self._multi_turn_generate_and_score(
                                    gen_batch, batch, self.max_turns, timing_raw)

                        # compute outcome reward with rule-based reward fn (for metrics only)
                        outcome_score_tensor, _ = compute_reward(batch, self.reward_fn)
                    else:
                        # ============================================================
                        # Standard path: generate → rule-based reward
                        # ============================================================
                        with marked_timer("gen", timing_raw, color="red"):
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                            if "timing" in gen_batch_output.meta_info:
                                timing_raw.update(gen_batch_output.meta_info["timing"])
                                gen_batch_output.meta_info.pop("timing", None)

                        batch = batch.union(gen_batch_output)

                        if "response_mask" not in batch.batch.keys():
                            batch.batch["response_mask"] = compute_response_mask(batch)

                        with marked_timer("reward", timing_raw, color="yellow"):
                            reward_tensor, _ = compute_reward(batch, self.reward_fn)
                            outcome_score_tensor = reward_tensor
                            reward_extra_infos_dict = {}

                    # Balance batch across DP ranks
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(
                        batch.batch["attention_mask"], dim=-1).tolist()
                    
                    # recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
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
                        batch.batch["token_level_scores"] = reward_tensor

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor
                        last_step_reward = reward_extra_infos_dict.get("last_step_reward", None)

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                            last_step_reward=last_step_reward,
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
                            batch.meta_info["multi_turn"] = self.use_genrm and self.max_turns > 1
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
                # update extra reward info
                additional_reward_metrics = {}
                additional_reward_metrics['reward/outcome_reward_mean'] = outcome_score_tensor.sum(-1).mean().item()

                if reward_extra_infos_dict:
                    reward_dist = reward_extra_infos_dict.get("reward_dist", {})
                    use_reward = reward_extra_infos_dict.get("use_reward", [])
                    raw_score_dist = reward_extra_infos_dict.get("raw_score_dist", {})
                    clip_reward_for_length_lst = reward_extra_infos_dict.get("clip_reward_for_length_lst", [])
                    additional_reward_metrics.update({
                        f"reward/value_{i}": reward_dist[i] for i in reward_dist.keys()
                        })
                    if np.any(clip_reward_for_length_lst):
                        additional_reward_metrics["reward/clip_reward_for_length_mean"] = np.mean(clip_reward_for_length_lst)
                    if use_reward:
                        additional_reward_metrics['reward/use_reward_mean'] = np.mean(use_reward)
                    additional_reward_metrics.update({
                        f"reward/raw_score_{i}": raw_score_dist[i] for i in raw_score_dist.keys()
                        })

                # track all 1s, 0s, and some non-0s scores
                uids = batch.non_tensor_batch["uid"]
                outcome_score_by_uid = defaultdict(list)
                for uid, outcome_score in zip(uids, outcome_score_tensor):
                    outcome_score_by_uid[uid].append(outcome_score.sum().item())
                outcome_score_grouped = np.array([np.mean(scores) for _, scores in outcome_score_by_uid.items()])
                all_ones = np.mean(outcome_score_grouped == 1)
                all_zeros = np.mean(outcome_score_grouped == 0)
                some_non_zeros = np.mean((0 < outcome_score_grouped) & (outcome_score_grouped < 1))
                additional_reward_metrics["reward/score_all_1s"] = all_ones
                additional_reward_metrics["reward/score_all_0s"] = all_zeros
                additional_reward_metrics["reward/score_some_non_0s"] = some_non_zeros

                # multi-turn per-turn metrics (from _aggregate_turn_scores)
                for key, val in reward_extra_infos_dict.items():
                    if key.startswith("multi_turn/"):
                        additional_reward_metrics[f"reward/{key}"] = val

                metrics.update(additional_reward_metrics)
                # collect metrics — guard against all-zero response_mask
                if batch.batch["response_mask"].any():
                    metrics.update(
                        compute_data_metrics(batch=batch, use_critic=self.use_critic)
                    )
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
