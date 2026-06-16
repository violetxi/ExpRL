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
import pickle
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


class RayLogp:
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

    def _compute_logp_vi_scaled(self, batch: DataProto):
        """Compute the logp for the batch. We are assuming instruct model here bc we will prepend 
        a prefix string to the response, which is used to ONLY compute logp(y* |x, z) reward."""
        prompt_ids_lst = []
        prompt_masks_lst = []
        think_ids_lst = []
        think_masks_lst = []
        # reference ids for thinking trace
        reference_ids_lst = []
        reference_masks_lst = []
        # token ids for reference solution from a different prompt (identified by uid)
        other_reference_ids_lst = []
        other_reference_masks_lst = []
        # max length to track for each for padding
        max_prompt_length = 0
        max_think_length = 0
        max_reference_length = 0
        max_other_reference_length = 0
        # to get each token str in the reference and other reference
        decoded_reference_token_str_lst = []
        decoded_other_reference_token_str_lst = []
        ## split z from response based on </think> token
        end_of_think_token_id = self.tokenizer.encode("</think>", add_special_tokens=False)[0]
        response_length = batch.batch['responses'].shape[-1]
        # uid list to identify the prompt
        uid_lst = batch.non_tensor_batch['uid']
        has_end_of_think_lst = []
        for i, batch_item in enumerate(batch):
            response_str = self.tokenizer.decode(
                torch.masked_select(batch_item.batch['responses'], batch_item.batch['response_mask'].bool()))
            has_end_of_think = response_str.find("</think>")
            has_end_of_think_lst.append(has_end_of_think)
            # reference - y*, append prefix if thinking is not finished
            reference = batch_item.non_tensor_batch['extra_info'][self.config.data.reference_key]
            # if <think> </think> in reference, then only take what comes after </think>
            if "<think>" in reference and "</think>" in reference:
                reference = reference.split("</think>")[1].strip().lstrip()
            else:
                reference = ''
            
            if has_end_of_think == -1:    # did not finish thinking
                reference_str = Template("Considering the limited time by the user, I have to give the solution based on the thinking directly now \n</think>\n\n{{reference}}").render(reference=reference.strip().lstrip())
            else:
                reference_str = f"\n\n{reference.strip().lstrip()}"
            reference_token_dict = self.tokenizer(reference_str, return_tensors="pt")
            reference_ids = reference_token_dict["input_ids"][0]
            reference_masks = reference_token_dict["attention_mask"][0]
            max_reference_length = max([max_reference_length, reference_masks.sum().item()])
            # other reference - y^
            current_uid = uid_lst[i]
            not_current_uid = uid_lst != current_uid
            # randomly select a position in not_current_uid
            other_reference_items = batch.non_tensor_batch['extra_info'][not_current_uid]
            # randomly select one item
            other_reference = other_reference_items[np.random.randint(0, len(other_reference_items))][self.config.data.reference_key]
            if "<think>" in other_reference and "</think>" in other_reference:
                other_reference = other_reference.split("</think>")[1].strip().lstrip()
            else:
                other_reference = ''
            # assert reference != other_reference, "reference and other reference should be different"
            
            if has_end_of_think == -1:
                other_reference_str = Template("Considering the limited time by the user, I have to give the solution based on the thinking directly now \n</think>\n\n{{reference}}").render(reference=other_reference)
            else:
                other_reference_str = f"\n\n{other_reference}"
            other_reference_token_dict = self.tokenizer(other_reference_str, return_tensors="pt")
            other_reference_ids = other_reference_token_dict["input_ids"][0]
            other_reference_masks = other_reference_token_dict["attention_mask"][0]
            max_other_reference_length = max([max_other_reference_length, other_reference_masks.sum().item()])
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
            other_reference_ids_lst.append(other_reference_ids)
            other_reference_masks_lst.append(other_reference_masks)
            # decode each token in the reference_ids to get all the token strings
            token_str_lst = []
            for j, reference_id in enumerate(reference_ids):
                # if thinking trace did not end with </think>, then skip the first 22 tokens
                if has_end_of_think_lst[i] == -1 and j <= 22:
                    continue
                token_str_lst.append(self.tokenizer.decode(reference_id))
            decoded_reference_token_str_lst.append(token_str_lst)
            # decode each token in the other_reference_ids to get all the token strings
            other_token_str_lst = []
            for j, other_reference_id in enumerate(other_reference_ids):
                # if thinking trace did not end with </think>, then skip the first 22 tokens
                if has_end_of_think_lst[i] == -1 and j <= 22:
                    continue
                other_token_str_lst.append(self.tokenizer.decode(other_reference_id))
            decoded_other_reference_token_str_lst.append(other_token_str_lst)
        
        ### pad prompt, input (prompt + think), reference, other reference
        padded_prompt_ids_lst = []
        padded_prompt_masks_lst = []
        padded_input_ids_lst = []
        padded_attention_masks_lst = []
        padded_reference_ids_lst = []
        padded_reference_masks_lst = []
        # other reference
        padded_other_input_ids_lst = []
        padded_other_attention_masks_lst = []
        padded_other_reference_ids_lst = []
        padded_other_reference_masks_lst = []
        pad_token_id = self.tokenizer.pad_token_id
        for i in range(len(prompt_ids_lst)):
            prompt_ids = prompt_ids_lst[i]
            prompt_masks = prompt_masks_lst[i]
            think_ids = think_ids_lst[i]
            think_masks = think_masks_lst[i]
            reference_ids = reference_ids_lst[i]
            reference_masks = reference_masks_lst[i]
            other_reference_ids = other_reference_ids_lst[i]
            other_reference_masks = other_reference_masks_lst[i]
            # left pad prompt
            padded_prompt_ids = torch.cat([
                torch.full((max_prompt_length - prompt_ids.shape[0],), 
                pad_token_id), prompt_ids])
            padded_prompt_masks = torch.cat([
                torch.full((max_prompt_length - prompt_masks.shape[0],), 
                0), prompt_masks])
            padded_prompt_ids_lst.append(padded_prompt_ids)
            padded_prompt_masks_lst.append(padded_prompt_masks)
            # left pad prompt + think to max prompt + think length
            prompt_padding_len = max_prompt_length - prompt_ids.shape[0] + max_think_length - think_ids.shape[0]
            padded_prompt_ids = torch.cat([torch.full((prompt_padding_len,), pad_token_id), prompt_ids])
            padded_prompt_masks = torch.cat([torch.full((prompt_padding_len,), 0), prompt_masks])
            # right pad reference so the total of think + reference = max reference length + max response length
            reference_padding_len = max_reference_length - reference_ids.shape[0]
            padded_reference_ids = torch.cat([reference_ids, torch.full((reference_padding_len,), pad_token_id)])
            padded_reference_masks = torch.cat([reference_masks, torch.full((reference_padding_len,), 0)])
            # right pad other reference so the total of think + other reference = max think length + max other reference length
            other_reference_padding_len = max_other_reference_length - other_reference_ids.shape[0]
            padded_other_reference_ids = torch.cat([other_reference_ids, torch.full((other_reference_padding_len,), pad_token_id)])
            padded_other_reference_masks = torch.cat([other_reference_masks, torch.full((other_reference_padding_len,), 0)])
            # now create input_ids and responses
            padded_input_ids = torch.cat([padded_prompt_ids, think_ids, padded_reference_ids])
            padded_attention_masks = torch.cat([padded_prompt_masks, think_masks, padded_reference_masks])
            padded_input_ids_lst.append(padded_input_ids)
            padded_attention_masks_lst.append(padded_attention_masks)
            padded_reference_ids_lst.append(padded_reference_ids)
            padded_reference_masks_lst.append(padded_reference_masks)
            # padded other input_ids and responses
            padded_other_input_ids = torch.cat([padded_prompt_ids, think_ids, padded_other_reference_ids])
            padded_other_attention_masks = torch.cat([padded_prompt_masks, think_masks, padded_other_reference_masks])
            padded_other_input_ids_lst.append(padded_other_input_ids)
            padded_other_attention_masks_lst.append(padded_other_attention_masks)
            padded_other_reference_ids_lst.append(padded_other_reference_ids)
            padded_other_reference_masks_lst.append(padded_other_reference_masks)
    
        padded_prompt_ids = torch.stack(padded_prompt_ids_lst)
        padded_prompt_masks = torch.stack(padded_prompt_masks_lst)
        padded_input_ids = torch.stack(padded_input_ids_lst)
        padded_attention_masks = torch.stack(padded_attention_masks_lst)
        padded_position_ids = compute_position_id_with_mask(padded_attention_masks)
        padded_reference_ids = torch.stack(padded_reference_ids_lst)
        padded_reference_masks = torch.stack(padded_reference_masks_lst)
        #### compute logp(y* |x, z)
        logp_z_batch = DataProto.from_single_dict({
            "input_ids": padded_input_ids,
            "attention_mask": padded_attention_masks,
            "position_ids": padded_position_ids,
            "responses": padded_reference_ids,
            "response_mask": padded_reference_masks,
        })
        logp_info = self.actor_rollout_wg.compute_log_prob(logp_z_batch)
        logp_z_lst = []
        for i, logp in enumerate(logp_info.batch['old_log_probs']):            
            logp_z = torch.masked_select(logp, logp_z_batch.batch['response_mask'][i].bool())
            if has_end_of_think_lst[i] == -1:
                logp_z = logp_z[22:]
            logp_z_lst.append(logp_z.cpu().numpy())
        
        #### compute logp(y* |x)
        logp_x_input_ids = torch.cat([padded_prompt_ids, padded_reference_ids], dim=1)
        logp_x_attention_mask = torch.cat([padded_prompt_masks, padded_reference_masks], dim=1)
        logp_x_position_ids = compute_position_id_with_mask(logp_x_attention_mask)
        logp_x_batch = DataProto.from_single_dict({
            "input_ids": logp_x_input_ids,
            "attention_mask": logp_x_attention_mask,
            "position_ids": logp_x_position_ids,
            "responses": padded_reference_ids,
            "response_mask": padded_reference_masks,
        })
        logp_x_info = self.actor_rollout_wg.compute_log_prob(logp_x_batch)
        logp_x_lst = []
        for i, logp in enumerate(logp_x_info.batch['old_log_probs']):
            logp_x = torch.masked_select(logp, logp_x_batch.batch['response_mask'][i].bool())
            if has_end_of_think_lst[i] == -1:
                logp_x = logp_x[22:]
            logp_x_lst.append(logp_x.cpu().numpy())
                
        #### compute logp(y* |x, z) - logp(y* |x)
        logp_diff_lst = []
        for logp_x, logp_z in zip(logp_x_lst, logp_z_lst):
            if has_end_of_think_lst[i] == -1:
                logp_z = logp_z[22:]
                logp_x = logp_x[22:]
            logp_diff_lst.append(logp_z - logp_x)

        # other reference (y^)
        padded_other_input_ids = torch.stack(padded_other_input_ids_lst)
        padded_other_attention_masks = torch.stack(padded_other_attention_masks_lst)
        padded_other_reference_ids = torch.stack(padded_other_reference_ids_lst)
        padded_other_reference_masks = torch.stack(padded_other_reference_masks_lst)
        # a batch to measure the logp(y^ |x, z) where y^ is solution from a different prompt (identified by uid)
        logp_other_y_input_ids = torch.cat([padded_other_input_ids, padded_other_reference_ids], dim=1)
        logp_other_y_attention_mask = torch.cat([padded_other_attention_masks, padded_other_reference_masks], dim=1)
        logp_other_y_position_ids = compute_position_id_with_mask(logp_other_y_attention_mask)
        logp_other_y_batch = DataProto.from_single_dict({
            "input_ids": logp_other_y_input_ids,
            "attention_mask": logp_other_y_attention_mask,
            "position_ids": logp_other_y_position_ids,
            "responses": padded_other_reference_ids,
            "response_mask": padded_other_reference_masks,
        })
        logp_other_y_info = self.actor_rollout_wg.compute_log_prob(logp_other_y_batch)
        logp_other_y_lst = []
        for i, logp in enumerate(logp_other_y_info.batch['old_log_probs']):
            logp_other_y = torch.masked_select(logp, logp_other_y_batch.batch['response_mask'][i].bool()).cpu().numpy()
            logp_other_y_lst.append(logp_other_y)
        return logp_z_lst, logp_x_lst, logp_other_y_lst, logp_diff_lst, \
            decoded_reference_token_str_lst, decoded_other_reference_token_str_lst

    def _compute_logp_vi_vanilla(self, batch: DataProto):
        prompt_ids_lst = []
        prompt_masks_lst = []        
        response_ids_lst = []
        response_masks_lst = []
        reference_ids_lst = []
        reference_masks_lst = []
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
            assert self.config.data.reference_key == "answer", "reference key must be answer to use answer as reference"            
            if "boxed{" in response_str:
                reference_str = reference + "}"
            else:
                reference_str = Template("<answer> \\boxed{ {{-reference-}} } </answer>").render(reference=reference)
            # if "<answer" in response_str:
            #     reference_str =  Template("> \\boxed{ {{-reference-}} } </answer>").render(reference=reference)
            # else:
            #     reference_str =  reference
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
            
        # re-do padding across all tensors
        padded_input_ids_lst = []
        padded_attention_masks_lst = []
        padded_reference_ids_lst = []
        padded_reference_masks_lst = []
        pad_token_id = self.tokenizer.pad_token_id
        for i in range(len(prompt_ids_lst)):
            prompt_id = prompt_ids_lst[i]
            prompt_mask = prompt_masks_lst[i]
            response_id = response_ids_lst[i]
            response_mask = response_masks_lst[i]
            reference_id = reference_ids_lst[i]
            reference_mask = reference_masks_lst[i]
            # left padding for prompt + response
            input_padding_len = max_prompt_len - prompt_id.shape[0] + max_response_len - response_id.shape[0]
            padded_input_ids = torch.cat([torch.full((input_padding_len,), pad_token_id), prompt_id, response_id])
            padded_attention_masks = torch.cat([torch.full((input_padding_len,), 0), prompt_mask, response_mask])
            padded_input_ids_lst.append(padded_input_ids)
            padded_attention_masks_lst.append(padded_attention_masks)
            # right padding for reference to max_reference_len
            reference_padding_len = max_reference_len - reference_id.shape[0]
            padded_reference_ids = torch.cat([reference_id, torch.full((reference_padding_len,), pad_token_id)])
            padded_reference_masks = torch.cat([reference_mask, torch.full((reference_padding_len,), 0)])
            padded_reference_ids_lst.append(padded_reference_ids)
            padded_reference_masks_lst.append(padded_reference_masks)
        
        padded_input_ids = torch.stack(padded_input_ids_lst)
        padded_attention_masks = torch.stack(padded_attention_masks_lst)        
        padded_reference_ids = torch.stack(padded_reference_ids_lst)
        padded_reference_masks = torch.stack(padded_reference_masks_lst)

        ##### for compute reward: 
        # - input_ids -> prompt_id + response_id + reference_id
        # - response_ids -> reference_id
        # - response_masks -> reference_mask
        logp_input_ids = torch.cat([padded_input_ids, padded_reference_ids], dim=1)
        logp_attention_mask = torch.cat([padded_attention_masks, padded_reference_masks], dim=1)
        logp_position_ids = compute_position_id_with_mask(logp_attention_mask)
        logp_responses = padded_reference_ids
        logp_response_mask = padded_reference_masks
        logp_batch = DataProto.from_single_dict({
            "input_ids": logp_input_ids,
            "attention_mask": logp_attention_mask,
            "position_ids": logp_position_ids,
            "responses": logp_responses,
            "response_mask": logp_response_mask
        })
        logprobs =self.actor_rollout_wg.compute_log_prob(logp_batch)
        logp_z_lst = []
        for i, logprob in enumerate(logprobs.batch['old_log_probs']):
            full_response_mask = logp_response_mask[i]
            effective_logp = torch.masked_select(logprob, full_response_mask.bool())
            logp_z_lst.append(effective_logp.cpu().numpy())
        return logp_z_lst

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
        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        self.max_steps_duration = 0                
        
        all_logp_z_lst = []
        all_logp_x_lst = []
        all_logp_other_y_lst = []
        all_logp_diff_lst = []
        all_decoded_token_lst = []
        all_decoded_other_token_lst = []
        all_reward_tensor_lst = []
        all_uid_lst = []
        for batch_dict in self.train_dataloader:
            metrics = {}
            timing_raw = {}
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
                # generate a batch
                with marked_timer("gen", timing_raw, color="red"):
                    if not self.async_rollout_mode:
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                    else:
                        gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)

                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )
                # repeat to align with repeated responses in rollout
                batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                batch = batch.union(gen_batch_output)
                all_uid_lst.extend(batch.non_tensor_batch['uid'])

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
                    if self.config.data.reference_key == "solution":
                        logp_z_lst, logp_x_lst, logp_other_y_lst, logp_diff_lst, \
                        decoded_token_lst, decoded_other_token_lst = self._compute_logp_vi_scaled(batch)
                        all_logp_z_lst.extend(logp_z_lst)
                        all_logp_x_lst.extend(logp_x_lst)
                        all_logp_other_y_lst.extend(logp_other_y_lst)
                        all_logp_diff_lst.extend(logp_diff_lst)
                        all_decoded_token_lst.extend(decoded_token_lst)
                        all_decoded_other_token_lst.extend(decoded_other_token_lst)
                    else:
                        logp_z_lst = self._compute_logp_vi_vanilla(batch)
                        all_logp_z_lst.extend(logp_z_lst)
                    reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
                    all_reward_tensor_lst.extend(reward_tensor)

            progress_bar.update(1)
            self.global_steps += 1

        # save all_logp_z_lst, all_logp_x_lst, all_logp_diff_lst, all_reward_tensor_lst to 
        def to_numpy(item):
            if torch.is_tensor(item):
                return item.detach().cpu().numpy()
            return np.asarray(item)
       
        all_rewards_lst = to_numpy(all_reward_tensor_lst).sum(-1)
        data = {
            "logp_z_lst": all_logp_z_lst, 
            "logp_x_lst": all_logp_x_lst, 
            "logp_diff_lst": all_logp_diff_lst, 
            "logp_other_y_lst": all_logp_other_y_lst,
            "decoded_token_lst": all_decoded_token_lst,
            "decoded_other_token_lst": all_decoded_other_token_lst,
            "rewards_lst": all_rewards_lst,
            "uid_lst": all_uid_lst
            }
        os.makedirs(self.config.trainer.default_local_dir, exist_ok=True)
        output_path = os.path.join(self.config.trainer.default_local_dir, "logps.pkl")
        print(f"Saved logps to {output_path}")
        f = open(output_path, "wb")
        pickle.dump(data, f)
        f.close()