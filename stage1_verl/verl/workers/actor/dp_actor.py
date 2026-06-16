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
Single Process Actor
"""

import logging
import os
import math
from collections import defaultdict
import numpy as np

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.device import get_device_name, is_cuda_available, is_npu_available
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

if is_cuda_available:
    from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import index_first_axis, pad_input, rearrange, unpad_input


__all__ = ["DataParallelPPOActor", "DataParallelDPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  #  use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()

    def _forward_micro_batch(
        self, micro_batch, temperature, calculate_entropy=False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            if "image_bound" in micro_batch["multi_modal_inputs"][0]:  # minicpm-o logic
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = [inputs[key] for inputs in micro_batch["multi_modal_inputs"]]
            else:
                for key in micro_batch["multi_modal_inputs"][0].keys():
                    multi_modal_inputs[key] = torch.cat(
                        [inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0
                    )

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = "multi_modal_inputs" in micro_batch.keys()
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
            self.actor_optimizer.zero_grad()
        else:
            self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)

        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]

        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        metrics = {}
        print(f"Number of updates: {len(mini_batches)}")
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    entropy, log_prob = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")                    
                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla
                    # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                    # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                    policy_loss_fn = get_policy_loss_fn(loss_mode)
                    loss_output = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config
                    )
                    if isinstance(loss_output, dict):
                        loss_dict = loss_output
                    else:
                        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower = loss_output
                        loss_dict = {
                            "pg_loss": pg_loss,
                            "pg_clipfrac": pg_clipfrac,
                            "ppo_kl": ppo_kl,
                            "pg_clipfrac_lower": pg_clipfrac_lower,
                        }

                    pg_loss = loss_dict.get("pg_loss", torch.tensor(0.0))
                    pg_clipfrac = loss_dict.get("pg_clipfrac", torch.tensor(0.0))
                    ppo_kl = loss_dict.get("ppo_kl", torch.tensor(0.0))
                    pg_clipfrac_lower = loss_dict["pg_clipfrac_lower"]

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff

                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item()
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * (response_mask.shape[0] / self.config.ppo_mini_batch_size)
                    else:
                        loss = policy_loss / self.gradient_accumulation
                    loss.backward()

                    micro_batch_metrics.update(
                        {
                            "actor/pg_loss": pg_loss.detach().item(),
                            "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                            "actor/ppo_kl": ppo_kl.detach().item(),
                            "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                        }
                    )
                    append_to_dict(metrics, micro_batch_metrics)
                                
                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics

class DataParallelDistillActor(DataParallelPPOActor):
    
    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
            "token_level_scores"
        ]
        # @TODO: should prob have a separate Actor class for VI updates
        assert self.config.policy_loss.get("loss_mode", "vanilla") == "distill", "Distill loss mode must be distill"
        
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)
        assert len(mini_batches) == 1, "Distill actor only supports one mini-batch"

        metrics = {}
        print(f"Number of updates: {len(mini_batches)}")
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    entropy, log_prob = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )

                    loss_mode = self.config.policy_loss.get("loss_mode", "distill")
                    policy_loss_fn = get_policy_loss_fn(loss_mode)                    
                    loss_dict = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,                        
                        loss_agg_mode=loss_agg_mode,
                        config=self.config
                    )
                    
                    distill_loss = loss_dict.get("distill_loss", torch.tensor(0.0))
                    final_loss = distill_loss
                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = final_loss * (response_mask.shape[0] / self.config.ppo_mini_batch_size)
                    else:
                        loss = final_loss / self.gradient_accumulation
                    loss.backward()

                    micro_batch_metrics.update(
                        {
                            "actor/loss": distill_loss.detach().item(),
                        }
                    )
                    append_to_dict(metrics, micro_batch_metrics)
                                
                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics


class DataParallelVIActor(DataParallelPPOActor):
    
    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
            "pg_loss_mask",
            "sft_loss_mask",
            "token_level_scores"
        ]
        # @TODO: should prob have a separate Actor class for VI updates
        assert self.config.policy_loss.get("loss_mode", "vanilla") == "vi_vanilla", "VI loss mode must be vi_vanilla"
        
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        metrics = {}
        print(f"Number of updates: {len(mini_batches)}")
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    # all return: (bsz, response_length)
                    calculate_entropy = False
                    if entropy_coeff != 0:
                        calculate_entropy = True
                    entropy, log_prob = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )

                    loss_mode = self.config.policy_loss.get("loss_mode", "vi_vanilla")
                    policy_loss_fn = get_policy_loss_fn(loss_mode)                    
                    loss_dict = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        pg_loss_mask=model_inputs["pg_loss_mask"],
                        sft_loss_mask=model_inputs["sft_loss_mask"],
                        rewards=model_inputs["token_level_scores"],
                        loss_agg_mode=loss_agg_mode,
                        config=self.config
                    )
                    
                    pg_loss = loss_dict.get("pg_loss", torch.tensor(0.0))
                    sft_loss = loss_dict.get("sft_loss", torch.tensor(0.0))
                    pg_clipfrac = loss_dict.get("pg_clipfrac", torch.tensor(0.0))
                    ppo_kl = loss_dict.get("ppo_kl", torch.tensor(0.0))
                    pg_clipfrac_lower = loss_dict["pg_clipfrac_lower"]

                    if entropy_coeff != 0:
                        entropy_loss = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        # compute policy loss
                        policy_loss = pg_loss - entropy_loss * entropy_coeff

                    else:
                        policy_loss = pg_loss

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item()
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    final_loss = policy_loss + sft_loss
                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = final_loss * (response_mask.shape[0] / self.config.ppo_mini_batch_size)
                    else:
                        loss = final_loss / self.gradient_accumulation
                    loss.backward()

                    micro_batch_metrics.update(
                        {
                            "actor/sft_loss": sft_loss.detach().item(),
                            "actor/pg_loss": pg_loss.detach().item(),
                            "actor/pg_clipfrac": pg_clipfrac.detach().item(),
                            "actor/ppo_kl": ppo_kl.detach().item(),
                            "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
                        }
                    )
                    append_to_dict(metrics, micro_batch_metrics)
                                
                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics


class DataParallelDPOActor(DataParallelPPOActor):

    def _create_dpo_batch(self, data: DataProto):
        input_ids = data.batch["input_ids"]
        attention_mask = data.batch["attention_mask"]
        position_ids = data.batch["position_ids"]
        # chosen
        chosen_labels = torch.cat([input_ids, data.batch["chosen_labels"]], dim=1)
        chosen_attention_mask = torch.cat([attention_mask, data.batch["chosen_attention_mask"]], dim=1)
        chosen_position_ids = torch.cat([position_ids, data.batch["chosen_position_ids"]], dim=1)
        assert chosen_labels.shape == chosen_attention_mask.shape == chosen_position_ids.shape
        chosen_data_dict = {
            "input_ids": chosen_labels,
            "attention_mask": chosen_attention_mask,
            "position_ids": chosen_position_ids,
            "responses": data.batch["chosen_labels"],
            "response_mask": data.batch["chosen_attention_mask"],
            "response_position_ids": data.batch["chosen_position_ids"]                        
        }
        if "ref_chosen_log_probs" in data.batch:
            chosen_data_dict["ref_log_prob"] = data.batch["ref_chosen_log_probs"]
        if "chosen_reasoning_penalty" in data.batch:
            chosen_data_dict["reasoning_penalty"] = data.batch["chosen_reasoning_penalty"]

        chosen_data = DataProto.from_dict(chosen_data_dict)
        chosen_data.meta_info = data.meta_info
        chosen_data.non_tensor_batch = data.non_tensor_batch

        # rejected
        rejected_labels = torch.cat([input_ids, data.batch["rejected_labels"]], dim=1)
        rejected_attention_mask = torch.cat([attention_mask, data.batch["rejected_attention_mask"]], dim=1)
        rejected_position_ids = torch.cat([position_ids, data.batch["rejected_position_ids"]], dim=1)
        assert rejected_labels.shape == rejected_attention_mask.shape == rejected_position_ids.shape        
        rejected_data_dict = {
            "input_ids": rejected_labels,
            "attention_mask": rejected_attention_mask,
            "position_ids": rejected_position_ids,
            "responses": data.batch["rejected_labels"],
            "response_mask": data.batch["rejected_attention_mask"],
            "response_position_ids": data.batch["rejected_position_ids"]
        }
        if "ref_rejected_log_probs" in data.batch:
            rejected_data_dict["ref_log_prob"] = data.batch["ref_rejected_log_probs"]
        if "rejected_reasoning_penalty" in data.batch:
            rejected_data_dict["reasoning_penalty"] = data.batch["rejected_reasoning_penalty"]

        rejected_data = DataProto.from_dict(rejected_data_dict)
        rejected_data.meta_info = data.meta_info
        rejected_data.non_tensor_batch = data.non_tensor_batch

        return chosen_data, rejected_data

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        # can't use this bc chosen and rejected have different lengths, and may result in different batch sizes
        assert not use_dynamic_bsz, "Dynamic bsz is not supported for DPO"
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        
        chosen_data, rejected_data = self._create_dpo_batch(data)

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        chosen_data = chosen_data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)
        rejected_data = rejected_data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)
        
        micro_batches_chosen = chosen_data.split(micro_batch_size)
        micro_batches_rejected = rejected_data.split(micro_batch_size)

        chosen_log_probs_lst = []
        chosen_entropy_lst = []
        rejected_log_probs_lst = []
        rejected_entropy_lst = []
        for micro_batch_chosen, micro_batch_rejected in zip(micro_batches_chosen, micro_batches_rejected):
            model_inputs_chosen = {**micro_batch_chosen.batch, **micro_batch_chosen.non_tensor_batch}
            model_inputs_rejected = {**micro_batch_rejected.batch, **micro_batch_rejected.non_tensor_batch}
            with torch.no_grad():
                entropy_chosen, log_probs_chosen = self._forward_micro_batch(
                    model_inputs_chosen, temperature=temperature, calculate_entropy=calculate_entropy
                )
                entropy_rejected, log_probs_rejected = self._forward_micro_batch(
                    model_inputs_rejected, temperature=temperature, calculate_entropy=calculate_entropy
                )
            chosen_log_probs_lst.append(log_probs_chosen)
            rejected_log_probs_lst.append(log_probs_rejected)
            if calculate_entropy:
                chosen_entropy_lst.append(entropy_chosen)
                rejected_entropy_lst.append(entropy_rejected)

        chosen_log_probs = torch.concat(chosen_log_probs_lst, dim=0)
        rejected_log_probs = torch.concat(rejected_log_probs_lst, dim=0)
        chosen_entropys = None
        rejected_entropys = None
        if calculate_entropy:
            chosen_entropys = torch.concat(chosen_entropy_lst, dim=0)
            rejected_entropys = torch.concat(rejected_entropy_lst, dim=0)

        return chosen_log_probs, rejected_log_probs, chosen_entropys, rejected_entropys

    def compute_dpo_policy_loss(self, 
                    chosen_log_prob: torch.Tensor, 
                    rejected_log_prob: torch.Tensor, 
                    ref_log_prob_chosen: torch.Tensor, 
                    ref_log_prob_rejected: torch.Tensor, 
                    chosen_mask: torch.Tensor, 
                    rejected_mask: torch.Tensor,                         
                    beta: float,
                    alpha: float = 0.0,
                    chosen_reasoning_penalty: torch.Tensor = None,
                    rejected_reasoning_penalty: torch.Tensor = None,
                    label_smoothing: float = 0.0,
                    loss_type: str = "sigmoid"):
        """
        Compute DPO policy loss.
        """
        import torch.nn.functional as F
        chosen_lp = (chosen_log_prob * chosen_mask).sum(dim=-1)
        rejected_lp = (rejected_log_prob * rejected_mask).sum(dim=-1)    
        ref_chosen_lp = (ref_log_prob_chosen.detach() * chosen_mask).sum(dim=-1)
        ref_rejected_lp = (ref_log_prob_rejected.detach() * rejected_mask).sum(dim=-1)

        lp_diff_chosen = beta * (chosen_lp - ref_chosen_lp)
        lp_diff_rejected = beta * (rejected_lp - ref_rejected_lp)

        margin = lp_diff_chosen - lp_diff_rejected
        accuracy = (margin > 0).float()
        
        think_penalty = chosen_reasoning_penalty - rejected_reasoning_penalty        

        if loss_type == "sigmoid":
            dpo_loss = -F.logsigmoid(margin + alpha * think_penalty) * (1 - label_smoothing) - F.logsigmoid(-margin - alpha * think_penalty) * label_smoothing
        elif loss_type == "ipo":
            dpo_loss = (margin + alpha * think_penalty) ** 2
        else:
            raise ValueError(f"Unsupported loss_type: {loss_type}. Choose 'sigmoid', 'ipo', or 'hinge'.")
        
        return dpo_loss.mean(), margin.mean(), lp_diff_chosen.mean(), lp_diff_rejected.mean(), accuracy.mean(), think_penalty.mean()

    def update_policy_dpo_with_ref(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        # --- DPO Parameters ---
        beta = self.config.get("dpo_beta", 0.1)
        alpha = self.config.get("dpo_alpha", 0.0)        
        loss_type = data.meta_info.get("dpo_loss_type", "sigmoid")
        label_smoothing = data.meta_info.get("dpo_label_smoothing", 0.0)        
        chosen_data, rejected_data = self._create_dpo_batch(data)

        print(f"DPO Parameters: alpha: {alpha}\n beta: {beta}\n loss_type: {loss_type}\n label_smoothing: {label_smoothing}")

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids", "response_mask", "response_position_ids", "ref_log_prob", "reasoning_penalty"]
        non_tensor_select_keys = data.non_tensor_batch.keys()        

        chosen_data = chosen_data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)
        rejected_data = rejected_data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches_chosen = chosen_data.split(self.config.ppo_mini_batch_size)
        mini_batches_rejected = rejected_data.split(self.config.ppo_mini_batch_size)

        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for mini_batch_chosen, mini_batch_rejected in zip(mini_batches_chosen, mini_batches_rejected):                
                self.gradient_accumulation = (
                    self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                )
                micro_batches_chosen = mini_batch_chosen.split(self.config.ppo_micro_batch_size_per_gpu)
                micro_batches_rejected = mini_batch_rejected.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch_chosen, micro_batch_rejected in zip(micro_batches_chosen, micro_batches_rejected):
                    micro_batch_metrics = {}
                    model_inputs_chosen = {**micro_batch_chosen.batch, **micro_batch_chosen.non_tensor_batch}
                    model_inputs_rejected = {**micro_batch_rejected.batch, **micro_batch_rejected.non_tensor_batch}
                    # loss mask                    
                    response_mask_chosen = model_inputs_chosen["response_mask"]
                    response_mask_rejected = model_inputs_rejected["response_mask"]
                    # ref log prob
                    ref_log_prob_chosen = model_inputs_chosen["ref_log_prob"]
                    ref_log_prob_rejected = model_inputs_rejected["ref_log_prob"]
                    # reasoning penalty
                    chosen_reasoning_penalty = model_inputs_chosen["reasoning_penalty"]
                    rejected_reasoning_penalty = model_inputs_rejected["reasoning_penalty"]

                    # all return: (bsz, response_length)
                    entropy_chosen, log_prob_chosen = self._forward_micro_batch(model_inputs_chosen, temperature=temperature, calculate_entropy=True)
                    entropy_rejected, log_prob_rejected = self._forward_micro_batch(model_inputs_rejected, temperature=temperature, calculate_entropy=True)                    

                    dpo_loss, margin, lp_diff_chosen, lp_diff_rejected, accuracy, reasoning_penalty = self.compute_dpo_policy_loss(
                        chosen_log_prob=log_prob_chosen,
                        rejected_log_prob=log_prob_rejected,
                        ref_log_prob_chosen=ref_log_prob_chosen,
                        ref_log_prob_rejected=ref_log_prob_rejected,
                        chosen_mask=response_mask_chosen,
                        rejected_mask=response_mask_rejected,                        
                        beta=beta,
                        alpha=alpha,
                        label_smoothing=label_smoothing,
                        loss_type=loss_type,
                        chosen_reasoning_penalty=chosen_reasoning_penalty,
                        rejected_reasoning_penalty=rejected_reasoning_penalty,
                    )
                    loss = dpo_loss / self.gradient_accumulation
                    loss.backward()

                    micro_batch_metrics.update(
                        {
                            "actor/dpo_loss": dpo_loss.detach().item(),
                            "actor/dpo_margin": margin.detach().item(),
                            "actor/dpo_reasoning_penalty": reasoning_penalty.detach().item(),
                            "actor/dpo_lp_diff_chosen": lp_diff_chosen.detach().item(),
                            "actor/dpo_lp_diff_rejected": lp_diff_rejected.detach().item(),
                            "actor/dpo_accuracy": accuracy.detach().item()
                        }
                    )
                    if entropy_chosen is not None:
                        micro_batch_metrics.update(
                            {
                                "actor/dpo_chosen_entropy": entropy_chosen.mean().detach().item(),
                                "actor/dpo_rejected_entropy": entropy_rejected.mean().detach().item(),
                            }
                        )
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics
