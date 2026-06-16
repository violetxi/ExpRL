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
"""
Synchronous proof trainer aligned with the QED process-reward workflow.
"""

import asyncio
import json
import time
import uuid
from collections import defaultdict
from pprint import pprint
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.proof_utils import (
    DEFAULT_STEP_DELIMITER,
    assign_chunk_values_to_output_tokens,
    coerce_variants,
    compute_chunk_advantages,
    compute_chunk_rewards,
    format_variants_block,
    get_grader_client,
    load_evaluator_prompt,
    maybe_clip_chunk_advantages_for_length,
    normalize_prefix_scores,
    parse_schema,
    resolve_provider,
    score_process_judge_prompts,
    split_reward_chunks,
    strip_trailing_chat_tokens,
    validate_unit_interval,
    verify_proof,
)
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, compute_response_mask
from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.tracking import Tracking


class RayProofTrainer(RayPPOTrainer):
    @staticmethod
    def _as_bool(value: Any) -> bool:
        if isinstance(value, str):
            return value.lower() in {"1", "true", "yes", "y"}
        return bool(value)

    def _validate_config(self):
        super()._validate_config()

        if self.config.actor_rollout_ref.rollout.mode == "async":
            raise NotImplementedError("RayProofTrainer only supports synchronous rollout.mode")
        if self.use_critic:
            raise ValueError("RayProofTrainer expects critic.enable=false; it uses precomputed token advantages")
        if self.config.algorithm.use_kl_in_reward:
            raise NotImplementedError("RayProofTrainer does not support KL-in-reward")

        process_grader_cfg = OmegaConf.select(self.config, "process_grader")
        if process_grader_cfg is None:
            raise ValueError("process_grader config must be provided for proof process reward training")
        if process_grader_cfg.get("name", None) is None:
            raise ValueError("process_grader.name must be configured")
        if process_grader_cfg.get("prompt_name", None) is None:
            raise ValueError("process_grader.prompt_name must be configured")
        if self._as_bool(process_grader_cfg.get("use_rubric", False)) and self._as_bool(
            process_grader_cfg.get("use_reference_only", False)
        ):
            raise ValueError("process_grader.use_rubric and process_grader.use_reference_only are mutually exclusive")

        needs_validation = (
            self.config.trainer.get("val_before_train", True)
            or (self.config.trainer.get("test_freq", -1) > 0)
            or self.config.trainer.get("val_only", False)
        )
        if needs_validation:
            llm_grader_cfg = OmegaConf.select(self.config, "llm_grader")
            if llm_grader_cfg is None:
                raise ValueError("llm_grader config must be provided for proof validation")
            if llm_grader_cfg.get("name", None) is None:
                raise ValueError("llm_grader.name must be configured")
            if llm_grader_cfg.get("prompt_name", None) is None:
                raise ValueError("llm_grader.prompt_name must be configured")

    def _has_validation(self) -> bool:
        return OmegaConf.select(self.config, "llm_grader") is not None and len(self.val_dataset) > 0

    @staticmethod
    def _run_async(coro):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        raise RuntimeError("Unexpected running event loop in RayProofTrainer")

    def _safe_compute_training_metrics(self, batch: DataProto) -> dict[str, Any]:
        valid_token_count = int(batch.batch["response_mask"].sum().item())
        if valid_token_count > 0:
            return compute_data_metrics(batch=batch, use_critic=False, process_reward=True)

        responses = batch.batch["responses"]
        response_mask = batch.batch["response_mask"].float()
        attention_mask = batch.batch["attention_mask"].float()
        response_length = response_mask.sum(-1)
        prompt_length = attention_mask[:, :-responses.shape[-1]].sum(-1)
        max_response_length = responses.shape[-1]
        max_prompt_length = attention_mask[:, :-responses.shape[-1]].shape[-1]

        metrics = {
            "critic/score/mean": 0.0,
            "critic/score/max": 0.0,
            "critic/score/min": 0.0,
            "critic/rewards/mean": 0.0,
            "critic/rewards/max": 0.0,
            "critic/rewards/min": 0.0,
            "critic/advantages/mean": 0.0,
            "critic/advantages/max": 0.0,
            "critic/advantages/min": 0.0,
            "critic/returns/mean": 0.0,
            "critic/returns/max": 0.0,
            "critic/returns/min": 0.0,
            "response_length/mean": torch.mean(response_length).detach().item(),
            "response_length/max": torch.max(response_length).detach().item(),
            "response_length/min": torch.min(response_length).detach().item(),
            "response_length/clip_ratio": torch.mean(torch.eq(response_length, max_response_length).float()).item(),
            "prompt_length/mean": torch.mean(prompt_length).detach().item(),
            "prompt_length/max": torch.max(prompt_length).detach().item(),
            "prompt_length/min": torch.min(prompt_length).detach().item(),
            "prompt_length/clip_ratio": torch.mean(torch.eq(prompt_length, max_prompt_length).float()).item(),
        }
        if "__num_turns__" in batch.non_tensor_batch:
            num_turns = batch.non_tensor_batch["__num_turns__"]
            metrics["num_turns/min"] = num_turns.min()
            metrics["num_turns/max"] = num_turns.max()
            metrics["num_turns/mean"] = num_turns.mean()
        return metrics

    def _get_extra_info(self, batch_item) -> dict[str, Any]:
        extra_info = batch_item.non_tensor_batch.get("extra_info", {})
        if isinstance(extra_info, dict):
            return extra_info
        return {}

    def compute_process_proof_reward(self, batch: DataProto):
        process_grader_cfg = OmegaConf.select(self.config, "process_grader")
        delimiter = process_grader_cfg.get("delimiter", DEFAULT_STEP_DELIMITER)
        delimiter_token_text = str(delimiter).rstrip()
        delimiter_token_ids = self.tokenizer.encode(delimiter_token_text, add_special_tokens=False)
        if len(delimiter_token_ids) != 1:
            raise ValueError(
                f"Process reward delimiter {delimiter_token_text!r} must tokenize to exactly one token, got {delimiter_token_ids}"
            )
        delimiter_token_id = int(delimiter_token_ids[0])
        prompt_template = load_evaluator_prompt(process_grader_cfg.get("prompt_name"))
        use_rubric_prompt = self._as_bool(process_grader_cfg.get("use_rubric", False))
        use_reference_only_prompt = self._as_bool(process_grader_cfg.get("use_reference_only", False))

        device = batch.batch["response_mask"].device
        reward_tensor = torch.zeros_like(batch.batch["response_mask"], dtype=torch.float32, device=device)
        advantage_tensor = torch.zeros_like(batch.batch["response_mask"], dtype=torch.float32, device=device)
        new_response_masks = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        prompt_texts: list[str] = []
        sample_states: list[dict[str, Any]] = []

        for sample_idx, batch_item in enumerate(batch):
            batch_response_mask = batch_item.batch["response_mask"]
            response_len = int(batch_response_mask.sum().item())
            output_token_ids = batch_item.batch["responses"][:response_len].detach().cpu().tolist()
            extra_info = self._get_extra_info(batch_item)
            reward_model_info = batch_item.non_tensor_batch.get("reward_model", {})
            question = extra_info.get("question", "")
            ref_solution = extra_info.get("solution", "")
            variants = coerce_variants(extra_info.get("variants", extra_info.get("answer")))
            rubric_payload = extra_info.get("rubric", reward_model_info.get("rubric", ""))
            marking_scheme = parse_schema(rubric_payload) if rubric_payload else ""

            if not output_token_ids:
                sample_states.append(
                    {
                        "response_mask": batch_response_mask,
                        "response_len": 0,
                        "output_token_ids": [],
                        "chunk_token_spans": [],
                        "result_indices": [],
                    }
                )
                continue

            chunks = split_reward_chunks(
                torch.tensor(output_token_ids, dtype=torch.long),
                delimiter_token_id=delimiter_token_id,
            )
            chunk_token_spans = [chunk.token_span for chunk in chunks]
            variants_block = format_variants_block(variants)
            sample_prompt_indices: list[int] = []
            for prefix_idx, chunk in enumerate(chunks):
                prefix_text = strip_trailing_chat_tokens(
                    self.tokenizer.decode(
                        output_token_ids[: chunk.token_span[1]],
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    )
                )
                if use_rubric_prompt:
                    prompt_texts.append(
                        prompt_template.format(
                            problem=question,
                            reasoning_so_far=prefix_text,
                            marking_scheme=marking_scheme,
                        )
                    )
                elif use_reference_only_prompt:
                    prompt_texts.append(
                        prompt_template.format(
                            problem=question,
                            reasoning_so_far=prefix_text,
                            reference_solution=ref_solution,
                        )
                    )
                else:
                    prompt_texts.append(
                        prompt_template.format(
                            problem=question,
                            reasoning_so_far=prefix_text,
                            reference_solution=ref_solution,
                            variants_block=variants_block,
                        )
                    )
                sample_prompt_indices.append(len(prompt_texts) - 1)

            sample_states.append(
                {
                    "response_mask": batch_response_mask,
                    "response_len": response_len,
                    "output_token_ids": output_token_ids,
                    "chunk_token_spans": chunk_token_spans,
                    "result_indices": sample_prompt_indices,
                }
            )

        client = get_grader_client(resolve_provider(process_grader_cfg.get("name"), process_grader_cfg.get("provider")))
        prompt_results = self._run_async(
            score_process_judge_prompts(
                prompt_texts=prompt_texts,
                model=process_grader_cfg.get("name"),
                sampling_kwargs=process_grader_cfg.get("sampling_kwargs", None),
                client=client,
                timeout_seconds=process_grader_cfg.get("timeout_seconds", 900),
                max_retries=process_grader_cfg.get("max_retries", 3),
                retry_backoff=list(process_grader_cfg.get("retry_backoff", [15, 30, 60, 90, 120])),
                provider=process_grader_cfg.get("provider", None),
                max_concurrency=process_grader_cfg.get("max_concurrency", None),
            )
        )
        prompt_result_map = {idx: result for idx, result in enumerate(prompt_results)}

        metric_accumulators = defaultdict(list)
        for sample_idx, state in enumerate(sample_states):            
            batch_response_mask = state["response_mask"]
            response_len = state["response_len"]
            output_token_ids = state["output_token_ids"]
            chunk_token_spans = state["chunk_token_spans"]
            sample_results = [prompt_result_map[idx] for idx in state["result_indices"]]

            raw_prefix_scores = [float(result.score) for result in sample_results]
            prefix_scores = normalize_prefix_scores(raw_prefix_scores)
            if prefix_scores:
                validate_unit_interval(prefix_scores, "process_reward.prefix_scores")

            invalid_prefix_indices = [
                local_idx
                for local_idx, result in enumerate(sample_results)
                if result.score <= 0 or getattr(result, "parse_failed", False)
            ]

            chunk_rewards: list[float] = []
            raw_chunk_advantages: list[float] = []
            chunk_advantages: list[float] = []
            is_overflow = False
            is_length_clipped = False
            final_prefix_score = 0.0
            raw_final_prefix_score = 0.0
            failure_cause = ""

            if invalid_prefix_indices:
                new_response_masks.append(torch.zeros_like(batch_response_mask))
                failing_results = [sample_results[idx] for idx in invalid_prefix_indices]
                failure_cause = next(
                    (result.failure_cause for result in failing_results if getattr(result, "failure_cause", None)),
                    "invalid_process_reward",
                )
            else:
                chunk_rewards = compute_chunk_rewards(prefix_scores)
                raw_chunk_advantages = compute_chunk_advantages(prefix_scores)
                chunk_advantages, is_overflow, is_length_clipped = maybe_clip_chunk_advantages_for_length(
                    output_token_ids=output_token_ids,
                    eos_token_id=getattr(self.tokenizer, "eos_token_id", None),
                    chunk_advantages=raw_chunk_advantages,
                    is_clip_length=bool(process_grader_cfg.get("is_clip_length", False)),
                )
                token_rewards = assign_chunk_values_to_output_tokens(
                    num_output_tokens=response_len,
                    chunk_token_spans=chunk_token_spans,
                    chunk_values=chunk_rewards
                )
                token_advantages = assign_chunk_values_to_output_tokens(
                    num_output_tokens=response_len,
                    chunk_token_spans=chunk_token_spans,
                    chunk_values=chunk_advantages
                )
                reward_tensor[sample_idx, :response_len] = torch.tensor(token_rewards, dtype=torch.float32, device=device)
                advantage_tensor[sample_idx, :response_len] = torch.tensor(
                    token_advantages, dtype=torch.float32, device=device
                )
                new_response_masks.append(batch_response_mask)
                final_prefix_score = float(prefix_scores[-1]) if prefix_scores else 0.0
                raw_final_prefix_score = float(raw_prefix_scores[-1]) if raw_prefix_scores else 0.0

            reward_extra_infos_dict["final_prefix_score"].append(final_prefix_score)
            reward_extra_infos_dict["raw_final_prefix_score"].append(raw_final_prefix_score)
            reward_extra_infos_dict["is_valid_process_reward"].append(int(not invalid_prefix_indices))
            reward_extra_infos_dict["is_overflow"].append(int(is_overflow))
            reward_extra_infos_dict["is_length_clipped"].append(int(is_length_clipped))
            reward_extra_infos_dict["num_prefixes"].append(len(chunk_token_spans))
            reward_extra_infos_dict["failure_cause"].append(failure_cause)
            reward_extra_infos_dict["invalid_prefix_indices"].append(json.dumps(invalid_prefix_indices))

            num_chunks = len(chunk_token_spans)
            metric_accumulators["final_prefix_score"].append(final_prefix_score)
            metric_accumulators["raw_final_prefix_score"].append(raw_final_prefix_score)
            metric_accumulators["invalid_trajectory_ratio"].append(float(bool(invalid_prefix_indices)))
            metric_accumulators["overflow_ratio"].append(float(is_overflow))
            metric_accumulators["length_clipped_ratio"].append(float(is_length_clipped))
            metric_accumulators["num_prefixes"].append(float(num_chunks))
            if 1 <= num_chunks <= 10:
                metric_accumulators[f"chunk_{num_chunks}_indicator"].append(1.0)
                metric_accumulators[f"chunk_{num_chunks}_final_prefix_score"].append(final_prefix_score)
            elif num_chunks > 10:
                metric_accumulators["chunk_gt_10_indicator"].append(1.0)
                metric_accumulators["chunk_gt_10_final_prefix_score"].append(final_prefix_score)

        batch_size = len(sample_states)
        batch.batch["response_mask"] = torch.stack(new_response_masks, dim=0)
        reward_metrics = {
            "process_reward/final_prefix_score_mean": float(np.mean(metric_accumulators["final_prefix_score"] or [0.0])),
            "process_reward/raw_final_prefix_score_mean": float(
                np.mean(metric_accumulators["raw_final_prefix_score"] or [0.0])
            ),
            "process_reward/invalid_trajectory_ratio": float(
                np.mean(metric_accumulators["invalid_trajectory_ratio"] or [0.0])
            ),
            "process_reward/overflow_ratio": float(np.mean(metric_accumulators["overflow_ratio"] or [0.0])),
            "process_reward/length_clipped_ratio": float(
                np.mean(metric_accumulators["length_clipped_ratio"] or [0.0])
            ),
            "process_reward/mean_num_prefixes": float(np.mean(metric_accumulators["num_prefixes"] or [0.0])),
            "exp_rl/avg_num_chunks": float(np.mean(metric_accumulators["num_prefixes"] or [0.0])),
            "exp_rl/final_step_reward/overall": float(
                np.mean(metric_accumulators["final_prefix_score"] or [0.0])
            ),
        }
        for chunk_count in range(1, 11):
            indicator_key = f"chunk_{chunk_count}_indicator"
            reward_key = f"chunk_{chunk_count}_final_prefix_score"
            reward_metrics[f"exp_rl/chunk_freq/chunk_{chunk_count}"] = (
                len(metric_accumulators[indicator_key]) / batch_size if batch_size else 0.0
            )
            reward_metrics[f"exp_rl/final_step_reward/chunk_{chunk_count}"] = float(
                np.mean(metric_accumulators[reward_key] or [0.0])
            )
        reward_metrics["exp_rl/chunk_freq/chunk_gt_10"] = (
            len(metric_accumulators["chunk_gt_10_indicator"]) / batch_size if batch_size else 0.0
        )
        reward_metrics["exp_rl/final_step_reward/chunk_gt_10"] = float(
            np.mean(metric_accumulators["chunk_gt_10_final_prefix_score"] or [0.0])
        )
        return reward_tensor, advantage_tensor, reward_extra_infos_dict, batch, reward_metrics

    def _maybe_log_val_generations(self, inputs, outputs, scores, judge_outputs=None, judge_reasonings=None):
        generations_to_log = self.config.trainer.log_val_generations
        if generations_to_log == 0:
            return

        import numpy as np

        if judge_outputs is None or judge_reasonings is None:
            return super()._maybe_log_val_generations(inputs, outputs, scores)

        samples = list(zip(inputs, outputs, scores, judge_outputs, judge_reasonings, strict=True))
        samples.sort(key=lambda x: x[0])

        rng = np.random.RandomState(42)
        rng.shuffle(samples)
        samples = samples[:generations_to_log]

        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _validate(self):
        llm_grader_cfg = OmegaConf.select(self.config, "llm_grader")
        rubric_max_score = int(llm_grader_cfg.get("max_score", 7))

        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)
        sample_inputs = []
        sample_outputs = []
        sample_scores = []
        sample_turns = []
        sample_judge_outputs = []
        sample_judge_reasonings = []
        verifier_time_s = 0.0

        grader_client = get_grader_client(resolve_provider(llm_grader_cfg.get("name"), llm_grader_cfg.get("provider")))

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n,
                interleave=True,
            )

            input_ids = test_batch.batch["input_ids"]
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

            size_divisor = self.actor_rollout_wg.world_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            if "response_mask" not in test_batch.batch:
                test_batch.batch["response_mask"] = compute_response_mask(test_batch)
            test_batch.meta_info["validate"] = True

            async def _score_validation_batch():
                semaphore = None
                if llm_grader_cfg.get("max_concurrency", None):
                    semaphore = asyncio.Semaphore(int(llm_grader_cfg.get("max_concurrency")))

                async def _score_one(batch_item):
                    extra_info = self._get_extra_info(batch_item)
                    reward_model_info = batch_item.non_tensor_batch.get("reward_model", {})
                    rubric_payload = extra_info.get("rubric", reward_model_info.get("rubric", ""))
                    schema = parse_schema(rubric_payload)
                    ref_solution = extra_info.get("solution", "")
                    if isinstance(ref_solution, str) and ref_solution.strip().upper() == "N/A":
                        ref_solution = ""

                    response_mask = batch_item.batch["response_mask"]
                    response_len = int(response_mask.sum().item())
                    output_token_ids = batch_item.batch["responses"][:response_len].detach().cpu().tolist()
                    generation = strip_trailing_chat_tokens(
                        self.tokenizer.decode(
                            output_token_ids,
                            skip_special_tokens=False,
                            clean_up_tokenization_spaces=False,
                        )
                    )

                    async def _call():
                        return await verify_proof(
                            problem=extra_info.get("question", ""),
                            ref_solution=ref_solution,
                            schema=schema,
                            generation=generation,
                            prompt_name=llm_grader_cfg.get("prompt_name"),
                            model=llm_grader_cfg.get("name"),
                            sampling_kwargs=llm_grader_cfg.get("sampling_kwargs", None),
                            client=grader_client,
                            timeout_seconds=llm_grader_cfg.get("timeout_seconds", 900),
                            max_retries=llm_grader_cfg.get("max_retries", 3),
                            retry_backoff=list(llm_grader_cfg.get("retry_backoff", [15, 30, 60, 90, 120])),
                            provider=llm_grader_cfg.get("provider", None),
                        )

                    if semaphore is None:
                        return await _call()
                    async with semaphore:
                        return await _call()

                return await asyncio.gather(*(_score_one(batch_item) for batch_item in test_batch))

            verifier_start = time.perf_counter()
            validation_results = self._run_async(_score_validation_batch())
            verifier_time_s += time.perf_counter() - verifier_start
            raw_scores = [float(result.score) for result in validation_results]
            normalized_scores = [score / rubric_max_score for score in raw_scores]
            sample_scores.extend(normalized_scores)
            judge_outputs = [result.output_text for result in validation_results]
            judge_reasonings = [result.reasoning_text for result in validation_results]
            sample_judge_outputs.extend(judge_outputs)
            sample_judge_reasonings.extend(judge_reasonings)

            reward_extra_infos_dict["reward"].extend(normalized_scores)
            reward_extra_infos_dict["raw_score"].extend(raw_scores)
            reward_extra_infos_dict["judge_output"].extend(judge_outputs)
            reward_extra_infos_dict["judge_reasoning"].extend(judge_reasonings)
            reward_extra_infos_dict["full_credit"].extend([float(score == rubric_max_score) for score in raw_scores])
            reward_extra_infos_dict["failure_cause"].extend(
                [result.failure_cause or "" for result in validation_results]
            )

            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * len(raw_scores)))

        self._maybe_log_val_generations(
            inputs=sample_inputs,
            outputs=sample_outputs,
            scores=sample_scores,
            judge_outputs=sample_judge_outputs,
            judge_reasonings=sample_judge_reasonings,
        )

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
                n_max = max(int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys())
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(prefix) for prefix in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        metric_sec = "val-aux"
                    metric_dict[f"{metric_sec}/{data_source}/{var_name}/{metric_name}"] = metric_val

        if sample_scores:
            metric_dict["val-core/proof/reward_mean"] = float(np.mean(sample_scores))
        if reward_extra_infos_dict["raw_score"]:
            metric_dict["val-aux/proof/raw_score_mean"] = float(np.mean(reward_extra_infos_dict["raw_score"]))
        if reward_extra_infos_dict["full_credit"]:
            metric_dict["val-aux/proof/full_credit_rate"] = float(np.mean(reward_extra_infos_dict["full_credit"]))
        metric_dict["val-aux/proof/verifier_time_s"] = verifier_time_s
        if sample_scores:
            metric_dict["val-aux/proof/verifier_time_per_sample_s"] = verifier_time_s / len(sample_scores)

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        return metric_dict

    def fit(self):
        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self._load_checkpoint()

        if self._has_validation() and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return

        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")
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
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch = gen_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    with marked_timer("gen", timing_raw, color="red"):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(batch.batch))],
                        dtype=object,
                    )
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)

                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    with marked_timer("reward", timing_raw, color="yellow"):
                        (
                            reward_tensor,
                            advantage_tensor,
                            reward_extra_infos_dict,
                            batch,
                            reward_metrics,
                        ) = self.compute_process_proof_reward(batch)
                        metrics.update(reward_metrics)

                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                        entropy_agg = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                        metrics["actor/entropy"] = entropy_agg.detach().item()
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)

                        if "rollout_log_probs" in batch.batch.keys():
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
                            if rollout_probs_diff.numel() > 0:
                                metrics.update(
                                    {
                                        "training/rollout_probs_diff_max": rollout_probs_diff.max().detach().item(),
                                        "training/rollout_probs_diff_mean": rollout_probs_diff.mean().detach().item(),
                                        "training/rollout_probs_diff_std": rollout_probs_diff.std().detach().item(),
                                    }
                                )

                    if self.use_reference_policy:
                        with marked_timer("ref", timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    with marked_timer("adv", timing_raw, color="brown"):
                        batch.batch["token_level_scores"] = reward_tensor
                        batch.batch["token_level_rewards"] = reward_tensor
                        batch.batch["advantages"] = advantage_tensor
                        batch.batch["returns"] = advantage_tensor.clone()

                    if self.config.trainer.critic_warmup <= self.global_steps:
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        metrics.update(actor_output_metrics)

                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
                            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                            scores = reward_extra_infos_dict.get(
                                "final_prefix_score",
                                batch.batch["token_level_scores"].sum(-1).cpu().tolist(),
                            )
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

                    if (
                        self._has_validation()
                        and self.config.trainer.test_freq > 0
                        and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                    ):
                        with marked_timer("testing", timing_raw, color="green"):
                            val_metrics = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    esi_close_to_expiration = should_save_ckpt_esi(
                        max_steps_duration=self.max_steps_duration,
                        redundant_time=self.config.trainer.esi_redundant_time,
                    )
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

                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                metrics.update(self._safe_compute_training_metrics(batch=batch))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                logger.log(data=metrics, step=self.global_steps)
                progress_bar.update(1)
                self.global_steps += 1

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return

                if hasattr(self.train_dataset, "on_batch_end"):
                    self.train_dataset.on_batch_end(batch=batch)
