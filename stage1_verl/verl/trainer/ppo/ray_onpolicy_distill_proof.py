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
On-policy distillation trainer with proof-style validation.
"""

import asyncio
import time
from collections import defaultdict

import numpy as np
from omegaconf import OmegaConf

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.trainer.ppo.metric_utils import process_validation_metrics
from verl.trainer.ppo.proof_utils import (
    get_grader_client,
    parse_schema,
    resolve_provider,
    strip_trailing_chat_tokens,
    verify_proof,
)
from verl.trainer.ppo.ray_onpolicy_distill import RayOnpolicyDistill, compute_response_mask


class RayOnpolicyDistillProof(RayOnpolicyDistill):
    def _should_compute_training_outcome_score(self) -> bool:
        return False

    def _validate_config(self):
        super()._validate_config()

        needs_validation = (
            self.config.trainer.get("val_before_train", True)
            or (self.config.trainer.get("test_freq", -1) > 0)
            or self.config.trainer.get("val_only", False)
        )
        if not needs_validation:
            return

        llm_grader_cfg = OmegaConf.select(self.config, "llm_grader")
        if llm_grader_cfg is None:
            raise ValueError("llm_grader config must be provided for proof validation")
        if llm_grader_cfg.get("name", None) is None:
            raise ValueError("llm_grader.name must be configured")
        if llm_grader_cfg.get("prompt_name", None) is None:
            raise ValueError("llm_grader.prompt_name must be configured")

    @staticmethod
    def _run_async(coro):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        raise RuntimeError("Unexpected running event loop in RayOnpolicyDistillProof")

    def _get_extra_info(self, batch_item):
        extra_info = batch_item.non_tensor_batch.get("extra_info", {})
        if isinstance(extra_info, dict):
            return extra_info
        return {}

    def _maybe_log_val_generations(self, inputs, outputs, scores, judge_outputs=None, judge_reasonings=None):
        generations_to_log = self.config.trainer.log_val_generations
        if generations_to_log == 0:
            return

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
        if llm_grader_cfg is None:
            return super()._validate()

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
                    if not isinstance(reward_model_info, dict):
                        reward_model_info = {}
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
            judge_outputs = [result.output_text for result in validation_results]
            judge_reasonings = [result.reasoning_text for result in validation_results]

            sample_scores.extend(normalized_scores)
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
