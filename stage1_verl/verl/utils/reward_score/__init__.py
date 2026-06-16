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
# from . import gsm8k, math, prime_math, prime_code

from verl.utils.import_utils import deprecated


def default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
):
    """Compute the score for a given solution based on the data source.

    Args:
        data_source (str): The source dataset identifier which determines the scoring method.
        solution_str (str): The solution string to be evaluated.
        ground_truth (str): The ground truth answer for comparison.
        extra_info (dict, optional): Additional information that might be needed for scoring. Defaults to None.

    Returns:
        float: The computed score as a floating point number. If the result is a dictionary,
               it returns the dictionary instead.

    Raises:
        NotImplementedError: If the reward function is not implemented for the given data source.
    """    
    if data_source == "openai/gsm8k":
        from . import gsm8k

        res = gsm8k.compute_score(solution_str, ground_truth)
    elif data_source in [
        "lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval", "HuggingFaceH4/MATH-500",
        "violetxi/omni-math-above-l7-rule-gemini-pro-filtered",
        "prime_test"    # for generated data without one single source
    ]:
        # from . import math
        # res = math.compute_score(solution_str, ground_truth)
        # [Optional] Math-Verify Integration
        # For enhanced accuracy, consider utilizing Math-Verify (https://github.com/huggingface/Math-Verify).
        # Note: Math-Verify needs to be manually installed via pip: `pip install math-verify`.
        # To use it, override the `compute_score` function with the following implementation:

        from . import math_verify
        res = math_verify.compute_score(solution_str, ground_truth)
    elif data_source == "math_dapo" or data_source.startswith("aime"):
        from . import math_dapo

        res = math_dapo.compute_score(solution_str, ground_truth)
    elif data_source in [
        "numina_aops_forum",
        "numina_synthetic_math",
        "numina_amc_aime",
        "numina_synthetic_amc",
        "numina_cn_k12",
        "numina_olympiads",
        'HuggingFaceH4/aime_2024',
        'opencompass/AIME2025',
        'HuggingFaceH4/MATH-500',
        'agentica-org/DeepScaleR-Preview-Dataset',
        'AI-MO/aimo-validation-amc',
        'Hothan/OlympiadBench',
        'violetxi/deepscale_qwen3-1.7b_box',
        'violetxi/deepscale_qwen3-1.7b_box-base',
        'CohenQu/POPE-hard-dataset-Qwen3-4B-Instruct-32k-128-filtered',
        'CohenQu/AceReason-Math-Qwen3-4B',
        'HerrHruby/acemath_rl_4b_inst_hard',
        'violetxi/omni-math-difficulty-2',
        'violetxi/omni-math-difficulty-5',
        'violetxi/omni-math-difficulty-8',
        'violetxi/omni-math-difficulty-1_2',
        'violetxi/omni-math-difficulty-2_3',
        'violetxi/omni-math-difficulty-4_6',
        'violetxi/omni-math-difficulty-5_7',
        "violetxi/qwen8b-rewrite-omni-l4",
        'violetxi/qwen8b-rewrite-omni-l5',
        'violetxi/qwen8b-rewrite-omni-l8',
        'violetxi/qwen8b-rewrite-omni-l1_2-score', 'violetxi/qwen4b-thinking-omni-l1_2-score',
        'violetxi/omni-math-difficulty-4', 'violetxi/qwen4b-thinking-omni-l1_2',
        'violetxi/omni-filtered-by-cohen', 'violetxi/test-qwen4b-rewrite-omni-l2',
        'violetxi/test-qwen8b-rewrite-omni-l1_2', 'violetxi/test-qwen8b-rewrite-omni-l1_2-score',
        'violetxi/pope-hard-w-guide-gemini-solution', 'violetxi/pope-hard-w-guide-gemini-solution-hint', 
        'violetxi/omni-math-above-l6', 'violetxi/omni-math-above-l7', 
        'violetxi/qwen4b-instruct-2507-omni-l7-score-steps',
        'd1shs0ap/unified-hard-set-with-student-solutions-guided-rl',
        'violetxi/judge-calibration_set1_16k_qwen3-4b-score-steps',
        'violetxi/set1_16k_feedback_qwen3-4b-instruct-2507',
        'MathArena/aime_2026', 'MathArena/hmmt_nov_2025',
        'Hwilner/imo-answerbench',
        'CohenQu/POPE-hard-dataset-Qwen3-4B-Instruct-32k-128-filtered-iter3-gemini-success',
        'd1shs0ap/unified-hard-set-with-student-solutions-guided-rl'
    ]:
        from . import prime_math
        res = prime_math.compute_score(solution_str, ground_truth)
    elif data_source in ["livecodebench/code_generation_lite", "livecodebench/code_generation"]:
        from recipe.r1.tasks import livecodebench

        res = livecodebench.compute_score(solution_str, ground_truth)
    elif data_source == "Idavidrein/gpqa":
        from recipe.r1.tasks import gpqa

        res = gpqa.compute_score(solution_str, ground_truth)
    elif data_source == "hicai-zju/SciKnowEval":
        from . import sciknoweval_judge

        res = sciknoweval_judge.compute_score(solution_str, ground_truth, extra_info=extra_info)
    elif data_source == "violetxi/olympiadbench_oe_physics":
        from . import olympiadbench_oe_judge

        res = olympiadbench_oe_judge.compute_score(solution_str, ground_truth, extra_info=extra_info)
    elif data_source in ["codecontests", "apps", "codeforces", "taco"]:
        # Use the passed sandbox_fusion_url if available
        if sandbox_fusion_url:
            from . import sandbox_fusion

            # Pass the URL directly, ground_truth likely contains test cases here
            res = sandbox_fusion.compute_score(
                sandbox_fusion_url, concurrent_semaphore, memory_limit_mb, solution_str, ground_truth, continuous=True
            )
        else:
            # If no sandbox URL is provided, fall back to prime_code or raise error
            from . import prime_code

            # Assuming prime_code doesn't need the URL
            res = prime_code.compute_score(solution_str, ground_truth, continuous=True)
    elif data_source in ["hiyouga/geometry3k"]:
        from . import geo3k

        res = geo3k.compute_score(solution_str, ground_truth)
    elif data_source in [
        "searchR1_nq",
        "searchR1_triviaqa",
        "searchR1_popqa",
        "searchR1_hotpotqa",
        "searchR1_2wikimultihopqa",
        "searchR1_musique",
        "searchR1_bamboogle",
    ]:
        from . import search_r1_like_qa_em

        res = search_r1_like_qa_em.compute_score(solution_str, ground_truth)

    else:
        raise NotImplementedError(f"Reward function is not implemented for {data_source=}")

    if isinstance(res, dict):
        return res
    elif isinstance(res, int | float | bool):
        return float(res)
    else:
        return float(res[0])


@deprecated("verl.utils.reward_score.default_compute_score")
def _default_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    sandbox_fusion_url=None,
    concurrent_semaphore=None,
    memory_limit_mb=None,
):
    """
    Legacy function API to be deprecated. Please use `default_compute_score` instead.
    """
    return default_compute_score(
        data_source, solution_str, ground_truth, extra_info, sandbox_fusion_url, concurrent_semaphore, memory_limit_mb
    )


__all__ = ["default_compute_score"]
