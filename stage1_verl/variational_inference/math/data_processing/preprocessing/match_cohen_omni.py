import datasets
import re
import os
import asyncio
from openai import AsyncOpenAI
from typing import Tuple, Optional
import unicodedata

SYSTEM_PROMPT = """You are a text comparison expert. Your task is to determine if Problem A and Problem B are **identical in their core problem statement**, ignoring all markup, formatting, and illustrative code blocks.

**Your Goal:**
Compare the *prose* of the problems. You must **completely ignore** the following types of "noise":
1.  Markup Blocks: Any text inside square brackets, especially `[asy]...[/asy]` blocks or `` tags.
2.  LaTeX: All LaTeX commands (e.g., `\mathcal{r}`, `\frac`, `\times`) and the math delimiters (`$`, `\(`, `\)`).
3.  Whitespace: Any differences in spaces, newlines, or tabs.
4.  Punctuation: Minor differences in punctuation.

**Your Reasoning Process:**
1.  Read Problem A and state its core text, *excluding* all the "noise" listed above.
2.  Read Problem B and state its core text, *excluding* all the "noise" listed above.
3.  Compare the two core texts and state if they are identical.

**Output Format:**
After your reasoning, you **must** end your response with a final line in this exact format, with no other text after it:
Answer: [YES or NO]
"""

USER_PROMPT = """Problem A:
{problem_a}

Problem B:
{problem_b}

Compare these two problems. Provide your reasoning first, then your final answer.
"""

# def normalize_problem_text(text):
#     """
#     Normalize problem text for robust matching by handling special characters,
#     whitespace, and formatting differences.
#     """
#     text = str(text)
#     text = text.replace('\r\n', '\n').replace('\r', '\n')
#     text = re.sub(r' +', ' ', text)
#     text = re.sub(r'\n+', '\n', text)
#     text = text.strip()
    
#     return text

def normalize_problem_text(s: str) -> str:
    """
    Strips LaTeX, [tag] blocks, and normalizes whitespace/unicode
    to get a 'flat' string for comparison.
    """
    if not s:
        return ""
        
    # 1. Remove [tag] blocks (like [asy]...[/asy])
    # This finds anything starting with '[' and ending with ']'
    s_no_tags = re.sub(r'\[.*?\]', '', s, flags=re.DOTALL)
    
    # 2. Remove LaTeX commands (like \mathcal{r})
    # This finds a backslash followed by letters
    s_no_latex = re.sub(r'\\[a-zA-Z]+', '', s_no_tags)
    
    # 3. Remove math delimiters ($) and other common "noise" chars
    # We remove $, {, }, ^, _
    s_no_symbols = re.sub(r'[${}^_]', '', s_no_latex)
    
    # 4. Unicode Normalization (handles \xa0, etc.)
    # This converts different-looking-but-same-meaning chars
    s_unicode = unicodedata.normalize('NFKC', s_no_symbols)
    
    # 5. Whitespace Collapsing
    # Replaces all newlines, tabs, and multi-spaces with one space
    s_collapsed = re.sub(r'\s+', ' ', s_unicode)
    
    # 6. Final clean: lowercase and strip (removes start/end spaces)
    s_final = s_collapsed.lower().strip()
    
    return s_final


async def llm_compare_problems(
    client: AsyncOpenAI,
    problem_a: str,
    problem_b: str,
    model: str,
    sem: asyncio.Semaphore,
    temperature: float = 0.0,
    enable_thinking: bool = False,
    use_local_model: bool = False,
) -> Tuple[str, bool]:
    """
    Use LLM to determine if two problems are the same.
    Returns (response_str, is_same) where is_same is True if problems match.
    """
    user_msg = USER_PROMPT.format(problem_a=problem_a, problem_b=problem_b)
    
    try:
        async with sem:
            if use_local_model:                
                # For local models (like Qwen), pass enable_thinking via extra_body
                response = await client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    max_tokens=512,
                    n=1,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                    extra_body={"chat_template_kwargs": {"enable_thinking": enable_thinking}},
                )
            else:
                # For OpenRouter models
                response = await client.chat.completions.create(
                    model=model,
                    temperature=temperature,
                    n=1,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_msg},
                    ],
                )
        response_str = response.choices[0].message.content.strip()        
    except Exception as e:
        print(f"Error in LLM call: {e}")
        return "", False
    
    # Parse the answer - look for YES or NO
    try:
        response_upper = response_str.upper()
        if "ANSWER:" in response_upper:
            answer = response_upper.split("ANSWER:")[1].split("\n")[0].strip().lstrip()
        elif "ANSWER" in response_upper:
            answer = response_upper.split("ANSWER")[1].split("\n")[0].strip().lstrip()
        else:
            answer = "NO"
    except Exception as e:
        print(f"Error parsing LLM response: {e}")
        answer = "NO"
    
    if "YES" in answer:
        is_same = True
    else:
        is_same = False
    return response_str, is_same


async def find_matches_for_one_problem(
    client: AsyncOpenAI,
    cohen_problem: str,
    ds_omni: datasets.Dataset,
    omni_problem_col: str,
    model: str,
    sem: asyncio.Semaphore,
    temperature: float = 0.0,
    stop_at_first_match: bool = True,
    enable_thinking: bool = False,
    use_local_model: bool = False,
) -> Tuple[list, int]:
    """
    Find matches for a single cohen problem in omni dataset.
    Returns (list of matched omni indices, number of API calls made).
    """
    # Create tasks for comparing this cohen problem against all omni problems
    # We'll create a dict mapping task -> omni_idx for lookup
    task_to_idx = {}
    tasks_list = []
    
    for omni_idx, omni_problem in enumerate(ds_omni[omni_problem_col]):
        task = llm_compare_problems(
            client, cohen_problem, omni_problem, model, sem, temperature,
            enable_thinking, use_local_model
        )
        tasks_list.append(task)
        task_to_idx[id(task)] = omni_idx  # Use id() to track task identity
        
    matched_indices = []
    api_calls = 0
    
    if stop_at_first_match:
        # Process results as they complete, stop when we find first match
        # Create tasks with their indices
        pending = set()
        task_to_omni_idx = {}
        
        for omni_idx, coro in enumerate(tasks_list):
            task = asyncio.create_task(coro)
            pending.add(task)
            task_to_omni_idx[task] = omni_idx
        
        # Process tasks as they complete
        while pending:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            
            for task in done:
                api_calls += 1
                try:
                    response, is_same = await task
                    omni_idx = task_to_omni_idx[task]
                    
                    if is_same:
                        matched_indices.append(omni_idx)
                        # Found a match! Cancel remaining tasks
                        for remaining_task in pending:
                            remaining_task.cancel()
                        return matched_indices, api_calls
                except Exception as e:
                    print(f"Error processing task: {e}")
                    continue
    else:
        # Execute all comparisons to find all matches
        results = await asyncio.gather(*tasks_list)
        api_calls = len(results)
        
        for omni_idx, (response, is_same) in enumerate(results):
            if is_same:
                matched_indices.append(omni_idx)
    
    return matched_indices, api_calls


async def find_llm_matches(
    client: AsyncOpenAI,
    unmatched_cohen_problems: dict,  # {normalized: original}
    ds_omni: datasets.Dataset,
    omni_problem_col: str,
    model: str,
    max_concurrency: int,
    temperature: float = 0.0,
    stop_at_first_match: bool = True,
) -> dict:
    """
    Use LLM to find matches for unmatched cohen problems.
    Processes one cohen problem at a time to allow early stopping.
    Returns dict mapping cohen problem (original text) to matched omni indices.
    """
    sem = asyncio.Semaphore(max_concurrency)
    
    print(f"\nPreparing LLM matching for {len(unmatched_cohen_problems)} unmatched problems...")
    print(f"Comparing against {len(ds_omni)} omni problems...")
    print(f"Max LLM calls per problem: {len(ds_omni)}")
    print(f"Stop at first match: {stop_at_first_match}")
    
    matches = {}  # cohen_original -> list of omni indices
    total_calls = 0
    
    # Process one cohen problem at a time
    for idx, cohen_original in enumerate(unmatched_cohen_problems.values(), 1):
        print(f"\nProcessing cohen problem {idx}/{len(unmatched_cohen_problems)}...")
        
        matched_indices = await find_matches_for_one_problem(
            client, cohen_original, ds_omni, omni_problem_col, 
            model, sem, temperature, stop_at_first_match
        )
        
        if matched_indices:
            matches[cohen_original] = matched_indices
            print(f"  ✓ Found {len(matched_indices)} match(es) for problem {idx}")
        else:
            print(f"  ✗ No matches found for problem {idx}")
        
        # Track total API calls made
        # Note: With early stopping, we don't know exact count without modifying the function
        # For now, estimate as len(ds_omni) per problem or until match found
        
    print(f"\n{'='*60}")
    print(f"LLM matching complete!")
    print(f"Found matches for {len(matches)}/{len(unmatched_cohen_problems)} unmatched problems")
    print(f"{'='*60}")
    
    return matches


async def filter_omni_by_cohen_async(
    ds_cohen, 
    ds_omni, 
    cohen_problem_col='problem', 
    omni_problem_col='problem',
    use_llm_matching=False,
    model="qwen/qwen3_8b",
    max_concurrency=1024,
    temperature=0.0,
    enable_thinking=False,
    use_local_model=False,
):
    """
    Filter ds_omni to keep only rows where the problem exists in ds_cohen.
    For each cohen problem: tries exact string matching first, then LLM if needed.
    
    Args:
        ds_cohen: Dataset containing reference problems
        ds_omni: Dataset to be filtered
        cohen_problem_col: Column name for problems in ds_cohen (default: 'problem')
        omni_problem_col: Column name for problems in ds_omni (default: 'problem')
        use_llm_matching: Whether to use LLM for unmatched problems (default: False)
        model: LLM model name (default: "qwen/qwen3_8b")
        max_concurrency: Max concurrent LLM calls (default: 256)
        temperature: LLM temperature (default: 0.0)
        enable_thinking: Enable thinking mode for Qwen models (default: False)
        use_local_model: Whether using local model (default: False)
    
    Returns:
        Filtered dataset containing only problems that exist in ds_cohen
    """
    print("=" * 60)
    print("MATCHING COHEN PROBLEMS TO OMNI DATASET")
    print("=" * 60)
    
    # Extract unique cohen problems
    cohen_problems = set()
    for problem in ds_cohen[cohen_problem_col]:
        normalized = normalize_problem_text(problem)
        cohen_problems.add(normalized)
    
    print(f"Total unique problems in ds_cohen: {len(cohen_problems)}")
    print(f"Total problems in ds_omni: {len(ds_omni)}")
    
    # Build a fast lookup index for omni problems (normalized -> indices)
    print("\nBuilding omni problem index...")
    omni_normalized_to_indices = {}
    for omni_idx, omni_problem in enumerate(ds_omni[omni_problem_col]):
        norm_omni = normalize_problem_text(omni_problem)
        if norm_omni not in omni_normalized_to_indices:
            omni_normalized_to_indices[norm_omni] = []
        omni_normalized_to_indices[norm_omni].append(omni_idx)
    
    print(f"Built index with {len(omni_normalized_to_indices)} unique omni problems")
    client = None
    if use_llm_matching:
        if use_local_model:
            # For local models (vLLM server)
            client = AsyncOpenAI(api_key="EMPTY", base_url="http://localhost:8000/v1")
        else:
            # For OpenRouter
            OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
            if not OPENROUTER_API_KEY:
                raise EnvironmentError("OPENROUTER_API_KEY env var is not set.")
            client = AsyncOpenAI(api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1")
        sem = asyncio.Semaphore(max_concurrency)
    
    # Process each cohen problem
    all_matched_indices = set()
    exact_matches = 0
    llm_matches = 0
    no_matches = 0
    total_llm_calls = 0
    
    print("\n" + "=" * 60)
    print("PROCESSING COHEN PROBLEMS")
    print("=" * 60)
    
    for idx, norm_cohen in enumerate(cohen_problems):        
        # Try exact match first (O(1) lookup)
        if norm_cohen in omni_normalized_to_indices:
            matched_indices = omni_normalized_to_indices[norm_cohen]
            all_matched_indices.update(matched_indices)
            exact_matches += 1
            if idx % 50 == 0 or idx <= 10:
                print(f"[{idx}/{len(cohen_problems)}] Exact match")
        else:
            if use_llm_matching:
                # Try LLM matching
                if idx % 10 == 0 or idx <= 10:
                    print(f"[{idx}/{len(cohen_problems)}] ⟳ No exact match, trying LLM...")
                
                matched_indices, api_calls = await find_matches_for_one_problem(
                    client, norm_cohen, ds_omni, omni_problem_col,
                    model, sem, temperature, stop_at_first_match=True,
                    enable_thinking=enable_thinking, use_local_model=use_local_model
                )
                total_llm_calls += api_calls
                if matched_indices:
                    all_matched_indices.update(matched_indices)
                    llm_matches += 1
                    
                    print(f"[{idx}/{len(cohen_problems)}] ✓ LLM match found (after {api_calls} calls)")
                else:
                    no_matches += 1                    
                    print(f"[{idx}/{len(cohen_problems)}] ✗ No match found (tried {api_calls} LLM calls)")
            else:
                no_matches += 1
                if idx % 50 == 0:
                    print(f"[{idx}/{len(cohen_problems)}] ✗ No exact match (LLM disabled)")
            
    
    # # Final results
    # print("\n" + "=" * 60)
    # print("FINAL RESULTS")
    # print("=" * 60)
    # print(f"Cohen problems processed: {len(cohen_problems)}")
    # print(f"  - Exact string matches: {exact_matches}")
    # print(f"  - LLM matches: {llm_matches}")
    # print(f"  - No matches: {no_matches}")
    # print(f"Total matches (exact + LLM): {exact_matches + llm_matches}")
    # print(f"Total unique omni indices matched: {len(all_matched_indices)}")
    
    if use_llm_matching:
        print(f"Total LLM API calls made: {total_llm_calls}")
    
    # Filter the dataset
    def is_matched(example, idx):
        return idx in all_matched_indices
    
    filtered_ds = ds_omni.filter(is_matched, with_indices=True)
    
    print(f"Filtered dataset size: {len(filtered_ds)}")    
    
    return filtered_ds


def filter_omni_by_cohen(
    ds_cohen, 
    ds_omni, 
    cohen_problem_col='problem', 
    omni_problem_col='problem',
    use_llm_matching=False,
    model="qwen/qwen3_8b",
    max_concurrency=1024,
    temperature=0.0,
    enable_thinking=False,
    use_local_model=False,
):
    """
    Synchronous wrapper for filter_omni_by_cohen_async.
    """
    return asyncio.run(filter_omni_by_cohen_async(
        ds_cohen, ds_omni, cohen_problem_col, omni_problem_col,
        use_llm_matching, model, max_concurrency, temperature,
        enable_thinking, use_local_model
    ))


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Match cohen problems with omni dataset")
    parser.add_argument("--use-llm", action="store_true", help="Use LLM matching for unmatched problems")
    parser.add_argument("--model", default="qwen/qwen3_8b", help="LLM model name")
    parser.add_argument("--max-concurrency", type=int, default=256, help="Max concurrent LLM calls")
    parser.add_argument("--temperature", type=float, default=0.0, help="LLM temperature")
    parser.add_argument("--enable-thinking", action="store_true", help="Enable thinking mode for Qwen models")
    parser.add_argument("--use-local-model", action="store_true", help="Use local vLLM server instead of OpenRouter")
    parser.add_argument("--output", default=None, help="Output dataset name for HuggingFace Hub (optional)")
    args = parser.parse_args()
    
    print("Loading datasets...")
    ds_cohen = datasets.load_dataset("CohenQu/POPE-MIX-first_guide-no_guide-0.0-0.64-1024-verl", split="train")
    # only select rows where `level` is `hard-without-guidance` and `hard-with-guidance`
    ds_cohen = ds_cohen.filter(lambda x: x["level"] in ["hard-without-guidance", "hard-with-guidance"])
    # only take the rows where `data_source` is `without-guidance`
    ds_cohen = ds_cohen.filter(lambda x: x["data_source"] == "without-guidance")
    ds_cohen = ds_cohen.map(lambda x: {"problem": x["prompt"][0]["content"]})
    ds_cohen = ds_cohen.map(lambda x: {"answer": x['reward_model']['ground_truth']})
    ds_omni = datasets.load_dataset("KbsdJames/Omni-MATH", split="test")
    
    print(f"\nInput datasets loaded:")
    print(f"  - Cohen: {len(ds_cohen)} rows")
    print(f"  - Omni: {len(ds_omni)} rows")
    
    # Filter ds_omni to keep only problems that exist in ds_cohen
    filtered_ds_omni = filter_omni_by_cohen(
        ds_cohen, 
        ds_omni,
        use_llm_matching=args.use_llm,
        model=args.model,
        max_concurrency=args.max_concurrency,
        temperature=args.temperature,
        enable_thinking=args.enable_thinking,
        use_local_model=args.use_local_model,
    )
    
    if args.output:
        print(f"\nPushing to HuggingFace Hub: {args.output}")
        filtered_ds_omni.push_to_hub(args.output, private=False)
        print("Done!")
    else:
        print("\nNo output specified, entering debug mode...")
