import datasets
import numpy as np



def create_steps(data_source):
    ds = datasets.load_dataset(data_source, split="train")
    df = ds.to_pandas()
    df = df.drop_duplicates(subset=['prompt'])
    # convert to dataset
    ds = datasets.Dataset.from_pandas(df)
    # add index for each data item
    # ds = ds.map(_extract_index)
    STEP_SPLIT_TOKEN = "###"
    new_data = {
        "problem": [], "problem_index": [], "step_solution": [], "step_index": [],
        "step_only": [],    # content only for the step
        # keep original information
        "prompt": [], "original_response": [], "ability": [], "data_source": [], 
        "original_score": [], "reward_model": [], "extra_info": [],        

    }
    for item in ds:
        problem = item['extra_info']['question']
        problem_index = item['extra_info']['index']
        # extract steps
        if "responses" in item:
            response = item['responses']
        else:
            response = item['response']
        steps = response.split(STEP_SPLIT_TOKEN)
        cur_step = ""        
        # # skip the last step
        # for step_idx, step in enumerate(steps[:-1]):
        # do not skip the last step
        for step_idx, step_str in enumerate(steps):
            if len(steps) > 10:    # skip too long steps
                continue
            # current step only has the string content of the step, matching the training setting
            # cur_step = step
            if step_idx == 0:
                cur_step = step_str
            else:
                cur_step = cur_step + STEP_SPLIT_TOKEN + step_str            
            new_data['problem'].append(problem)
            new_data['problem_index'].append(problem_index)
            new_data['step_solution'].append(cur_step)
            new_data['step_index'].append(step_idx)
            new_data['step_only'].append(step_str)
            # retrain old data items
            new_data['prompt'].append(item['prompt'])
            new_data['original_response'].append(response)
            new_data['ability'].append(item['ability'])
            new_data['data_source'].append(item['data_source'])
            new_data['original_score'].append(item['score'])
            new_data['reward_model'].append(item['reward_model'])
            new_data['extra_info'].append(item['extra_info'])
            print(f"Number of steps: {len(steps)}")

    new_ds = datasets.Dataset.from_dict(new_data)    
    new_ds.push_to_hub(f"{data_source}-steps", private=False)
    print(f"Saved {len(new_ds)} rows to {data_source}-steps")


if __name__ == "__main__":
    # data_source = "violetxi/qwen4b-no-thinking-omni-l5-score"
    # create_steps(data_source)
    # data_source = "violetxi/qwen4b-no-thinking-omni-l2-score"
    # create_steps(data_source)
    # # step 30
    # data_source = "violetxi/test_cohen_qwen4b_genrm-smooth-delta_30-score"
    # create_steps(data_source)
    # # step 60
    # data_source = "violetxi/test_cohen_qwen4b_genrm-smooth-delta_60-score"
    # create_steps(data_source)
    # # step 90
    # data_source = "violetxi/test_cohen_qwen4b_genrm-smooth-delta_90-score"
    # create_steps(data_source)
    # # step 120
    # data_source = "violetxi/test_cohen_qwen4b_genrm-smooth-delta_120-score"
    # create_steps(data_source)
    # # main
    # data_source = "violetxi/test_cohen_qwen4b_genrm-smooth-delta_main-score"
    # create_steps(data_source)

    # #### POPE HARD (canot solve w/ guidance)
    # data_source = "violetxi/test_pope_hard_qwen4b_instruct_2507-score"
    # create_steps(data_source)

    # #### Omni Math L7 and above (Qwen3-4B-Instruct-2507)
    # data_source = "violetxi/omni-rule-l7-above-gemini-pro-filtered_qwen3-4b-instruct-2507-score"
    # create_steps(data_source)

    ##### ICML Set1 ######
    data_source = "violetxi/judge-calibration_set1_16k_qwen3-4b-score"
    create_steps(data_source)
