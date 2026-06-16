###### Evaluation on Set2 ######

### Qwen3-4B ####
## 16k
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/set2_16k_qwen3-4b data.response_key=response

# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime24_16k_qwen3-4b data.response_key=response

# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime25_16k_qwen3-4b data.response_key=response


#### GRPO ####
## 16k
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/set2_16k_grpo_set2_cont data.response_key=response

# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime24_16k_grpo_set2_cont data.response_key=response

# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime25_16k_grpo_set2_cont data.response_key=response

#### GenRM ####
## 16k
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/set2_16k_genrm_set2_cont data.response_key=response

# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime24_16k_genrm_set2_cont data.response_key=response

# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime25_16k_genrm_set2_cont data.response_key=response

#### PR-Delta ####
## 16k
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/set2_16k_pr_delta_set2_cont data.response_key=response

python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime24_16k_pr_delta_set2_cont data.response_key=response

python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime25_16k_pr_delta_set2_cont data.response_key=response

#### PR-GPPO ####
## 16k
python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/set2_16k_pr_gppo_set2_cont data.response_key=response

python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime24_16k_pr_gppo_set2_cont data.response_key=response

python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime25_16k_pr_gppo_set2_cont data.response_key=response

#### PR-E ####
## 16k
python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/set2_16k_pr_e_set2_cont data.response_key=response

python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime24_16k_pr_e_set2_cont data.response_key=response

python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime25_16k_pr_e_set2_cont data.response_key=response

###### Evaluation on Set1 (Stage 2) ######
#### SFT GRPO ####
## 16k
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/set1_16k_cont_sft_grpo data.response_key=response

# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime24_16k_cont_sft_grpo data.response_key=response

python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime25_16k_cont_sft_grpo data.response_key=response

#### SFT ####
## 16k
python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/set1_16k_sft data.response_key=response

python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime24_16k_sft data.response_key=response

python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime25_16k_sft data.response_key=response


### GRPO ####
## 16k
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/set1_16k_cont_grpo data.response_key=response

# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime24_16k_cont_grpo data.response_key=response

# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime25_16k_cont_grpo data.response_key=response

## 32k
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/set1_32k_cont_grpo data.response_key=response

# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime24_32k_cont_grpo data.response_key=response

# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime25_32k_cont_grpo data.response_key=response

#### PR-E ckpt 150 ####
## 16k (need to re-run inference)
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/set1_16k_cont_pr_e_ckpt150 data.response_key=response

# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime24_16k_cont_pr_e_ckpt150 data.response_key=response

python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime25_16k_cont_pr_e_ckpt150 data.response_key=response

## 32k
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/set1_32k_cont_pr_e_ckpt150 data.response_key=response

# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime24_32k_cont_pr_e_ckpt150 data.response_key=response

# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime25_32k_cont_pr_e_ckpt150 data.response_key=response

#### PR-E main ####
## 16k
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/set1_16k_cont_pr_e data.response_key=response

# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime24_16k_cont_pr_e data.response_key=response

# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime25_16k_cont_pr_e data.response_key=response

## 32k
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/set1_32k_cont_pr_e data.response_key=response

# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime24_32k_cont_pr_e data.response_key=response

#  python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime25_32k_cont_pr_e data.response_key=response

#### PR-Delta ####
## 16k
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/set1_16k_cont_pr_delta data.response_key=response

# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime24_16k_cont_pr_delta data.response_key=response

# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime25_16k_cont_pr_delta data.response_key=response

## 32k
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/set1_32k_cont_pr_delta data.response_key=response

# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime24_32k_cont_pr_delta data.response_key=response

# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime25_32k_cont_pr_delta data.response_key=response


#### PR-GPPO ####
## 16k
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/set1_16k_cont_pr_gppo data.response_key=response

# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime24_16k_cont_pr_gppo data.response_key=response

# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime25_16k_cont_pr_gppo data.response_key=response

## 32k
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/set1_32k_cont_pr_gppo data.response_key=response

# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime24_32k_cont_pr_gppo data.response_key=response

# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime25_32k_cont_pr_gppo data.response_key=response

#### GenRM ####
## 16k
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/set1_16k_genrm_cont data.response_key=response

# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime24_16k_genrm_cont data.response_key=response

# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime25_16k_genrm_cont data.response_key=response

## 32k
# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/set1_32k_genrm_cont data.response_key=response

# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime24_32k_genrm_cont data.response_key=response

# python verl/trainer/main_eval.py data.path=${HF_NAMESPACE:-violetxi}/aime25_32k_genrm_cont data.response_key=response