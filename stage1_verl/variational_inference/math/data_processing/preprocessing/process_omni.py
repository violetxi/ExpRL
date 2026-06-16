"""
Process the Omni dataset. Run once is enough.
"""
import numpy as np
from datasets import load_dataset, DatasetDict, concatenate_datasets


ds = load_dataset("asingh15/Omni-MATH-Rule")
ds_l7_rule = ds.filter(lambda x: x["difficulty"] >= 7)
ds_l7_rule = concatenate_datasets([ds_l7_rule['train'], ds_l7_rule['test']])
ds_l7_rule.push_to_hub("violetxi/omni-math-above-l7-rule")
print(ds_l7_rule)

# ds = load_dataset("KbsdJames/Omni-MATH", split="test")
# # L6 and above
# ds_l6 = ds.filter(lambda x: x["difficulty"] >= 6)
# ds_l6 = DatasetDict({"train": ds_l6})
# print(ds_l6)
# print(f"Length of difficulty L6 and above: {len(ds_l6)}")
# ds_l6.push_to_hub("violetxi/omni-math-above-l6")
# L7 and above
# ds_l7 = ds.filter(lambda x: x["difficulty"] >= 7)
# ds_l7 = DatasetDict({"train": ds_l7})
# print(ds_l7)
# print(f"Length of difficulty L7 and above: {len(ds_l7)}")
# ds_l7.push_to_hub("violetxi/omni-math-above-l7")
# # select difficulty 4
# ds_4 = ds.filter(lambda x: x["difficulty"] == 4)
# ds_4 = DatasetDict({"train": ds_4})
# print(f"Length of difficulty 4: {len(ds_4)}")
# ds_4.push_to_hub("violetxi/omni-math-difficulty-4")
# select difficulty 1-2
# ds_1_2 = ds.filter(lambda x: x["difficulty"] >= 1 and x["difficulty"] < 2)
# ds_1_2 = DatasetDict({"train": ds_1_2})
# print(f"Length of difficulty 1-2: {len(ds_1_2)}")
# ds_1_2.push_to_hub("violetxi/omni-math-difficulty-1_2")
# # select difficulty 2-3
# ds_2_3 = ds.filter(lambda x: x["difficulty"] >= 2 and x["difficulty"] < 3)
# ds_2_3 = DatasetDict({"train": ds_2_3})
# print(f"Length of difficulty 2-3: {len(ds_2_3)}")
# ds_2_3.push_to_hub("violetxi/omni-math-difficulty-2_3")
# # select difficulty 2-5
# ds_2 = ds.filter(lambda x: x["difficulty"] == 2)
# ds_2 = DatasetDict({"train": ds_2})
# print(f"Length of difficulty 2: {len(ds_2)}")
# ds_2.push_to_hub("violetxi/omni-math-difficulty-2")
# ds_5 = ds.filter(lambda x: x["difficulty"] == 5)
# ds_5 = DatasetDict({"train": ds_5})
# print(f"Length of difficulty 5: {len(ds_5)}")
# ds_5.push_to_hub("violetxi/omni-math-difficulty-5")
# ds_8 = ds.filter(lambda x: x["difficulty"] == 8)
# ds_8 = DatasetDict({"train": ds_8})
# print(f"Length of difficulty 8: {len(ds_8)}")
# ds_8.push_to_hub("violetxi/omni-math-difficulty-8")
# # ds_2_5 = ds.filter(lambda x: x["difficulty"] >= 2 and x["difficulty"] <= 5)
# # ds_2_5 = DatasetDict({"train": ds_2_5})
# # print(f"Length of difficulty 2-5: {len(ds_2_5)}")
# # ds_2_5.push_to_hub("violetxi/omni-math-difficulty-2_5")
# # # select difficulty 4-6
# # ds_4_6 = ds.filter(lambda x: x["difficulty"] >= 4 and x["difficulty"] <= 6)
# # ds_4_6 = DatasetDict({"train": ds_4_6})
# # print(f"Length of difficulty 4-6: {len(ds_4_6)}")
# # ds_4_6.push_to_hub("violetxi/omni-math-difficulty-4_6")
# # # select difficulty 5-7
# # ds_5_7 = ds.filter(lambda x: x["difficulty"] >= 5 and x["difficulty"] <= 7)
# # ds_5_7 = DatasetDict({"train": ds_5_7})
# # print(f"Length of difficulty 5-7: {len(ds_5_7)}")
# # ds_5_7.push_to_hub("violetxi/omni-math-difficulty-5_7")