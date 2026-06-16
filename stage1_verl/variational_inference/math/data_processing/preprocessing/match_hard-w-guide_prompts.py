from datasets import load_dataset
import re
import unicodedata

# ---------------- config ----------------
data_path_A = "CohenQu/POPE-arxiv-no_guide-0.0-0.64-256"
data_path_B = "CohenQu/POPE-hard-dataset-Qwen3-4B-Instruct-32k-128-filtered-iter3-gemini-success"
out_repo = "violetxi/pope-hard-w-guide-gemini-solution"

MIN_SUBSTR_LEN = 80

# ---------------- matching utils ----------------
def norm(s: str) -> str:
    s = "" if s is None else s
    s = unicodedata.normalize("NFKC", s)        # normalize unicode variants
    s = s.replace("\u00a0", " ")                # NBSP -> space
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)               # collapse spaces/tabs
    s = re.sub(r"\n{3,}", "\n\n", s)            # collapse many blank lines
    return s.strip()

def in_A_by_exact_or_substring(prob_norm: str, A_set: set, A_list: list) -> bool:
    # fast path: exact normalized match
    if prob_norm in A_set:
        return True

    # A contains B
    if len(prob_norm) >= MIN_SUBSTR_LEN:
        for a in A_list:
            if prob_norm in a:
                return True

    # B contains A
    for a in A_list:
        if len(a) >= MIN_SUBSTR_LEN and a in prob_norm:
            return True

    return False

ds_a = load_dataset(data_path_A)
ds_b = load_dataset(data_path_B)

b_test = ds_b["test"]
b_cols = b_test.column_names

print(f"Dataset B columns: {b_cols}")
print(f"Total rows in B (test): {len(b_test)}")

A_set = set()
for split in ds_a:
    for row in ds_a[split]:
        A_set.add(norm(row["prompt"][0]["content"]))
A_list = list(A_set)

print(f"Total unique normalized problems in A: {len(A_set)}")

def keep_if_not_in_A(example, idx):
    prob_norm = norm(example["problem"])
    inA = in_A_by_exact_or_substring(prob_norm, A_set, A_list)
    if idx % 200 == 0:
        print(f"[{idx}] in A? {inA}")
    return not inA

ds_b_filtered = b_test.filter(
    keep_if_not_in_A,
    with_indices=True,    
)

ds_b_filtered = ds_b_filtered.select_columns(b_cols)
print(f"Rows in B after filtering (NOT in A): {len(ds_b_filtered)}")
print(f"Rows removed (overlap with A): {len(b_test) - len(ds_b_filtered)}")
print(f"Filtered columns: {ds_b_filtered.column_names}")

# ---------------- push / save ----------------
ds_b_filtered.push_to_hub(out_repo)
# ds_b_filtered.save_to_disk("pope_hard_not_in_A")
