import datasets
import random


def expand_dataset(dataset):
    expanded_data = {
        'problem': [],
        'answer': [],
        'gemini_solution': [],
        'hints': [],
        'hint_percentage': []
    }
    
    for example in dataset:
        gemini_solution = example['gemini_solution']
        lines = gemini_solution.split('\n')
        total_lines = len(lines)
        
        for _ in range(4):
            # Randomly sample a percentage between 0 and 30
            hint_pct = random.uniform(2, 30)
            # Calculate the number of lines to use
            num_lines = int(total_lines * hint_pct / 100)
            # Take the first N lines
            hint = '\n'.join(lines[:num_lines]) if num_lines > 0 else ""
            
            expanded_data['problem'].append(example['problem'])
            expanded_data['answer'].append(example['answer'])
            expanded_data['gemini_solution'].append(gemini_solution)
            expanded_data['hints'].append(hint)
            expanded_data['hint_percentage'].append(round(hint_pct, 2))
    
    return datasets.Dataset.from_dict(expanded_data)


ds = datasets.load_dataset("violetxi/pope-hard-w-guide-gemini-solution")
ds_expanded = expand_dataset(ds['test'])

print(f"Original dataset size: {len(ds['test'])}")
print(f"Expanded dataset size: {len(ds_expanded)}")
print(f"Columns: {ds_expanded.column_names}")
print(f"\nSample entries (first 4 rows from same problem):")
for i in range(4):
    hint_lines = len(ds_expanded[i]['hints'].split('\n')) if ds_expanded[i]['hints'] else 0
    total_lines = len(ds_expanded[i]['gemini_solution'].split('\n'))
    print(f"  [{i}] hint_pct: {ds_expanded[i]['hint_percentage']:5.2f}% | hint_lines: {hint_lines:3d} / {total_lines} total")
ds_expanded.push_to_hub("violetxi/pope-hard-w-guide-gemini-solution-hint")