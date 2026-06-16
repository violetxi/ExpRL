# Judge misplace rate

Misplace rate = (FNR + FPR) / 2; judge asserts *correct* iff Likert ≥ 4.

### Math (INT / Qwen8B rollouts)

| LLM Judge | Reference condition | Misplace rate |
|---|---|---|
| Qwen3-0.6B | ref | 48.6% |
| Qwen3-0.6B | noref | 48.5% |
| Qwen3-0.6B | wrongref | 47.5% |
| Qwen3-4B | ref | 17.8% |
| Qwen3-4B | noref | 39.2% |
| Qwen3-4B | wrongref | 50.4% |
| Qwen3-8B | ref | 18.8% |
| Qwen3-8B | noref | 36.0% |
| Qwen3-8B | wrongref | 52.6% |
| Qwen3-14B | ref | 18.2% |
| Qwen3-14B | noref | 38.5% |
| Qwen3-14B | wrongref | 50.2% |

### SciKnowEval MCQ (L3 loose)

| LLM Judge | Reference condition | Misplace rate |
|---|---|---|
| Qwen3-0.6B | ref | 42.0% |
| Qwen3-0.6B | noref | 47.1% |
| Qwen3-0.6B | wrongref | 44.1% |
| Qwen3-4B | ref | 14.0% |
| Qwen3-4B | noref | 37.5% |
| Qwen3-4B | wrongref | 46.0% |
| Qwen3-8B | ref | 9.8% |
| Qwen3-8B | noref | 31.9% |
| Qwen3-8B | wrongref | 48.8% |
| Qwen3-14B | ref | 14.7% |
| Qwen3-14B | noref | 29.3% |
| Qwen3-14B | wrongref | 47.6% |

### SciKnowEval OE (v1)

| LLM Judge | Reference condition | Misplace rate |
|---|---|---|
| Qwen3-0.6B | ref | 49.4% |
| Qwen3-0.6B | noref | 50.0% |
| Qwen3-0.6B | wrongref | 51.6% |
| Qwen3-4B | ref | 11.4% |
| Qwen3-4B | noref | 25.2% |
| Qwen3-4B | wrongref | 36.7% |
| Qwen3-8B | ref | 19.1% |
| Qwen3-8B | noref | 37.0% |
| Qwen3-8B | wrongref | 43.1% |
| Qwen3-14B | ref | 12.3% |
| Qwen3-14B | noref | 27.6% |
| Qwen3-14B | wrongref | 36.4% |

### LiveCodeBench v6 (code)

| LLM Judge | Reference condition | Misplace rate |
|---|---|---|
| Qwen3-4B | ref | 9.7% |
| Qwen3-4B | noref | 8.2% |
| Qwen3-4B | wrongref | 10.0% |

