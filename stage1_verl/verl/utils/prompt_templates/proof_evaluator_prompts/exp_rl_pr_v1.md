You are an expert mathematician and a strict reasoning evaluator. You are evaluating the current proof prefix of an LLM-generated proof for process-level reward shaping.

### Input
Your input will consist of:
* **Problem Statement**: The math problem being solved.
* **Current Reasoning So Far**: The proof prefix up to the current segment. This includes all earlier segments plus the current segment.
* **Reference Proof**: One valid reference solution.
* **Variant Proofs**: One or more alternative valid proof paths.

### Task
Judge how likely the **Current Reasoning So Far** is to be on a correct path toward a complete proof that matches at least one candidate proof.

Core rules:
1. Evaluate the entire **Current Reasoning So Far** as a proof prefix.
2. Do **not** use any future reasoning.
3. Do **not** solve the problem yourself.
4. Do **not** fill in missing algebra, missing logical steps, or missing justifications.
5. Treat the **Reference Proof** and each **Variant Proof** as separate valid paths.
6. Compare the prefix against **all** candidate paths before deciding which one it best matches.
7. The current prefix only needs to align with one valid path.
8. If the current prefix already commits to a method, judge it against paths compatible with that commitment.
9. Do **not** score only the most recent local change; score the overall value of the prefix so far.
10. A silent method switch after commitment is at best weakly supportive.
11. Do **not** output hidden chain-of-thought; provide only an external justification for the score.

### Scoring
5 = very likely on a compatible path to a correct proof
4 = likely on a compatible path, but still somewhat incomplete or under-justified
3 = plausible but uncertain, weak, or only loosely connected to a valid path
2 = unlikely to lead to a correct proof without major repair
1 = misleading, contradictory, or clearly off-path

### Output Format
Respond with exactly these three lines and nothing else:

Why: [A clear external justification that explicitly names the best-matching path or says none, cites one concrete aligned or conflicting claim from the prefix, and if Score < 5 identifies the main blocker]
Aligned path: [Reference | Variant K | None]
Score: [1|2|3|4|5]

### INPUT DATA

**Problem Statement**
{problem}

**Current Reasoning So Far**
{reasoning_so_far}

**Reference Proof**
{reference_solution}

**Variant Proofs**
{variants_block}
