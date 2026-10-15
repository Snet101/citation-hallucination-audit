# Citation Hallucination Audit

A collaborative TinyLlama citation-evaluation project containing historical model outputs, retrieval-verification code, and an experimental retrieval-augmented correction pipeline.

> **Historical evaluation under review.** A code and artifact audit found duplicated observations, an expanding correction merge, stale correction labels, and evaluation-boundary problems. Previously reported percentages should not be interpreted as validated citation-fabrication rates or established RAG improvements. The historical files are preserved unchanged while repairs are developed. See [results review](docs/results-correction.md).

## Project scope

The intended experiment compares a neutral prompt with a version requesting sources, then extracts citation-like statements, retrieves related evidence with MiniLM/FAISS, and attempts a retrieval-augmented correction. Embedding similarity is a retrieval proxy; it does not prove that a cited source exists or supports a claim.

## What the saved artifacts establish

At baseline commit `e99aa094384fcc5d23085d64a3bc2fc99c167f2f`:

| Historical file | Saved rows | Distinct full records |
|---|---:|---:|
| `artifacts/tinyllama_completions.jsonl` | 500 | 250 |
| `artifacts/verification_results.jsonl` | 500 | 250 |
| `artifacts/corrected_completions.jsonl` | 78 | 39 |

The distinct baseline records represent 125 prompt IDs in two variants. The saved manifest and current manifest builder have different coverage. The repository does not currently establish a completed experiment with 250 distinct prompts and 500 independent responses.

These observations describe the committed artifacts. They do not establish what additional original files or runs may have existed elsewhere.

## Current limitations

- The final correction merge can multiply observations when both inputs contain repeated keys.
- Correction generation retains the original support label; the final merge does not consume fresh per-answer correction judgments from the separate reverification step.
- Saved generation fields include prompt text, and saved corrected fields include retrieved context. The present evaluation does not reliably isolate generated answer text.
- Missing detected citations are counted as unsupported; that category is not equivalent to fabricated citations. One matching fragment can mark an entire answer supported.
- Citation extraction differs across scripts, and the bootstrap entry point consumes aggregate summaries rather than paired prompt-level observations.
- The exact historical retrieval corpus/index and a complete tested setup recipe are not available in this repository. Full historical reproduction has not been demonstrated.

## Repository contents

```text
src/                              historical pipeline scripts
analysis/                         historical analysis scripts
artifacts/                        unchanged historical outputs and intermediate files
docs/results-correction.md        findings, claim status, and repair criteria
docs/baseline-manifest.json       SHA-256 hashes of the preserved baseline files
citation_hallucination_audit.pdf  historical paper; findings are under review
```

The scripts are retained for inspection. A verified end-to-end quickstart will be added after the pipeline and environment are repaired; the current repository should not be described as fully reproducible.

## Repair status

Step 1 preserves the baseline and corrects the public description. No model inference, artifact deduplication, evaluator repair, or new result computation is included in this step.

Next repairs will enforce unique evaluation identities, separate inputs/context from generated answers, persist fresh correction judgments, validate one-to-one merges, align detectors, and calculate uncertainty from paired prompt observations. Historical and newly generated results will remain distinguishable.

## Historical paper

[`citation_hallucination_audit.pdf`](citation_hallucination_audit.pdf) is retained as a historical document. Its sample-size, headline-rate, and RAG-improvement claims are under review; retaining the PDF does not endorse those claims as validated results.

## Authors

Sanvi Nethikunta and Rithik Gumpu. Individual contribution details will be documented with the collaborators as the project is repaired.
