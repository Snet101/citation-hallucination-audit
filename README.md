# Citation Hallucination Audit

A collaborative citation-evaluation project with a repaired, inspectable generation/retrieval pipeline and preserved historical artifacts.

> **Historical research results remain under review.** The original files contain duplicated observations and were evaluated with flawed answer boundaries, correction merging, and statistics. The repair below passes software tests; it does not retroactively validate the historical percentages or establish new TinyLlama/RAG findings. Read [results correction](docs/results-correction.md) and [repair notes](docs/pipeline-repair.md).

## What is now implemented

- Explicit run/category/prompt/variant keys; duplicate or incomplete pairs are rejected.
- Sequential answer-only generation, with model input, retrieved evidence, and generated answer in separate fields.
- One structured citation-candidate detector used before and after correction.
- Per-candidate retrieval logs, fresh corrected-answer judgments, answer hashes, and complete correction verification.
- Strict one-to-one correction replacement with an unchanged evaluation population.
- Paired prompt-level bootstrap analyses with observed point estimates and explicit undefined-pair exclusions.
- Normalized inner-product FAISS indexing with corpus/model/index hashes and dimension/count checks.
- New output directories, exclusive file creation, visible failed-run status, and no automatic append/resume.
- Offline fixtures, tests, and CI that require no credentials or model download.

## Run the tested offline example

From the repository root, use Python 3.12:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.lock.txt
python -m pip install --no-deps --no-build-isolation -e .
python -m pytest -q
python -m ruff check .
python -m citation_audit run --mode fixture --manifest tests/fixtures/manifest.jsonl --fixture-responses tests/fixtures/responses.jsonl --corpus tests/fixtures/corpus.jsonl --run-id fixture-demo --out runs/fixture-demo --threshold 0.95 --B 1000
```

On macOS/Linux use `source .venv/bin/activate` instead. The example has three paired questions (six canned answers), three synthetic passages, and two selected corrections. It uses a deterministic token-hash encoder and actual FAISS retrieval. **It is a software fixture, not a new language-model experiment or evidence of retrieval quality.** Its reference markers are synthetic, not real bibliography entries.

Use a fresh output directory for each invocation. Reusing `runs/fixture-demo` is deliberately refused. An interrupted or failed run retains an incomplete/failed `status.json`; do not interpret its partial files as completed research. Resume is intentionally not implemented.

The full `run` command requires at least two complete prompt pairs per category because it computes paired uncertainty. A one-pair smoke test can use the separate inference/verification commands without requesting bootstrap intervals.

## Inspect the outputs

```text
status.json                     state, mode, inputs/hashes, code/package provenance
generation-config.json          generator identity and settings
manifest.jsonl                  complete planned evaluation identities
index/                          passages, FAISS index, hashed metadata
completions.jsonl               exact model input and answer-only text
baseline-verification.jsonl     every candidate and its retrieval evidence
selected.jsonl                  citation-bearing answers with at least one mismatch
contexts.jsonl                  question-based retrieval queries and passages
corrected-generations.jsonl     separate correction input and corrected answer
corrected-verification.jsonl    fresh judgments of corrected answers
final-records.jsonl             explicit before/after records; same population
summary-before.json             counts/rates with named denominators
summary-after.json              corresponding post-correction summary
transitions.json                counts and status transitions
bootstrap-conditions.json       paired with-citation minus neutral citation-presence effect
bootstrap-correction.json       paired after-minus-before citation-presence effect
result.json                     completed run counts
```

The two default bootstrap files measure **detected citation presence**, which is defined for all answers. They do not measure citation truth. Other supported binary retrieval metrics can be requested with the standalone bootstrap command; null outcomes exclude whole prompt pairs and their counts are reported. Too few defined pairs cause an error rather than an invented interval.

## Metric meaning and limits

| Field | Meaning |
|---|---|
| `answer_status` | `empty_answer`, `no_detected_citation`, or `citation_present` |
| `abstention_cue` | A simple phrase heuristic; not a validated refusal classifier |
| `any_retrieval_support` | At least one candidate's associated claim passes the configured similarity threshold; null when no candidates |
| `all_candidate_retrieval_support` | Every detected candidate passes that proxy; null when no candidates |
| `source_exists` / `source_supports_claim` | Always unknown (`null`) in this automated pipeline |

Missing detected citations are not fabricated citations. A matching passage is not proof that a reference is real or supports a claim. Numbered references are detected as markers; the code does not resolve them to bibliography entries. A partial corpus cannot prove that a missing source does not exist. No citation-fabrication rate is calculated.

The detector recognizes a limited set of structured formats: URLs, DOI strings, bracketed numbers, and simple author-year forms. General phrases such as “according to experts” are not treated as formal citations. The small labeled test set checks specific regressions, not real-world detector accuracy. Similarity thresholds require independent validation for a research use; `0.95` above belongs only to the controlled fixture.

## Audit the historical files without modifying them

```powershell
python -m citation_audit audit-history --repo . --out runs/historical-audit
```

This writes separately labeled derived copies, collapses only identical full records with the same historical key, and refuses conflicting records. It records source hashes and counts. It **does not strip guessed prefixes, reverify old answers, repair old labels, or reconstruct missing experiment settings**.

| Historical file | Raw rows | Distinct full records |
|---|---:|---:|
| `tinyllama_completions.jsonl` | 500 | 250 |
| `verification_results.jsonl` | 500 | 250 |
| `corrected_completions.jsonl` | 78 | 39 |

The baseline covers 125 prompt IDs in two variants. The saved manifest has different coverage from the current builder. The original paper and all files under `artifacts/` remain unchanged; their hashes are recorded in the baseline manifest.

Three historical files have different Windows-checkout and Git-stored line endings. `docs/baseline-git-objects.json` records both original byte fingerprints without changing the original archive manifest. Preservation checks verify these exact known representations and the unchanged indexed Git objects across platforms.

## New model runs: prerequisites, not reproduced results

The Hugging Face adapters passed a bounded CPU smoke run using pinned TinyLlama and MiniLM revisions: four real generated answers completed the pipeline without population changes. A separate synthetic correction input exercised real correction generation and fresh MiniLM/FAISS reverification. See [model smoke validation](docs/model-smoke.md). These are integration checks on a tiny synthetic corpus, not new research findings. The original FineWeb subset is not recovered; do not describe the historical experiment as reproduced.

The tested optional CPU environment is recorded in `requirements-model.lock.txt`. Install that snapshot, then the editable package, to reproduce the environment. Alternatively `python -m pip install -e ".[models]"` resolves the broader supported ranges; that can select different versions and requires new validation. The model snapshot is separate from the small offline/CI lock. Record and validate the actual environment before publishing a model experiment.

For a new run supply:

1. A paired manifest (or use `manifest --help` with a TSV containing `data_type` and `vanilla`). The builder refuses duplicate source text or insufficient sampling population.
2. A licensed, local passage corpus in JSONL with nonempty string `passage_id`, `text`, and `source`. Preserve source document IDs, URLs, license/retrieval details, corpus revision, filtering/subset recipe, and chunking settings. No exact historical FineWeb reconstruction recipe can be supplied from the saved repository.
3. Immutable 40-character model and embedding-model commit revisions. Both are required in model mode before downloads start.
4. A fresh run ID/output directory, declared threshold, generation token limit/device, and actual source/protocol metadata.

Example command shape (replace the revision placeholders with real resolved commits):

```text
python -m citation_audit run --mode model --manifest NEW_MANIFEST.jsonl --corpus NEW_CORPUS.jsonl --run-id NEW_RUN_ID --out runs/NEW_RUN_ID --model-revision MODEL_COMMIT_40_HEX --embedding-revision EMBED_COMMIT_40_HEX --max-new-tokens 128 --device -1
```

Model mode uses the tokenizer's chat template, deterministic decoding, and answer-only output. This is a **new protocol**, not a retroactive reconstruction of the saved run. Greedy decoding does not guarantee bitwise identical outputs across all hardware/backends. Use bounded benign examples to validate the actual model integration first. Inspect truncation/model context limits for the chosen corpus and model before scaling up.

## Individual stages and compatibility

`python -m citation_audit --help` lists explicit subcommands: `manifest`, `build-index`, `infer`, `extract`, `verify`, `select`, `retrieve`, `correct`, `merge`, `summarize`, `bootstrap`, `plot`, `audit-history`, and `run`.

Each command has `--help`, requires a **new** `--out` directory, and records provenance/status. Standalone stage outputs use `records.jsonl` and/or `result.json`; pass those exact paths to the next stage. For corrected verification, use `verify --answer-field corrected_answer_text`. `merge` requires baseline judgments, fresh correction judgments, and the corresponding expected corrected generations. `summarize` accepts final records with `--phase before` or `--phase after`.

The original filenames under `src/` and `analysis/` are now thin CLI adapters; they have no import-time model loading, downloads, plotting, or output writes. They no longer silently read or overwrite the historical artifacts. Their arguments have changed; run them with `--help` after installing the package.

Optional plotting: `python -m pip install -e ".[plots]"`, then `python -m citation_audit plot --input runs/fixture-demo/summary-before.json --out runs/fixture-plot`. Plot labels use explicit proxy metric names, not hallucination-rate labels.

## Authors and review

Sanvi Nethikunta and Rithik Gumpu. The repair was prepared with AI assistance; individual contribution claims should identify work each collaborator recognizes and can explain. Newly repaired code and tests are not presented as the original conference version. See [pipeline repair](docs/pipeline-repair.md) for design choices and validation limits.
