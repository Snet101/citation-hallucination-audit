# Stage 3: pipeline repair

Baseline: `cec0a5d` (merged step-1 documentation); original artifacts trace to `e99aa094384fcc5d23085d64a3bc2fc99c167f2f`. Prepared October 15, 2026.

## Implementation changes

- Replaced nine historical stage scripts and three analysis scripts with explicit CLI adapters. Shared implementation lives in `src/citation_audit/` and is importable without loading models or downloading resources.
- New-run key: `(run_id, source_path, prompt_id, variant)`. Null/duplicate identities and incomplete comparison pairs are refused before inference. Historical deduplication is an isolated command and rejects conflicting records rather than choosing one.
- Model generation is sequential and requests continuation-only text. Exact formatted inputs, answer-only text, correction context, and corrected answer text have separate fields. Stable model revisions are required; the chat-template protocol is newly specified.
- Before/after verification uses the same detector, index, threshold, and claim-level retrieval code. Every candidate is evaluated; partial support is visible. Source existence and entailment remain unknown.
- Correction verification persists stable IDs, answer hash, actual evaluated field, candidate evidence, and verifier configuration. Merge rejects duplicate/missing/unknown identities, stale/wrong-field judgments, changed configuration, and malformed labels. Uncorrected records remain in the population.
- Aggregation exposes missing/empty/cited categories and explicit denominators. Bootstrap samples paired prompt IDs and reports the observed effect separately from percentile uncertainty. Correction comparisons preserve both conditions within each sampled prompt. Undefined retrieval outcomes exclude complete prompt pairs with explicit counts.
- Corpus construction consumes a declared local JSONL, builds normalized `IndexFlatIP`, and saves hashes/model revision/passage ordering/dimensions. Retrieval checks index and passage hashes/counts/dimensions, finite vectors, zero vectors, fewer-than-k cases, and invalid result IDs. Inner product on unit vectors is interpreted as a cosine proxy.
- All commands reserve new output directories and expose incomplete/failed/completed status. Existing results cannot be appended or overwritten. No automatic resume is implemented. Failed intermediate outputs must not be interpreted as a completed run.

## Evidence and limitations

The included six-answer demo exercises the same generation-stage orchestration, actual FAISS construction/retrieval, verification, selection, correction, reverification, merge, summary, and bootstrap code as the model-mode path. Its generator returns canned answers and its encoder hashes tokens. It is exclusively software-validation evidence.

The optional Hugging Face generation and SentenceTransformer adapters have explicit contracts and immutable revisions. A bounded real-model CPU smoke run and a separate injected correction-adapter check completed successfully; see [model validation](model-smoke.md) for exact scope and limitations. No historical answer prefixes were guessed or removed. No new scientific rates or RAG effect claims are made.

The corpus builder does not magically reconstruct the missing historical FineWeb subset. A new research corpus requires an independently recorded acquisition/selection/chunking/license recipe. The detector's labeled fixtures cover narrow examples and regression cases, not production detector validity. Similarity matches are not reference resolution, factuality, or source entailment.

The historical artifacts and original paper remain byte-for-byte unchanged. Deduplicated derived records keep their old labels and must not be called revalidated outcomes. `docs/results-correction.md` continues to apply to those historical claims.

## Checks

The test suite covers the six-answer end-to-end path, individual CLI stages, actual FAISS behavior, exclusive outputs and failed status, unchanged historical input hashes, paired statistics, detector cases, no import-time side effects, and fake-model continuation-only generation. See `docs/validation.md` for the completed test counts and commands. Remote CI remains pending until this branch is pushed.

## References

- [Hugging Face pipeline API](https://huggingface.co/docs/transformers/main_classes/pipelines)
- [SentenceTransformer API](https://www.sbert.net/docs/package_reference/sentence_transformer/model.html)
- [FAISS metrics and cosine search](https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances)
