# Real-model integration smoke checks

Completed October 15, 2026 on Windows/Python 3.12.14, CPU. Dependency consistency passed in the separate optional-model environment. `requirements-model.lock.txt` records its installed versions; the exact lock was not reinstalled into a second independent environment.

## Pinned models

- TinyLlama/TinyLlama-1.1B-Chat-v1.0: `fe8a4ea1ffedaf415f4da2f062534de366a451e6`.
- sentence-transformers/all-MiniLM-L6-v2: `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`.
- Torch 2.14.1, Transformers 5.20.0, Sentence Transformers 5.7.0, FAISS CPU 1.15.1, NumPy 2.5.4.
- Greedy answer-only generation, 64 new-token cap, CPU, actual model dtype bfloat16, model context window 2048. The embedding model runs on CPU with its actual sequence limit recorded in index metadata.

The source commit plus dirty-tree flag, package-source hashes, input hashes, model revisions, chat-template hash, package versions, effective generation settings, and timestamps were saved with the local runs. These checks were performed before committing the repair, so the recorded package hashes are the relevant source fingerprints.

## Check A: four real baseline generations

`examples/benign-smoke-manifest.jsonl` contains two elementary questions, each in neutral and citation-requested variants. The corpus is the three original synthetic passages in `tests/fixtures/corpus.jsonl`, not FineWeb.

```text
python -m citation_audit run --mode model --manifest examples/benign-smoke-manifest.jsonl --corpus tests/fixtures/corpus.jsonl --run-id benign-model-smoke --out runs/benign-model-smoke --model-revision fe8a4ea1ffedaf415f4da2f062534de366a451e6 --embedding-revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 --max-new-tokens 64 --device -1 --threshold 0.95 --B 1000
```

The process exited successfully with four generated answers and four final records, complete keys, and separate formatted-input/answer fields. No answer contained the full formatted model input. No structured citation candidates were detected in this tiny sample, so no responses were selected for correction. This checks the zero-correction path and preserves missing-citation outcomes rather than assigning a fabrication label.

The token budget is deliberately small, outputs were not assessed as research-quality answers, and the detector has limited format coverage. Neither absence of detected citations nor the produced bootstrap JSON establishes anything general about model capability or truthfulness.

## Check B: real correction and fresh reverification

Because check A did not enter the correction path, `examples/injected-correction-smoke.jsonl` supplies one **explicitly synthetic** input: a deliberately wrong software-test answer and an elementary evidence passage. It is not a generated baseline or an independent research observation.

The `correct` command used the same pinned TinyLlama checkpoint and 64-token cap. The `verify` command loaded check A's hashed MiniLM/FAISS index and evaluated `corrected_answer_text` at the same threshold. Both processes exited successfully. One correction and one fresh per-answer verification record were saved. Its answer hash matched the corrected text, and source existence/claim-support fields remained unknown.

```text
python -m citation_audit correct --mode model --input examples/injected-correction-smoke.jsonl --out runs/real-correction-adapter-smoke --model-revision fe8a4ea1ffedaf415f4da2f062534de366a451e6 --max-new-tokens 64 --device -1
python -m citation_audit verify --mode model --input runs/real-correction-adapter-smoke/records.jsonl --out runs/real-correction-verification-smoke --index runs/benign-model-smoke/index --embedding-revision 1110a243fdf4706b3f48f1d95db1a4f5529b4d41 --answer-field corrected_answer_text --threshold 0.95
```

Use new output directories if rerunning. The recorded correction contained two candidates; output text/candidate counts need not be bitwise identical on every backend. This check does not merge the injected input into check A's four-answer population.

The model introduced a source URL not present in the provided evidence. It was logged as a candidate, not assumed to be real or claim-supporting. This demonstrates why successful integration and retrieval similarity cannot establish citation validity or correction quality. No RAG-improvement percentage is inferred from this injected example.

Nonfatal library warnings included a deprecated embedding-dimension method alias and a model-default maximum-length setting overridden by the explicit new-token cap. They did not prevent the successful checks. No credentials were required; public model downloads were used.

## Remaining research work

Reconstruct or newly specify a licensed corpus and sampling protocol, validate the detector and similarity threshold against reviewed labels, inspect source existence and entailment independently, choose/record generation stopping and token-budget rules, and run an appropriately sized paired experiment. Historical results remain under review. Remote Linux CI is still pending until this repair branch is pushed.
