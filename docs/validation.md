# Stage-3 validation — October 15, 2026

Baseline: merged documentation commit `cec0a5d488419c214578cfbadd658b2bc465a274`.

## Offline repair checks

Tested in a new Windows/Python 3.12.14 environment. Exact offline versions are in `requirements.lock.txt`; optional model packages are installed separately.

The full suite also runs in the separate optional-model environment after the real-model smoke checks. The saved smoke-run package-source hashes were compared with the final package files and matched exactly. Two additional cross-platform preservation cases bring the final suite to 79 tests.

| Check | Result |
|---|---|
| `python -m pytest -q` | 79 tests passed |
| `python -m ruff check .` | Passed |
| Dependency consistency | Passed: no broken requirements in the offline environment |
| Full fixture run | Six answers, two fresh corrections, unchanged six-answer population |
| Individual stage commands | Index, inference, extraction, verification, selection, retrieval, correction, reverification, merge, summary, bootstrap, plotting tested |
| Historical preservation | All seven artifact files and original PDF match baseline SHA-256 hashes |
| Historical derivation | 500→250, 500→250, and 78→39 exact-record deduplication counts reproduced |
| Failure handling | Existing outputs, duplicates/conflicts, missing correction judgments, stale hashes, wrong fields/configuration, malformed JSON, invalid vectors/index metadata, and too-small pairs rejected |

The 24 labeled detector examples include ordinary “by reading” text, structured positive formats, nested DOI/URL detection, punctuation, and multiple candidates. Additional tests cover lowercase sentence boundaries and author-year abbreviations. These fixtures establish software behavior on the examples, not natural-data detector accuracy.

The original Windows archive hashes and Git-object hashes differ only in line endings for three historical files. Both exact representations are recorded in `baseline-git-objects.json`; the original `baseline-manifest.json` is unchanged. Preservation checks accept only those known bytes and require the indexed historical Git objects to remain unchanged. Tests exercise the Windows working-copy representation and the original Git blobs used by a Linux checkout. No artifact values or paper bytes were rewritten.

Statistics checks include reproducible paired resampling, an independently specified correction effect, explicit undefined-pair exclusions, and preservation of both variants within each prompt cluster. They do not calibrate detector errors or establish a real-world RAG effect.

## Scope

The fixture generator emits canned answers and the encoder hashes tokens; FAISS indexing/search itself is real. Fixture reference markers are synthetic. No fixture summary is a scientific model result.

The original labels/paper have not been revalidated. The exact historical FineWeb corpus/index remains missing. Remote Linux GitHub Actions cannot be reported as passed until the repair branch is pushed and checked.

Real-model integration validation completed successfully and is recorded in [model-smoke.md](model-smoke.md). It exercised adapters on bounded benign inputs; it cannot reproduce the historical experiment or establish a statistically useful model comparison.
