# Historical results review and baseline preservation

Recorded October 15, 2026. Baseline: `e99aa094384fcc5d23085d64a3bc2fc99c167f2f`.

## What changed in step 1

The README now distinguishes the intended experiment from the saved evidence and identifies the historical results as under review. The baseline source files, artifacts, and PDF remain unchanged. No new experimental result is claimed.

All 21 baseline tracked files were archived and their SHA-256 hashes checked against the archive. `docs/baseline-manifest.json` records paths, byte counts, and hashes. This manifest describes the baseline before these documentation edits; it is not a hash of the repaired README.

An external baseline ZIP and Git history bundle were also saved for recovery. Those backup files are separate deliverables, not additional datasets or a newly completed experiment.

## Findings and claim status

1. **Sample size:** both saved baseline JSONL files have 500 rows but 250 distinct full records, covering 125 prompt IDs in two variants. The corrected file has 78 rows but 39 distinct full records. Repeated identical rows are not independent model responses. Original additional runs, if any, have not been recovered.
2. **Correction merge:** repeated keys on both sides allow a many-to-many merge. The population must remain unchanged in a valid before/after comparison.
3. **Fresh correction evaluation:** `reverify_corrected.py` writes an aggregate without stable per-answer IDs, while `final_reverify.py` reads the original retained `supported` label from corrected generation records. The currently reported RAG improvement is not established by that pipeline.
4. **Answer boundary:** saved response fields contain input text and corrected response fields contain supplied evidence. Evaluation must isolate newly generated answers before interpreting results.
5. **Metric meaning:** missing citations, retrieval mismatch, fabricated sources, and unsupported claims are different outcomes. The existing complement-of-support calculation is not a validated citation-fabrication rate.
6. **Statistics:** the current bootstrap CLI reads aggregated condition rows. Uncertainty and condition comparisons require suitable prompt-level observations and paired sampling.
7. **Reproduction:** manifest coverage, script defaults, missing historical index/corpus recipe, inconsistent extraction, and absent environment specification remain unresolved.

The previous 97.6%/98.4% headline percentages and claimed sub-one-percentage-point RAG improvement are retained only in historical documents/artifacts. They should not be used as validated portfolio findings while these issues remain open. Deduplicating rows alone does not repair the label, boundary, metric, or statistical issues.

## Criteria before publishing replacement results

- Preserve historical artifacts with provenance; reject conflicting evaluation identities.
- Use stable unique keys through generation, correction, verification, and aggregation.
- Keep prompt, retrieved context, and generated answer in distinct fields.
- Persist fresh per-answer correction judgments and require complete correction verification.
- Enforce one-to-one merges and unchanged evaluation population/key set.
- Validate one shared citation detector and state the limits of retrieval-based judgments.
- Analyze paired prompt observations with explicit denominators and uncertainty assumptions.
- Record input/corpus/model revisions, configuration, dependencies, and run completeness.
- Execute meaningful offline fixtures and a newly documented bounded full run before making new result claims.

## Historical documents

### Stage-3 implementation update

The pipeline implementation has now been repaired and tested with offline fixtures; see [pipeline repair](pipeline-repair.md). This does not revalidate the historical labels or percentages. A newly documented real-model experiment remains outstanding. The step-1 preservation description above refers to that earlier documentation-only change, not to the later source-code repairs.

The PDF and generated CSV/JSONL summaries are preserved byte-for-byte for auditability. They are not silently rewritten to match the new description. A later erratum or updated paper must distinguish historical claims from repaired/new evidence.
