# Follow-up repair: October 18, 2026

Baseline source commit: `a2b46dd76e4858225d9013b56a88e1dea55d3d3f`. This repair is a separate source snapshot prepared with AI assistance after the 30-answer Python-documentation pilot. The historical paper, artifacts and completed pilot are preserved. No new research improvement is claimed.

## Demonstrated failures and resulting behavior

| Failure | Repair | Boundary |
|---|---|---|
| CSV bibliography URLs were embedded instead of the reliability assertion. | Separate reference entries stay unresolved unless an explicit numbered inline marker links them to a claim. Search each linked claim after removing citation markers. | Conservative parser; ambiguous/multi-number links can remain unresolved. No semantic source-support verification. |
| One supporting fragment could mask another failing claim. | Persist individual claim judgments; every associated claim must pass for an all-claims retrieval match. | A similarity match still is not entailment. |
| Unknown associations could be confused with failures or omitted silently. | Three-valued proxy labels, unresolved/evaluable counts, null full-population rates when incomplete. | Evaluable-only rates can be selective and require their explicit denominator. |
| Code indexes such as `[1]` could be counted as citations. | Ignore candidates inside fenced and inline code. | Markdown/code detection remains heuristic. |
| 24 pilot passages exceeded MiniLM's 256-token limit. | Split original character spans until the actual tokenizer count including special tokens fits; save parent ID, offsets and token count. Reject oversized query text explicitly. | Does not infer good semantic segmentation, fix missing corpus coverage, or calibrate retrieval. |
| A role-marker correction replaced a cited answer with a nonresponsive question. | Preserve the attempted answer and fresh judgment, reject it from promotion, and keep the original observation in the main comparison. | Heuristic guards cannot identify every well-formed wrong or off-topic answer. Keeping the original is an audit policy, not an endorsement. |
| Token-cap artifacts were not visible during generation. | Record input and re-tokenized output lengths, budget and near-cap diagnostics; reject near-cap correction promotion and reject context overflow before inference. | Re-tokenized counts are diagnostics, not an exact generation stop reason. |
| Missing requested citations never entered correction. | Explicit `--correct-missing-citations` opt-in with recorded selection reason for the requested condition. | Missing citations are not classified as fabrications; neutral uncited answers are not automatically selected. |

## Output and schema changes

Detector version is `structured-candidates-v2-conservative-association`. Each candidate includes `association_status`, `association_method`, `claim_spans`, nullable `claim_text`, and individual `claim_evaluations` with saved retrieval queries/evidence. Unresolved candidates have no retrieval query, no evidence and a null retrieval match. Source existence and source-support fields remain null. Old judgments are rejected rather than silently migrated.

Bibliography association is intentionally narrow: explicit single-number `[1]` definitions can be linked to matching substantive inline claims. Multiple claims are evaluated individually. A free-standing unnumbered reference list cannot establish which claim a source supports, so the parser does not guess. The parser also does not implement complete scholarly citation or Markdown syntax.

Model encoder metadata identifies the token-budget policy. Index metadata records original passage count and chunking policy. Character offsets are within the parent input passage; original source and other provenance fields are preserved. New chunk IDs are checked for collisions. Old indexes fail the metadata check and should be rebuilt from retained inputs.

Merged records retain `before`, `after` and `attempted_after`. `after` changes only when the attempted output passes the explicitly limited guards. Counts distinguish attempted, accepted and rejected corrections. Attempts remain inspectable even when rejected. A normal-looking inaccurate answer can still pass these guards, and a legitimate abstention may be retained as an attempt rather than promoted; do not equate acceptance with scientific success.

Correction instructions use a separate system role in the Hugging Face adapter and put the original question before evidence. This is a prompt protocol change, not a guarantee that the model follows instructions. Exact formatted input and original generated text remain saved, including defects.

## Validation plan

1. Retain the original 79 tests and add regressions for the saved reference-list failure, explicit numbered association, multiple linked claims, code spans, unknown denominators, failed correction retention, token budgets, context limits and system-role formatting.
2. Rebuild the actual pilot corpus with pinned MiniLM and verify zero over-budget chunks. Replay all 30 original answers and the saved failed correction through v2 without regenerating or changing their text. Revalidate identity/population invariants.
3. Perform one separately labeled real TinyLlama correction integration check on the previously failed CSV question, with the same pinned model and explicitly recorded int8 configuration and a 256-token budget. Record its actual disposition, including failure if applicable.

These are regression/integration checks on inspected data, not independent scientific validation. Any next experiment needs a newly declared protocol, reviewed labels, threshold calibration and source/claim-support evaluation. Sanvi should review and explain the changes before describing them as personal understanding in an application.

## Completed validation

All **96 tests passed**, including all original 79 cases; Ruff lint passed. The real MiniLM rebuild converted 115 parent passages to **143 token-safe chunks**, with zero above the configured token budget. Replaying all 30 saved answers left the eight unlinked reference candidates unresolved and preserved all 30 observation identities. The saved failed correction was rejected, retained in `attempted_after`, and could not remove its original answer from the cited denominator. Default selection produced three unresolved-reference answers; explicit missing-citation opt-in selected sixteen answers, without assigning fabrication labels.

The one new real TinyLlama correction check generated a near-cap list of passage IDs rather than a useful answer. It was rejected for missing/lost requested citations and possible truncation. All 30 originals remained in the main comparison. This validates the guard on a real failure; it does not demonstrate improved generation quality or a RAG effect. The supplied validation evidence retains the full text, new formatted prompt, diagnostics, fresh judgment and rejection reasons.
