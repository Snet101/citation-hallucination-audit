"""Assert the known software-fixture outcomes without claiming a model result."""

import json
import sys
from pathlib import Path

from citation_audit.core import key, read_jsonl

directory = Path(sys.argv[1])
status = json.loads((directory / "status.json").read_text(encoding="utf-8"))
assert status["state"] == "complete" and status["mode"] == "fixture"
assert status["claim_scope"] == "software fixture only"
before = read_jsonl(directory / "baseline-verification.jsonl")
corrected = read_jsonl(directory / "corrected-verification.jsonl")
final = read_jsonl(directory / "final-records.jsonl")
assert len(before) == len(final) == 6
assert len(corrected) == 2
assert {key(row["before"]) for row in final} == {key(row) for row in before}
assert {key(row["before"]) for row in final} == {key(row["after"]) for row in final}
assert all(row["evaluated_answer_field"] == "corrected_answer_text" for row in corrected)
assert (
    sum(
        row["before"]["any_retrieval_support"] != row["after"]["any_retrieval_support"]
        for row in final
    )
    == 1
)
assert (
    sum(
        row["before"]["all_candidate_retrieval_support"]
        != row["after"]["all_candidate_retrieval_support"]
        for row in final
    )
    == 2
)
assert sum(row["answer_status"] == "empty_answer" for row in before) == 1
assert sum(row["answer_status"] == "no_detected_citation" for row in before) == 1
assert all(
    item["source_exists"] is None and item["source_supports_claim"] is None
    for row in before
    for item in row["candidates"]
)
print(
    "PASS: six fixture answers, two fresh corrections, unchanged population; no source-truth claims"
)
