import json

import pytest
from conftest import FIXTURES

from citation_audit.core import (
    build_manifest,
    deduplicate_historical,
    extract_candidates,
    output_directory,
    read_jsonl,
    validate,
    write_json,
    write_jsonl,
)


@pytest.mark.parametrize("case", json.loads((FIXTURES / "citation_cases.json").read_text()))
def test_labeled_detector_cases(case):
    candidates = extract_candidates(case["text"])
    assert [candidate["kind"] for candidate in candidates] == case["expected_kinds"]
    for candidate in candidates:
        assert case["text"][candidate["start"] : candidate["end"]] == candidate["text"]


def test_id_normalization_rejects_duplicates(completions):
    first = {**completions[0], "prompt_id": 1}
    second = {**first, "prompt_id": "1"}
    with pytest.raises(ValueError, match="Duplicate"):
        validate([first, second])


def test_claim_boundaries_handle_lowercase_next_sentence_and_author_periods():
    candidates = extract_candidates("Birds build nests [3]. submarines fly to Jupiter [999].")
    assert candidates[0]["claim_text"] == "Birds build nests [3]."
    assert candidates[1]["claim_text"] == "submarines fly to Jupiter [999]."
    text = "A claim (Smith et al. 2020). Another claim [1]."
    assert extract_candidates(text)[0]["claim_text"] == "A claim (Smith et al. 2020)."


@pytest.mark.parametrize("value", [None, "", " ", True, float("nan")])
def test_invalid_key_rejected(completions, value):
    with pytest.raises((ValueError, KeyError)):
        validate([{**completions[0], "run_id": value}])


def test_missing_pair_rejected(completions):
    with pytest.raises(ValueError, match="pairs"):
        validate(completions[1:], paired=True)


def test_historical_duplicates_and_conflicts(completions):
    first = completions[0]
    unique, removed = deduplicate_historical([first, first])
    assert len(unique) == 1 and removed == 1
    with pytest.raises(ValueError, match="Conflicting"):
        deduplicate_historical([first, {**first, "answer_text": "different"}])


def test_output_is_exclusive_and_failed_run_is_visible(tmp_path):
    path = tmp_path / "run"
    with pytest.raises(RuntimeError), output_directory(path, {"mode": "fixture"}):
        raise RuntimeError("fake model failure")
    assert json.loads((path / "status.json").read_text())["state"] == "failed"
    with pytest.raises(FileExistsError), output_directory(path, {}):
        pass


def test_json_writes_do_not_overwrite_or_serialize_nonfinite(tmp_path):
    path = tmp_path / "rows.jsonl"
    write_jsonl(path, [{"value": 1}])
    with pytest.raises(FileExistsError):
        write_jsonl(path, [])
    other = tmp_path / "bad.jsonl"
    with pytest.raises(ValueError):
        write_jsonl(other, [{"value": float("nan")}])
    assert not other.exists()
    with pytest.raises(FileExistsError):
        write_json(path, {})


@pytest.mark.parametrize("text", ["{}\n\n", "{broken}\n", "[1]\n", '{"x":NaN}\n'])
def test_malformed_input_rejected(tmp_path, text):
    path = tmp_path / "bad.jsonl"
    path.write_text(text)
    with pytest.raises(ValueError):
        read_jsonl(path)


def test_manifest_is_paired_reproducible_and_does_not_invent_rows(tmp_path):
    path = tmp_path / "input.tsv"
    path.write_text("data_type\tvanilla\nbenign\tQuestion A?\nbenign\tQuestion B?\n")
    first = build_manifest(path, ["benign"], 2, 42)
    assert first == build_manifest(path, ["benign"], 2, 42)
    assert len(first) == 4
    validate([{**row, "run_id": "r"} for row in first], paired=True)
    with pytest.raises(ValueError, match="Insufficient"):
        build_manifest(path, ["benign"], 3, 42)
    path.write_text("data_type\tvanilla\nbenign\tQuestion A?\nbenign\tQuestion A?\n")
    with pytest.raises(ValueError, match="Duplicate source"):
        build_manifest(path, ["benign"], 1, 42)
