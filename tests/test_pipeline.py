import copy
import json

import pytest
from conftest import FIXTURES

from citation_audit.cli import main
from citation_audit.core import (
    correction_bootstrap,
    evaluate,
    key,
    merge_evaluations,
    paired_bootstrap,
    read_jsonl,
    select_corrections,
    summarize,
    transitions,
)
from citation_audit.generation import FixtureGenerator, attach_context, correct


def prepare(completions, retriever):
    before = evaluate(completions, retriever, 0.95)
    generated = correct(
        attach_context(select_corrections(before), retriever),
        FixtureGenerator(read_jsonl(FIXTURES / "responses.jsonl")),
    )
    judgments = evaluate(generated, retriever, 0.95, "corrected_answer_text")
    return before, generated, judgments


def test_real_faiss_fixture_pipeline_preserves_population(completions, retriever):
    before, generated, judgments = prepare(completions, retriever)
    assert len(generated) == 2
    merged = merge_evaluations(before, judgments, generated)
    assert len(merged) == len(before) == 6
    assert {tuple(row["key"].values()) for row in merged} == {key(row) for row in before}
    assert (
        sum(
            row["before"]["any_retrieval_support"] != row["after"]["any_retrieval_support"]
            for row in merged
        )
        == 1
    )
    partial = next(
        row for row in before if row["prompt_id"] == "p2" and row["variant"] == "neutral"
    )
    assert partial["any_retrieval_support"] is True
    assert partial["all_candidate_retrieval_support"] is False
    assert all(item["source_exists"] is None for row in before for item in row["candidates"])
    summaries = summarize(before)
    assert sum(row["responses"] for row in summaries) == 6
    assert sum(row["empty_answers"] for row in summaries) == 1
    assert sum(row["no_detected_citation"] for row in summaries) == 1
    changes = transitions(merged)["proxy_transitions"]
    assert changes["any_retrieval_support"]["false -> true"] == 1
    assert changes["all_candidate_retrieval_support"]["false -> true"] == 2
    assert changes["any_retrieval_support"]["not_applicable -> not_applicable"] == 2


def test_prompt_and_context_cannot_be_evaluated_as_answer(completions, retriever):
    row = {
        **completions[0],
        "prompt_text": "Plants use sunlight to make sugar [1].",
        "model_input_text": "Plants use sunlight to make sugar [1].",
        "retrieved_passages": [{"text": "Plants use sunlight to make sugar [1]."}],
        "answer_text": "I cannot verify this.",
    }
    result = evaluate([row], retriever)[0]
    assert result["answer_status"] == "no_detected_citation"
    assert result["any_retrieval_support"] is None and result["abstention_cue"]


@pytest.mark.parametrize(
    "mutation",
    ["duplicate_before", "duplicate_after", "missing", "unknown", "stale", "wrong_field", "config"],
)
def test_merge_rejects_corruption(completions, retriever, mutation):
    before, generated, judgments = prepare(completions, retriever)
    if mutation == "duplicate_before":
        before.append(before[0])
    elif mutation == "duplicate_after":
        judgments.append(judgments[0])
    elif mutation == "missing":
        judgments.pop()
    elif mutation == "unknown":
        judgments[0] = {**judgments[0], "prompt_id": "unknown"}
        generated[0] = {**generated[0], "prompt_id": "unknown"}
    elif mutation == "stale":
        judgments[0]["answer_sha256"] = "not-the-new-answer"
    elif mutation == "wrong_field":
        judgments[0]["evaluated_answer_field"] = "answer_text"
    else:
        judgments[0]["threshold"] = 0.1
    with pytest.raises(ValueError):
        merge_evaluations(before, judgments, generated)


def test_no_corrections_still_preserves_all_records(completions, retriever):
    before = evaluate(completions, retriever)
    merged = merge_evaluations(before, [], [])
    assert len(merged) == 6
    assert all(row["before"] == row["after"] and not row["was_corrected"] for row in merged)


def test_paired_bootstrap_uses_observed_effect_and_reports_undefined(completions, retriever):
    rows = evaluate(completions, retriever)
    first = paired_bootstrap(rows, B=100, seed=1)
    assert first == paired_bootstrap(rows, B=100, seed=1)
    assert first == paired_bootstrap(list(reversed(rows)), B=100, seed=1)
    # Cited: p0 1-1, p1 1-0, p2 0-1.
    assert first["estimate"] == 0 and first["n_prompt_pairs"] == 3
    with pytest.raises(ValueError, match="pairs"):
        paired_bootstrap(rows[1:])
    changed = copy.deepcopy(rows)
    for row in changed:
        row["any_retrieval_support"] = row["variant"] == "with_citation"
    changed[0]["any_retrieval_support"] = None
    result = paired_bootstrap(changed, "any_retrieval_support", B=100)
    assert result["estimate"] == 1
    assert result["excluded_undefined_pairs"] == 1 and result["n_prompt_pairs"] == 2
    with pytest.raises(ValueError):
        paired_bootstrap(rows + [rows[0]])


def test_correction_bootstrap_retains_unchanged_population(completions, retriever):
    before, generated, judgments = prepare(completions, retriever)
    merged = merge_evaluations(before, judgments, generated)
    result = correction_bootstrap(merged, B=100)
    assert result["estimate"] == 0 and result["n_prompt_pairs"] == 3
    assert result["excluded_undefined_pairs"] == 0


def test_cli_offline_end_to_end_and_refuses_rerun(tmp_path):
    out = tmp_path / "demo"
    args = [
        "run",
        "--mode",
        "fixture",
        "--manifest",
        str(FIXTURES / "manifest.jsonl"),
        "--fixture-responses",
        str(FIXTURES / "responses.jsonl"),
        "--corpus",
        str(FIXTURES / "corpus.jsonl"),
        "--run-id",
        "fixture-run",
        "--out",
        str(out),
        "--threshold",
        "0.95",
        "--B",
        "100",
    ]
    main(args)
    assert json.loads((out / "status.json").read_text())["state"] == "complete"
    assert len(read_jsonl(out / "final-records.jsonl")) == 6
    assert json.loads((out / "result.json").read_text())["corrected"] == 2
    with pytest.raises(SystemExit, match="Refused"):
        main(args)


def test_failed_inference_does_not_publish_completed_results(tmp_path):
    out = tmp_path / "failed"
    bad = tmp_path / "responses.jsonl"
    bad.write_text('{"prompt_id":"missing","variant":"neutral","answer_text":"fake"}\n')
    with pytest.raises(SystemExit):
        main(
            [
                "run",
                "--mode",
                "fixture",
                "--manifest",
                str(FIXTURES / "manifest.jsonl"),
                "--fixture-responses",
                str(bad),
                "--corpus",
                str(FIXTURES / "corpus.jsonl"),
                "--run-id",
                "failed-run",
                "--out",
                str(out),
            ]
        )
    assert json.loads((out / "status.json").read_text())["state"] == "failed"
    assert not (out / "final-records.jsonl").exists()


def test_no_citation_run_has_null_support_and_no_selected_corrections(tmp_path):
    responses = read_jsonl(FIXTURES / "responses.jsonl")
    for row in responses:
        row["answer_text"] = "I cannot verify this."
    path = tmp_path / "abstentions.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in responses))
    out = tmp_path / "no-citations"
    main(
        [
            "run",
            "--mode",
            "fixture",
            "--manifest",
            str(FIXTURES / "manifest.jsonl"),
            "--fixture-responses",
            str(path),
            "--corpus",
            str(FIXTURES / "corpus.jsonl"),
            "--run-id",
            "no-citations",
            "--out",
            str(out),
            "--B",
            "20",
        ]
    )
    assert read_jsonl(out / "corrected-verification.jsonl") == []
    summary = json.loads((out / "summary-after.json").read_text())
    assert all(row["any_retrieval_support_rate_among_cited"] is None for row in summary)
    assert all(row["citation_present_rate_all_answers"] == 0 for row in summary)


def test_known_correction_effect_resamples_prompt_clusters(completions):
    toy = []
    for row in completions:
        toy.append(
            {
                "before": {**row, "citation_present": False},
                "after": {**row, "citation_present": row["variant"] == "neutral"},
            }
        )
    result = correction_bootstrap(toy, B=100, seed=42)
    assert result["estimate"] == result["ci_low"] == result["ci_high"] == 0.5
    assert result["n_prompt_pairs"] == 3
    with pytest.raises(ValueError):
        correction_bootstrap(toy, B=0)
