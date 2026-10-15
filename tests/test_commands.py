import copy
import json
import runpy
from pathlib import Path

import pytest
from conftest import FIXTURES

from citation_audit.cli import main
from citation_audit.core import (
    correction_bootstrap,
    digest,
    evaluate,
    merge_evaluations,
    read_jsonl,
)


def test_all_component_commands_form_the_same_pipeline(tmp_path):
    def invoke(command, name, *args):
        out = tmp_path / name
        main([command, "--out", str(out), *map(str, args)])
        assert json.loads((out / "status.json").read_text())["state"] == "complete"
        return out

    index = (
        invoke("build-index", "index", "--input", FIXTURES / "corpus.jsonl", "--mode", "fixture")
        / "index"
    )
    inferred = (
        invoke(
            "infer",
            "infer",
            "--manifest",
            FIXTURES / "manifest.jsonl",
            "--run-id",
            "r",
            "--mode",
            "fixture",
            "--fixture-responses",
            FIXTURES / "responses.jsonl",
        )
        / "records.jsonl"
    )
    extracted = invoke("extract", "extract", "--input", inferred)
    assert len(read_jsonl(extracted / "records.jsonl")) == 6
    baseline = (
        invoke(
            "verify",
            "verify",
            "--input",
            inferred,
            "--index",
            index,
            "--mode",
            "fixture",
            "--threshold",
            "0.95",
        )
        / "records.jsonl"
    )
    selected = invoke("select", "select", "--input", baseline) / "records.jsonl"
    context = (
        invoke("retrieve", "retrieve", "--input", selected, "--index", index, "--mode", "fixture")
        / "records.jsonl"
    )
    corrected = (
        invoke(
            "correct",
            "correct",
            "--input",
            context,
            "--mode",
            "fixture",
            "--fixture-responses",
            FIXTURES / "responses.jsonl",
        )
        / "records.jsonl"
    )
    verified = (
        invoke(
            "verify",
            "reverify",
            "--input",
            corrected,
            "--index",
            index,
            "--mode",
            "fixture",
            "--answer-field",
            "corrected_answer_text",
            "--threshold",
            "0.95",
        )
        / "records.jsonl"
    )
    merged = (
        invoke(
            "merge",
            "merge",
            "--before",
            baseline,
            "--corrected",
            verified,
            "--expected-corrections",
            corrected,
        )
        / "records.jsonl"
    )
    summary = invoke("summarize", "summary", "--input", merged)
    assert sum(row["responses"] for row in json.loads((summary / "result.json").read_text())) == 6
    invoke("bootstrap", "bootstrap", "--input", baseline, "--B", "20")
    invoke("bootstrap", "bootstrap-correction", "--input", merged, "--correction", "--B", "20")


def test_plot_has_explicit_metric_and_saved_output(tmp_path, completions, retriever):
    pytest.importorskip("matplotlib")
    from citation_audit.core import summarize

    path = tmp_path / "summary.json"
    path.write_text(json.dumps(summarize(evaluate(completions, retriever))))
    out = tmp_path / "plot"
    main(["plot", "--input", str(path), "--out", str(out)])
    assert (out / "proxy-metrics.png").read_bytes().startswith(b"\x89PNG")


def test_historical_audit_preserves_originals(tmp_path):
    repo = tmp_path / "repo"
    artifacts = repo / "artifacts"
    artifacts.mkdir(parents=True)
    names = [
        "tinyllama_completions.jsonl",
        "verification_results.jsonl",
        "corrected_completions.jsonl",
    ]
    row = '{"source_path":"fixture","prompt_id":0,"variant":"neutral","response":"example"}\n'
    for name in names:
        (artifacts / name).write_text(row + row)
    hashes = {name: digest(artifacts / name) for name in names}
    main(["audit-history", "--repo", str(repo), "--out", str(tmp_path / "audit")])
    assert {name: digest(artifacts / name) for name in names} == hashes
    report = json.loads((tmp_path / "audit" / "result.json").read_text())
    assert all(value["unique_records"] == 1 for value in report["files"].values())


def test_entrypoints_have_no_download_training_or_output_side_effects(tmp_path, monkeypatch):
    repo = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(tmp_path)
    for path in list((repo / "src").glob("*.py")) + list((repo / "analysis").glob("*.py")):
        runpy.run_path(str(path))
    assert list(tmp_path.iterdir()) == []


def test_permuted_correction_records_cannot_change_pairing(completions, retriever):
    before = evaluate(completions, retriever)
    merged = merge_evaluations(before, [], [])
    modified = copy.deepcopy(merged)
    modified[0]["after"], modified[2]["after"] = modified[2]["after"], modified[0]["after"]
    with pytest.raises(ValueError, match="Misaligned"):
        correction_bootstrap(modified, B=20)


def test_too_small_run_refused_before_model_loading(tmp_path, monkeypatch):
    manifest = tmp_path / "one-pair.jsonl"
    manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in read_jsonl(FIXTURES / "manifest.jsonl")[:2])
    )

    def forbidden(args):
        raise AssertionError("Backend must not be loaded for invalid run")

    monkeypatch.setattr("citation_audit.cli.make_encoder", forbidden)
    with pytest.raises(SystemExit, match="at least two"):
        main(
            [
                "run",
                "--mode",
                "fixture",
                "--manifest",
                str(manifest),
                "--fixture-responses",
                str(FIXTURES / "responses.jsonl"),
                "--corpus",
                str(FIXTURES / "corpus.jsonl"),
                "--run-id",
                "small",
                "--out",
                str(tmp_path / "small"),
            ]
        )


def test_model_mode_requires_revisions_before_downloads(tmp_path):
    with pytest.raises(SystemExit, match="immutable"):
        main(
            [
                "infer",
                "--mode",
                "model",
                "--manifest",
                str(FIXTURES / "manifest.jsonl"),
                "--run-id",
                "model",
                "--out",
                str(tmp_path / "model"),
            ]
        )
    assert not (tmp_path / "model").exists()
