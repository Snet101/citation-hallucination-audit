import hashlib
import json
import runpy
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.parametrize("representation", ["windows_archive", "git_blob"])
def test_preservation_accepts_only_recorded_original_bytes(tmp_path, representation):
    repo = Path(__file__).resolve().parents[1]
    function = runpy.run_path(str(repo / "scripts/check_preservation.py"))["check_preservation"]
    (tmp_path / "docs").mkdir()
    for name in ("baseline-manifest.json", "baseline-git-objects.json", "retired-artifacts.json"):
        shutil.copyfile(repo / "docs" / name, tmp_path / "docs" / name)
    manifest = json.loads((repo / "docs/baseline-git-objects.json").read_text())
    for entry in manifest["files"]:
        if entry["path"] == "citation_hallucination_audit.pdf":
            continue
        target = tmp_path / entry["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        original = (repo / entry["path"]).read_bytes()
        # Restore only temporary test bytes with an exact recorded fingerprint.
        # Source ZIPs need no Git; no project artifact is rewritten.
        lf = original.replace(b"\r\n", b"\n")
        variants = {hashlib.sha256(data).hexdigest(): data
                    for data in (original, lf, lf.replace(b"\n", b"\r\n"))}
        data = variants[entry[representation + "_sha256"]]
        target.write_bytes(data)
    assert function(tmp_path) == {
        "preserved_files": 7, "retired_files": 1, "git_index_checked": False}
    damaged = tmp_path / manifest["files"][0]["path"]
    damaged.write_bytes(damaged.read_bytes()+b"changed")
    with pytest.raises(ValueError, match="Historical file changed"):
        function(tmp_path)


@pytest.fixture
def retired_source(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    (tmp_path / "docs").mkdir()
    for name in ("baseline-manifest.json", "baseline-git-objects.json", "retired-artifacts.json"):
        shutil.copyfile(repo / "docs" / name, tmp_path / "docs" / name)
    manifest = json.loads((repo / "docs/baseline-git-objects.json").read_text())
    for entry in manifest["files"]:
        if entry["path"] == "citation_hallucination_audit.pdf":
            continue
        target = tmp_path / entry["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo / entry["path"], target)
    function = runpy.run_path(str(repo / "scripts/check_preservation.py"))["check_preservation"]
    return tmp_path, function


def test_retirement_does_not_allow_other_missing_artifacts(retired_source):
    root, check = retired_source
    (root / "artifacts/prompts_manifest.csv").unlink()
    with pytest.raises(ValueError, match="Historical file missing"):
        check(root)


def test_retired_paper_cannot_silently_return(retired_source):
    root, check = retired_source
    (root / "citation_hallucination_audit.pdf").write_bytes(b"not a current deliverable")
    with pytest.raises(ValueError, match="Retired paper must remain absent"):
        check(root)


def test_retired_paper_cannot_remain_staged(retired_source):
    root, check = retired_source
    git = shutil.which("git")
    if git is None:
        pytest.skip("Git is unavailable")
    subprocess.run([git, "-C", str(root), "init", "-q"], check=True)
    subprocess.run([git, "-C", str(root), "config", "core.autocrlf", "false"], check=True)
    manifest = json.loads((root / "docs/baseline-git-objects.json").read_text())
    for entry in manifest["files"]:
        if entry["path"] == "citation_hallucination_audit.pdf":
            continue
        target = root / entry["path"]
        data = target.read_bytes()
        lf = data.replace(b"\r\n", b"\n")
        variants = {hashlib.sha256(value).hexdigest(): value
                    for value in (data, lf, lf.replace(b"\n", b"\r\n"))}
        target.write_bytes(variants[entry["git_blob_sha256"]])
    paper = root / "citation_hallucination_audit.pdf"
    paper.write_bytes(b"retired document staged by mistake")
    subprocess.run([git, "-C", str(root), "add", "."], check=True)
    paper.unlink()
    with pytest.raises(ValueError, match="Retired paper must remain absent from the Git index"):
        check(root)


@pytest.mark.parametrize("mutation", ["other_path", "duplicate", "fingerprint", "blank_reason"])
def test_retirement_policy_cannot_exempt_other_files(retired_source, mutation):
    root, check = retired_source
    path = root / "docs/retired-artifacts.json"
    policy = json.loads(path.read_text())
    entry = policy["retired_files"][0]
    if mutation == "other_path":
        entry["path"] = "artifacts/prompts_manifest.csv"
    elif mutation == "duplicate":
        policy["retired_files"].append(entry.copy())
    elif mutation == "fingerprint":
        entry["git_blob_sha256"] = "0" * 64
    else:
        entry["reason"] = " "
    path.write_text(json.dumps(policy))
    with pytest.raises(ValueError, match="Unexpected retirement policy|Retired paper fingerprint"):
        check(root)
