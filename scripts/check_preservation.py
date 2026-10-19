"""Verify retained historical bytes and the explicitly retired paper's absence."""

import hashlib
import json
import shutil
import subprocess
from pathlib import Path


def check_preservation(root):
    root = Path(root)
    original = json.loads((root / "docs/baseline-manifest.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "docs/baseline-git-objects.json").read_text(encoding="utf-8"))
    archive_hashes = {entry["path"]: entry["sha256"] for entry in original["files"]}
    expected_paths = {entry["path"] for entry in original["files"]
                      if entry["path"].startswith("artifacts/") or entry["path"].endswith(".pdf")}
    if (len(manifest["files"]) != 8 or {entry["path"] for entry in manifest["files"]} != expected_paths
            or manifest["baseline_commit"] != original["baseline_commit"]):
        raise ValueError("Unexpected historical artifact inventory or baseline")
    policy = json.loads((root / "docs/retired-artifacts.json").read_text(encoding="utf-8"))
    retired = policy["retired_files"]
    # This is a deliberate exception for one retired document, not a general
    # missing-file allowance or a way to exempt historical result artifacts.
    paper = "citation_hallucination_audit.pdf"
    if len(retired) != 1 or retired[0]["path"] != paper or not retired[0]["reason"].strip():
        raise ValueError("Unexpected retirement policy")
    recorded_paper = next(entry for entry in manifest["files"] if entry["path"] == paper)
    for field in ("git_blob_oid", "git_blob_sha256"):
        if retired[0][field] != recorded_paper[field]:
            raise ValueError("Retired paper fingerprint differs from the historical baseline")
    index_checked = (root / ".git").exists() and shutil.which("git") is not None
    for entry in manifest["files"]:
        if entry["windows_archive_sha256"] != archive_hashes[entry["path"]]:
            raise ValueError("Original archive fingerprint was changed")
        if entry["path"] == paper:
            if (root / paper).exists():
                raise ValueError("Retired paper must remain absent from the working tree")
            if index_checked:
                record = subprocess.check_output(
                    ["git", "-C", str(root), "ls-files", "--stage", "--", paper], text=True)
                if record:
                    raise ValueError("Retired paper must remain absent from the Git index")
            continue
        if not (root / entry["path"]).is_file():
            raise ValueError(f"Historical file missing: {entry['path']}")
        actual = hashlib.sha256((root / entry["path"]).read_bytes()).hexdigest()
        if actual not in {entry["windows_archive_sha256"], entry["git_blob_sha256"]}:
            raise ValueError(f"Historical file changed: {entry['path']}")
        if index_checked:
            record = subprocess.check_output(
                ["git", "-C", str(root), "ls-files", "--stage", "--", entry["path"]], text=True)
            if not record or record.split()[1] != entry["git_blob_oid"]:
                raise ValueError(f"Historical Git object changed: {entry['path']}")
    return {"preserved_files": 7, "retired_files": 1, "git_index_checked": index_checked}


if __name__ == "__main__":
    report = check_preservation(Path(__file__).resolve().parents[1])
    print("PASS: seven historical files match original byte fingerprints; "
          f"one paper explicitly retired; Git index checked={report['git_index_checked']}")
