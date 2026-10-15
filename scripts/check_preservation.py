"""Verify historical artifact/PDF bytes; source code is intentionally repaired."""

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
    index_checked = (root / ".git").exists() and shutil.which("git") is not None
    for entry in manifest["files"]:
        if entry["windows_archive_sha256"] != archive_hashes[entry["path"]]:
            raise ValueError("Original archive fingerprint was changed")
        actual = hashlib.sha256((root / entry["path"]).read_bytes()).hexdigest()
        if actual not in {entry["windows_archive_sha256"], entry["git_blob_sha256"]}:
            raise ValueError(f"Historical file changed: {entry['path']}")
        if index_checked:
            record = subprocess.check_output(
                ["git", "-C", str(root), "ls-files", "--stage", "--", entry["path"]], text=True)
            if not record or record.split()[1] != entry["git_blob_oid"]:
                raise ValueError(f"Historical Git object changed: {entry['path']}")
    return {"preserved_files": 8, "git_index_checked": index_checked}


if __name__ == "__main__":
    report = check_preservation(Path(__file__).resolve().parents[1])
    print(f"PASS: eight historical files match original byte fingerprints; Git index checked={report['git_index_checked']}")
