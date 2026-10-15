import hashlib
import json
import runpy
import shutil
from pathlib import Path

import pytest


@pytest.mark.parametrize("representation", ["windows_archive", "git_blob"])
def test_preservation_accepts_only_recorded_original_bytes(tmp_path, representation):
    repo = Path(__file__).resolve().parents[1]
    function = runpy.run_path(str(repo / "scripts/check_preservation.py"))["check_preservation"]
    (tmp_path / "docs").mkdir()
    for name in ("baseline-manifest.json", "baseline-git-objects.json"):
        shutil.copyfile(repo / "docs" / name, tmp_path / "docs" / name)
    manifest = json.loads((repo / "docs/baseline-git-objects.json").read_text())
    for entry in manifest["files"]:
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
    assert function(tmp_path) == {"preserved_files": 8, "git_index_checked": False}
    damaged = tmp_path / manifest["files"][0]["path"]
    damaged.write_bytes(damaged.read_bytes()+b"changed")
    with pytest.raises(ValueError, match="Historical file changed"):
        function(tmp_path)
