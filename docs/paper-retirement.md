# Historical paper retirement

The author intentionally removed `citation_hallucination_audit.pdf` in commit
`f58290b17888a539de56e228c81c0160fe99b9b1`. Its conclusions are not validated,
and it is no longer distributed as a current project deliverable. This revision
does not restore the PDF or rewrite Git history.

The original baseline manifests remain unchanged, including their recorded PDF
fingerprint. `retired-artifacts.json` records this one intentional retirement.
The preservation check requires the PDF to remain absent from both the working
tree and Git index and verifies its recorded identity against the baseline.
The seven retained historical result files still require their exact recorded
fingerprints and Git object identities. Other missing or changed artifacts fail.

The previous check expected all eight historical files to remain present, so the
intentional deletion broke both preservation test representations. New tests
cover the retirement, reject other missing artifacts, reject a reintroduced PDF,
and reject policy changes that attempt to exempt another file or alter the
recorded fingerprint.

Run `python -m pytest -q`, `python -m ruff check .`, and
`python scripts/check_preservation.py`. A successful preservation check now
reports seven retained files and one explicitly retired paper.
