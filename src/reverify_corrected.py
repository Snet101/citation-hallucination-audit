"""Compatibility entry point; use --help for explicit repaired inputs/outputs."""

import sys

from citation_audit.cli import main

if __name__ == "__main__":
    main(["verify", "--answer-field", "corrected_answer_text"] + sys.argv[1:])
