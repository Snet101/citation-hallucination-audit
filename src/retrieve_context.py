"""Compatibility entry point; use --help for explicit repaired inputs/outputs."""

import sys

from citation_audit.cli import main

if __name__ == "__main__":
    main(["retrieve"] + sys.argv[1:])
