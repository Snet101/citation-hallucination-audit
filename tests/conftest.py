from pathlib import Path

import pytest

from citation_audit.core import read_jsonl
from citation_audit.generation import FixtureGenerator, infer
from citation_audit.retrieval import FixtureEncoder, build_index

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def retriever(tmp_path):
    return build_index(FIXTURES / "corpus.jsonl", tmp_path / "index", FixtureEncoder())


@pytest.fixture
def completions():
    generator = FixtureGenerator(read_jsonl(FIXTURES / "responses.jsonl"))
    return infer(read_jsonl(FIXTURES / "manifest.jsonl"), generator, "test-run")
