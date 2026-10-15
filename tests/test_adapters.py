import json

import numpy as np
import pytest

from citation_audit.core import digest
from citation_audit.generation import HuggingFaceGenerator
from citation_audit.retrieval import FixtureEncoder, IndexRetriever, normalized


def test_generation_requests_only_continuation_and_keeps_exact_input():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs == {"tokenize": False, "add_generation_prompt": True}
            return f"INPUT {messages[0]['content']} ANSWER"

    class Pipeline:
        tokenizer = Tokenizer()

        def __call__(self, model_input, **kwargs):
            assert kwargs["return_full_text"] is False
            assert kwargs["do_sample"] is False
            assert kwargs["add_special_tokens"] is False
            return [{"generated_text": "continuation only"}]

    generator = HuggingFaceGenerator("fake", "a" * 40, pipeline_object=Pipeline())
    model_input, answer = generator.generate("question [1]", {})
    assert model_input == "INPUT question [1] ANSWER"
    assert answer == "continuation only" and model_input not in answer


def test_mutable_model_revision_refused_before_loading():
    with pytest.raises(ValueError, match="immutable"):
        HuggingFaceGenerator("fake", "main")


@pytest.mark.parametrize(
    "values", [[[0, 0]], [[float("nan"), 1]], [[float("inf"), 1]], [[1, 2, 3]]]
)
def test_invalid_vectors_rejected(values):
    with pytest.raises(ValueError):
        normalized(values, 2)


def test_faiss_handles_fewer_than_k_passages(retriever):
    results = retriever.search("Plants use sunlight to make sugar [1].", k=100)
    assert len(results) == 3
    assert results[0]["passage_id"] == "plants"
    assert results[0]["score"] == pytest.approx(1)


def test_invalid_faiss_ids_are_not_python_negative_indices(retriever):
    class FakeIndex:
        d, ntotal = 128, 3

        def search(self, vector, k):
            return np.array([[0.9, 0.8, 0.7]]), np.array([[0, -1, 999]])

    retriever.index = FakeIndex()
    assert [item["passage_id"] for item in retriever.search("plant")] == ["plants"]


def test_tampered_passage_file_rejected(retriever, tmp_path):
    path = tmp_path / "index" / "passages.jsonl"
    path.write_text(path.read_text() + "{}\n")
    with pytest.raises(ValueError, match="hash"):
        IndexRetriever(tmp_path / "index", FixtureEncoder())


@pytest.mark.parametrize(
    "field,value", [("count", 99), ("dimension", 99), ("metric", "l2"), ("normalized", False)]
)
def test_index_metadata_mismatch_rejected(retriever, tmp_path, field, value):
    path = tmp_path / "index" / "metadata.json"
    data = json.loads(path.read_text())
    data[field] = value
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        IndexRetriever(tmp_path / "index", FixtureEncoder())


def test_empty_and_mismatched_corpus_rejected(tmp_path):
    from citation_audit.retrieval import build_index

    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    with pytest.raises(ValueError, match="empty"):
        build_index(empty, tmp_path / "bad", FixtureEncoder())
    assert digest(empty)
