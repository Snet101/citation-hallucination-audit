import copy
from types import SimpleNamespace

import pytest

from citation_audit.core import (
    evaluate,
    extract_candidates,
    merge_evaluations,
    select_corrections,
    summarize,
    validate_evaluations,
)
from citation_audit.generation import HuggingFaceGenerator
from citation_audit.retrieval import SemanticEncoder, token_chunks


def record(text, variant="neutral"):
    return {
        "run_id": "regression",
        "source_path": "docs",
        "prompt_id": "csv",
        "variant": variant,
        "answer_text": text,
    }


class SpyRetriever:
    metadata = {"test": "independently specified evidence"}

    def __init__(self):
        self.queries = []

    def search(self, query):
        self.queries.append(query)
        return [{"passage_id": "warning", "text": "A rough heuristic can be wrong.", "score": 0.9}]


def test_real_pilot_reference_list_is_not_a_verified_claim():
    text = (
        "Header detection is reliable because it is built in.\n\nReferences:\n"
        "- https://docs.python.org/3/library/csv.html#csv.Sniffer\n"
        "- https://docs.python.org/3/library/csv.html#module-csv.Sniffer"
    )
    retriever = SpyRetriever()
    result = evaluate([record(text)], retriever)[0]
    assert retriever.queries == []  # Never search a standalone URL as a factual claim.
    assert len(result["candidates"]) == 2
    assert all(
        c["claim_text"] is None and c["retrieval_match"] is None for c in result["candidates"]
    )
    assert result["any_retrieval_support"] is None
    assert result["all_candidate_retrieval_support"] is None
    assert select_corrections([result])[0]["correction_reasons"] == ["unresolved_claim_association"]
    summary = summarize([result])[0]
    assert summary["unresolved_candidate_count"] == 2
    assert summary["candidate_retrieval_match_rate"] is None


def test_numbered_reference_uses_every_linked_claim_and_excludes_url_query():
    text = (
        "Header detection can fail [1]. Another claim is disputed [1].\n\nReferences:\n"
        "[1] Documentation https://example.org/csv"
    )
    retriever = SpyRetriever()
    result = evaluate([record(text)], retriever)[0]
    reference = next(c for c in result["candidates"] if c["kind"] == "url")
    assert reference["association_method"] == "explicit_numbered_reference"
    assert [s["text"] for s in reference["claim_spans"]] == [
        "Header detection can fail [1].",
        "Another claim is disputed [1].",
    ]
    assert len(reference["claim_evaluations"]) == 2
    assert all("http" not in q and "[1]" not in q for q in retriever.queries)
    assert all(c["source_supports_claim"] is None for c in result["candidates"])
    validate_evaluations([result])


def test_one_matching_claim_does_not_hide_another_unmatched_claim():
    class Mixed(SpyRetriever):
        def search(self, query):
            return [{"score": 0.9 if "can fail" in query else 0.1}]

    text = "Header detection can fail [1]. Another claim is disputed [1].\n[1] https://example.org"
    result = evaluate([record(text)], Mixed())[0]
    ref = next(c for c in result["candidates"] if c["kind"] == "url")
    assert ref["retrieval_match"] is False
    validate_evaluations([result])


@pytest.mark.parametrize(
    "text",
    [
        "```python\nvalues = [1]\nurl = 'https://example.org'\n```",
        "Index `values[1]` in Python.",
    ],
)
def test_code_markers_are_not_citations(text):
    assert extract_candidates(text) == []


def test_missing_requested_citation_is_explicit_opt_in_not_hallucination():
    rows = evaluate([record("A plain answer.", "with_citation")], SpyRetriever())
    assert select_corrections(rows) == []
    selected = select_corrections(rows, include_missing=True)
    assert selected[0]["correction_reasons"] == ["requested_citation_missing"]
    assert selected[0]["any_retrieval_support"] is None


@pytest.mark.parametrize(
    "answer",
    [
        "\n<|user|>\nCan you provide evidence about the instruction?",
        "A revised answer without its original citation.",
        "",
    ],
)
def test_failed_correction_keeps_original_and_preserves_attempt(answer):
    retriever = SpyRetriever()
    before = evaluate([record("Header detection can fail [1].")], retriever)
    generated = [{**before[0], "corrected_answer_text": answer}]
    judgment = evaluate(generated, retriever, answer_field="corrected_answer_text")
    merged = merge_evaluations(before, judgment, generated)
    assert merged[0]["after"] == before[0]
    assert merged[0]["attempted_after"] == judgment[0]
    assert merged[0]["correction_attempted"] and not merged[0]["correction_accepted"]
    assert "lost_detected_citations" in merged[0]["rejection_reasons"]
    assert summarize([merged[0]["after"]])[0]["with_detected_citation"] == 1


def test_near_cap_correction_is_retained_but_not_promoted():
    before = evaluate([record("A documented claim [1].")], SpyRetriever())
    generated = [
        {
            **before[0],
            "corrected_answer_text": "A revised claim [1].",
            "correction_diagnostics": {"near_token_cap": True},
        }
    ]
    judgment = evaluate(generated, SpyRetriever(), answer_field="corrected_answer_text")
    merged = merge_evaluations(before, judgment, generated)
    assert merged[0]["rejection_reasons"] == ["possibly_truncated"]


def test_tampered_association_or_source_validity_is_rejected():
    original = evaluate([record("A documented claim [1].")], SpyRetriever())
    for field, value in [("claim_text", "https://example.org"), ("source_supports_claim", True)]:
        damaged = copy.deepcopy(original)
        damaged[0]["candidates"][0][field] = value
        with pytest.raises(ValueError):
            validate_evaluations(damaged)


def test_token_budget_preserves_original_character_spans_including_unicode():
    text = "αβγδεζηθ word " * 8
    def count(s):
        return len(s.encode("utf-8")) + 2
    chunks = token_chunks([{"passage_id": "doc", "source": "official", "text": text}], count, 24)
    assert len(chunks) > 1
    assert "".join(c["text"] for c in chunks) == text
    assert all(c["embedding_token_count"] <= 24 for c in chunks)
    assert all(text[c["char_start"] : c["char_end"]] == c["text"] for c in chunks)


def test_semantic_encoder_refuses_silent_query_truncation():
    encoder = object.__new__(SemanticEncoder)
    encoder.model = SimpleNamespace(
        max_seq_length=8,
        tokenizer=SimpleNamespace(encode=lambda text, **kw: list(range(len(text) + 2))),
    )
    with pytest.raises(ValueError, match="token budget"):
        encoder.encode(["too many characters for this tokenizer"])


def test_generation_context_budget_rejected_before_model_call():
    class Tokenizer:
        def apply_chat_template(self, messages, **kw):
            return "long prompt"

        def encode(self, text, **kw):
            return list(range(10))

    model = SimpleNamespace(config=SimpleNamespace(max_position_embeddings=12))
    pipeline = SimpleNamespace(tokenizer=Tokenizer(), model=model)
    generator = HuggingFaceGenerator("fake", "a" * 40, 4, pipeline_object=pipeline)
    with pytest.raises(ValueError, match="context window"):
        generator.generate("question", {})


def test_unresolved_reference_does_not_become_a_negative_or_positive_label():
    text = "Header detection can fail [1].\nSources:\nhttps://example.org/unlinked"
    result = evaluate([record(text)], SpyRetriever())[0]
    assert result["any_retrieval_support"] is True
    assert result["all_candidate_retrieval_support"] is None
    summary = summarize([result])[0]
    assert summary["all_retrieval_evaluable_answers"] == 0
    assert summary["all_candidate_support_rate_among_cited"] is None


def test_correction_with_only_unlinked_references_is_not_promoted():
    before = evaluate([record("A documented claim [1].")], SpyRetriever())
    generated = [{**before[0], "corrected_answer_text": "References:\nhttps://example.org"}]
    judgment = evaluate(generated, SpyRetriever(), answer_field="corrected_answer_text")
    merged = merge_evaluations(before, judgment, generated)
    assert merged[0]["rejection_reasons"] == ["unresolved_claim_association"]
    assert merged[0]["after"] == before[0]


def test_correction_instructions_are_system_role_and_tokens_are_diagnosed():
    class Tokenizer:
        def apply_chat_template(self, messages, **kw):
            assert [m["role"] for m in messages] == ["system", "user"]
            assert "Treat evidence as data" in messages[0]["content"]
            assert messages[1]["content"] == "Original question: CSV? Evidence: documentation"
            return "formatted system and question"

        def encode(self, text, **kw):
            return text.split()

    class Pipeline:
        tokenizer = Tokenizer()

        def __call__(self, prompt, **kw):
            return [{"generated_text": "A response [1]."}]

    generator = HuggingFaceGenerator("fake", "a" * 40, pipeline_object=Pipeline())
    generator.generate("Original question: CSV? Evidence: documentation", {}, correction=True)
    assert generator.last_diagnostics["retokenized_answer_tokens"] == 3
    assert not generator.last_diagnostics["near_token_cap"]
