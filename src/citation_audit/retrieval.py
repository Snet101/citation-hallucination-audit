"""Normalized inner-product FAISS retrieval with persisted provenance and checks."""

import hashlib
import re
from pathlib import Path

import faiss
import numpy as np

from .core import digest, read_jsonl, write_json, write_jsonl


class FixtureEncoder:
    """Deterministic token-hash encoder for software tests, NOT semantic evidence."""

    def __init__(self):
        self.metadata = {
            "model_id": "fixture-token-hash",
            "revision": "v1",
            "dimension": 128,
            "mode": "fixture",
        }

    def encode(self, texts):
        vectors = np.zeros((len(texts), 128), dtype="float32")
        for i, text in enumerate(texts):
            for word in re.findall(r"[a-z]+", text.lower()):
                index = int(hashlib.sha256(word.encode()).hexdigest()[:8], 16) % 128
                vectors[i, index] += 1
        return vectors


class SemanticEncoder:
    def __init__(self, model_id, revision):
        if not re.fullmatch(r"[0-9a-f]{40}", revision or ""):
            raise ValueError("Embedding revision must be an immutable 40-character commit hash")
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_id, revision=revision, trust_remote_code=False, device="cpu")
        self.metadata = {
            "model_id": model_id,
            "revision": revision,
            "dimension": self.model.get_sentence_embedding_dimension(),
            "max_seq_length": self.model.max_seq_length,
            "device": "cpu",
            "mode": "model",
        }

    def encode(self, texts):
        return self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)


def normalized(vectors, dimension):
    values = np.asarray(vectors, dtype="float32")
    if values.ndim != 2 or values.shape[1] != dimension or not np.isfinite(values).all():
        raise ValueError("Invalid embedding dimensions or nonfinite values")
    lengths = np.linalg.norm(values, axis=1, keepdims=True)
    if (lengths == 0).any():
        raise ValueError("Cannot index or query a zero embedding")
    return np.ascontiguousarray(values / lengths)


def build_index(corpus_path, output, encoder):
    passages = read_jsonl(corpus_path)
    if not passages:
        raise ValueError("Cannot build an empty index")
    ids = []
    for passage in passages:
        if not all(
            isinstance(passage.get(name), str) and passage[name].strip()
            for name in ("passage_id", "text", "source")
        ):
            raise ValueError("Corpus requires nonempty string passage_id, text, and source")
        ids.append(passage["passage_id"])
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate passage IDs")
    vectors = normalized(
        encoder.encode([row["text"] for row in passages]), encoder.metadata["dimension"]
    )
    if len(vectors) != len(passages):
        raise ValueError("Embedding row count differs from corpus")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_jsonl(output / "passages.jsonl", passages)
    faiss.write_index(index, str(output / "index.faiss"))
    metadata = {
        "encoder": encoder.metadata,
        "metric": "normalized_inner_product",
        "index_type": "IndexFlatIP",
        "normalized": True,
        "count": len(passages),
        "dimension": vectors.shape[1],
        "source_corpus_sha256": digest(corpus_path),
        "passages_sha256": digest(output / "passages.jsonl"),
        "index_sha256": digest(output / "index.faiss"),
    }
    write_json(output / "metadata.json", metadata)
    return IndexRetriever(output, encoder)


class IndexRetriever:
    def __init__(self, directory, encoder):
        import json

        directory = Path(directory)
        self.metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        metadata = self.metadata
        if (
            metadata["encoder"] != encoder.metadata
            or metadata["metric"] != "normalized_inner_product"
        ):
            raise ValueError("Embedding model/revision or metric mismatch")
        if metadata.get("normalized") is not True or metadata.get("index_type") != "IndexFlatIP":
            raise ValueError("Unsupported index normalization/type")
        for filename, field in (
            ("passages.jsonl", "passages_sha256"),
            ("index.faiss", "index_sha256"),
        ):
            if digest(directory / filename) != metadata[field]:
                raise ValueError(f"Index input hash mismatch: {filename}")
        self.passages = read_jsonl(directory / "passages.jsonl")
        self.index = faiss.read_index(str(directory / "index.faiss"))
        if (
            not self.index.ntotal
            or self.index.ntotal != len(self.passages)
            or self.index.ntotal != metadata["count"]
        ):
            raise ValueError("Empty index or passage count mismatch")
        if self.index.d != metadata["dimension"] or self.index.d != encoder.metadata["dimension"]:
            raise ValueError("Index dimension mismatch")
        if self.index.metric_type != faiss.METRIC_INNER_PRODUCT:
            raise ValueError("Index is not inner-product retrieval")
        self.encoder = encoder

    def search(self, query, k=3):
        if k < 1:
            raise ValueError("k must be positive")
        vectors = normalized(self.encoder.encode([query]), self.index.d)
        if len(vectors) != 1:
            raise ValueError("Query encoder must return exactly one vector")
        scores, ids = self.index.search(vectors, min(k, self.index.ntotal))
        evidence = []
        for score, index in zip(scores[0], ids[0]):
            if not 0 <= int(index) < len(self.passages):
                continue
            if not np.isfinite(score):
                raise ValueError("Nonfinite retrieval score")
            passage = self.passages[int(index)]
            evidence.append({**passage, "score": float(np.clip(score, -1, 1))})
        return evidence
