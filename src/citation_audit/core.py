"""Dependency-light identity, provenance, detection, evaluation, and statistics."""

import csv
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

KEY = ("run_id", "source_path", "prompt_id", "variant")
VARIANTS = {"neutral", "with_citation"}
DETECTOR_VERSION = "structured-candidates-v1"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(record):
    return json.dumps(record, sort_keys=True, ensure_ascii=False, allow_nan=False)


def key(record):
    fields = []
    for name in KEY:
        value = record[name]
        if isinstance(value, bool) or not isinstance(value, (str, int)) or not str(value).strip():
            raise ValueError(f"Invalid key field {name}")
        if name != "prompt_id" and not isinstance(value, str):
            raise ValueError(f"Key field {name} must be a string")
        fields.append(str(value))
    if record["variant"] not in VARIANTS:
        raise ValueError("Unknown prompt variant")
    return tuple(fields)


def validate(rows, *, paired=False):
    keys = [key(row) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate evaluation key")
    if paired:
        groups = defaultdict(set)
        for item in keys:
            groups[item[:3]].add(item[3])
        if not rows or any(variants != VARIANTS for variants in groups.values()):
            raise ValueError("Expected complete neutral/with_citation prompt pairs")
    return rows


def read_jsonl(path):
    result = []
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"Blank JSONL row {number}: {path}")
            try:
                row = json.loads(
                    line,
                    parse_constant=lambda value: (_ for _ in ()).throw(
                        ValueError(f"Nonfinite JSON value {value}")
                    ),
                )
            except (ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"Invalid JSONL row {number}: {path}") from error
            if not isinstance(row, dict):
                raise ValueError("Every JSONL record must be an object")
            result.append(row)
    return result


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_jsonl(path, rows):
    # Validate serialization before opening, so bad records cannot leave a partial file.
    lines = [canonical(row) for row in rows]
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        stream.writelines(line + "\n" for line in lines)


@contextmanager
def output_directory(path, metadata):
    """Reserve a new directory. Failed runs remain visible and cannot be resumed blindly."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    status = {
        "schema_version": 2,
        "state": "incomplete",
        "started_at_utc": datetime.now(UTC).isoformat(),
        **metadata,
    }
    write_json(path / "status.json", status)
    try:
        yield path
    except Exception as error:
        status.update(state="failed", error_type=type(error).__name__)
        raise
    else:
        status["state"] = "complete"
    finally:
        status["finished_at_utc"] = datetime.now(UTC).isoformat()
        # Only the status owned by this newly reserved run is updated.
        temporary = path / "status.next.json"
        write_json(temporary, status)
        temporary.replace(path / "status.json")


def deduplicate_historical(rows):
    """Historical migration only; never call this to repair a new run silently."""
    unique, encodings = {}, {}
    for row in rows:
        identity = tuple(str(row[name]) for name in ("source_path", "prompt_id", "variant"))
        text = canonical(row)
        if identity in encodings and encodings[identity] != text:
            raise ValueError(f"Conflicting historical records: {identity}")
        unique.setdefault(identity, row)
        encodings[identity] = text
    return list(unique.values()), len(rows) - len(unique)


def build_manifest(input_path, categories, count, seed):
    if count < 1 or not categories or len(categories) != len(set(categories)):
        raise ValueError("Require positive count and distinct categories")
    with Path(input_path).open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if not {"data_type", "vanilla"} <= set(reader.fieldnames or []):
            raise ValueError("TSV requires data_type and vanilla columns")
        source = list(reader)
    rng = random.Random(seed)
    source_hash = digest(input_path)
    output = []
    for category in sorted(categories):
        eligible = [
            (i, row["vanilla"].strip())
            for i, row in enumerate(source)
            if row["data_type"] == category and row["vanilla"].strip()
        ]
        if len({text for _, text in eligible}) != len(eligible):
            raise ValueError(f"Duplicate source prompt text in {category}; review source first")
        if len(eligible) < count:
            raise ValueError(f"Insufficient unique prompts in {category}")
        for source_row, text in rng.sample(eligible, count):
            prompt_id = hashlib.sha256((category + "\0" + text).encode()).hexdigest()[:20]
            for variant in ("neutral", "with_citation"):
                output.append(
                    {
                        "prompt_id": prompt_id,
                        "source_path": category,
                        "source_row": source_row,
                        "base_prompt_text": text,
                        "source_input_sha256": source_hash,
                        "sampling_seed": seed,
                        "variant": variant,
                        "prompt_text": text
                        + (" Please cite your sources." if variant == "with_citation" else ""),
                    }
                )
    return output


# Detection identifies structured *candidates*, not valid sources or true claims.
PATTERNS = (
    ("url", re.compile(r"https?://[^\s<>\[\]\"']+", re.IGNORECASE)),
    ("doi", re.compile(r"\b10\.\d{4,9}/[^\s<>\[\]\"']+", re.IGNORECASE)),
    ("numbered", re.compile(r"\[\d+(?:\s*[,;\-]\s*\d+)*\]")),
    ("author_year", re.compile(r"\([A-Z][A-Za-z'\-]+(?: et al\.)?,? (?:19|20)\d{2}[a-z]?\)")),
)


def extract_candidates(text):
    if not isinstance(text, str):
        raise ValueError("Answer must be a string")
    matches = []
    for kind, pattern in PATTERNS:
        for match in pattern.finditer(text):
            start, end = match.span()
            value = match.group().rstrip(".,;:!?)]") if kind in {"url", "doi"} else match.group()
            end = start + len(value)
            # Avoid nested DOI detection inside a URL; keep the encompassing candidate.
            if any(
                start >= candidate["start"] and end <= candidate["end"] for candidate in matches
            ):
                continue
            # Do not split inside a detected URL/DOI/author-year marker. Sentence
            # boundaries are a documented heuristic, not a linguistic entailment parser.
            previous = list(re.finditer(r"[.!?]\s+|\n", text[:start]))
            left = previous[-1].end() if previous else 0
            following = re.search(r"[.!?](?=\s|$)|\n", text[end:])
            right = end + following.end() if following else len(text)
            claim = text[left:right]
            matches.append(
                {
                    "kind": kind,
                    "text": value,
                    "start": start,
                    "end": end,
                    "claim_text": claim.strip(),
                }
            )
    return sorted(matches, key=lambda candidate: candidate["start"])


def evaluate(rows, retriever, threshold=0.6, answer_field="answer_text"):
    validate(rows)
    if not math.isfinite(threshold) or not -1 <= threshold <= 1:
        raise ValueError("Cosine threshold must be finite and between -1 and 1")
    evaluated = []
    for row in rows:
        answer = row[answer_field]
        candidates = extract_candidates(answer)
        judgments = []
        for candidate in candidates:
            evidence = retriever.search(candidate["claim_text"])
            score = max((item["score"] for item in evidence), default=None)
            judgments.append(
                {
                    **candidate,
                    "evidence": evidence,
                    "max_cosine_similarity": score,
                    "retrieval_match": score is not None and score >= threshold,
                    "source_exists": None,
                    "source_supports_claim": None,
                }
            )
        matches = [item["retrieval_match"] for item in judgments]
        evaluated.append(
            {
                **row,
                "evaluated_answer_field": answer_field,
                "answer_sha256": hashlib.sha256(answer.encode()).hexdigest(),
                "detector_version": DETECTOR_VERSION,
                "threshold": threshold,
                "retriever_metadata": retriever.metadata,
                "answer_status": "citation_present"
                if candidates
                else ("empty_answer" if not answer.strip() else "no_detected_citation"),
                "abstention_cue": bool(
                    re.search(
                        r"\b(i cannot verify|i cannot answer|i don't know|insufficient evidence)\b",
                        answer,
                        re.IGNORECASE,
                    )
                ),
                "citation_present": bool(candidates),
                "candidates": judgments,
                "any_retrieval_support": any(matches) if matches else None,
                "all_candidate_retrieval_support": all(matches) if matches else None,
            }
        )
    return evaluated


def select_corrections(rows):
    validate_evaluations(rows)
    return [row for row in rows if row["all_candidate_retrieval_support"] is False]


def validate_evaluations(rows):
    """Reject malformed/stale per-answer logs before selection, merge, or summary."""
    validate(rows)
    for row in rows:
        field = row["evaluated_answer_field"]
        if field not in {"answer_text", "corrected_answer_text"}:
            raise ValueError("Unknown evaluated answer field")
        answer = row[field]
        if row["answer_sha256"] != hashlib.sha256(answer.encode()).hexdigest():
            raise ValueError("Stored judgment does not match answer text")
        if row["detector_version"] != DETECTOR_VERSION:
            raise ValueError("Unknown detector version")
        threshold = row["threshold"]
        if not math.isfinite(threshold) or not -1 <= threshold <= 1:
            raise ValueError("Invalid judgment threshold")
        detected = extract_candidates(answer)
        if len(detected) != len(row["candidates"]):
            raise ValueError("Stored candidates differ from the evaluated answer")
        flags = []
        for candidate, expected in zip(row["candidates"], detected):
            if any(candidate[name] != value for name, value in expected.items()):
                raise ValueError("Candidate span/claim differs from evaluated answer")
            scores = [item["score"] for item in candidate["evidence"]]
            if any(not math.isfinite(score) or not -1 <= score <= 1 for score in scores):
                raise ValueError("Invalid retrieval score")
            maximum = max(scores, default=None)
            flag = maximum is not None and maximum >= threshold
            if (
                candidate["max_cosine_similarity"] != maximum
                or candidate["retrieval_match"] is not flag
            ):
                raise ValueError("Retrieval judgment differs from saved evidence")
            flags.append(flag)
        status = (
            "citation_present"
            if flags
            else ("empty_answer" if not answer.strip() else "no_detected_citation")
        )
        if row["answer_status"] != status or row["citation_present"] is not bool(flags):
            raise ValueError("Invalid answer classification")
        if row["any_retrieval_support"] is not (any(flags) if flags else None):
            raise ValueError("Invalid any-candidate support label")
        if row["all_candidate_retrieval_support"] is not (all(flags) if flags else None):
            raise ValueError("Invalid all-candidate support label")
    return rows


def merge_evaluations(before, corrected, expected_corrections):
    validate_evaluations(before)
    validate_evaluations(corrected)
    validate(expected_corrections)
    originals = {key(row): row for row in before}
    revisions = {key(row): row for row in corrected}
    expected = {key(row): row for row in expected_corrections}
    if revisions.keys() != expected.keys():
        raise ValueError("Missing or unexpected corrected verification records")
    if not revisions.keys() <= originals.keys():
        raise ValueError("Correction contains an unknown key")
    for identity, revision in revisions.items():
        text = expected[identity]["corrected_answer_text"]
        if revision.get("evaluated_answer_field") != "corrected_answer_text":
            raise ValueError("Correction verification evaluated the wrong answer field")
        if revision.get("answer_sha256") != hashlib.sha256(text.encode()).hexdigest():
            raise ValueError("Correction judgment does not match the generated answer")
        for name in ("detector_version", "threshold", "retriever_metadata"):
            if revision[name] != originals[identity][name]:
                raise ValueError("Before/after verification configuration differs")
    merged = [
        {
            "key": dict(zip(KEY, key(row))),
            "was_corrected": key(row) in revisions,
            "before": row,
            "after": revisions.get(key(row), row),
        }
        for row in before
    ]
    if len(merged) != len(before):
        raise ValueError("Evaluation population changed")
    return merged


def summarize(rows):
    validate_evaluations(rows)
    groups = defaultdict(list)
    for row in rows:
        groups[(str(row["run_id"]), row["source_path"], row["variant"])].append(row)
    output = []
    for (run_id, source, variant), group in sorted(groups.items()):
        cited = [row for row in group if row["citation_present"]]
        candidates = [item for row in group for item in row["candidates"]]
        output.append(
            {
                "run_id": run_id,
                "source_path": source,
                "variant": variant,
                "responses": len(group),
                "with_detected_citation": len(cited),
                "no_detected_citation": sum(
                    row["answer_status"] == "no_detected_citation" for row in group
                ),
                "empty_answers": sum(row["answer_status"] == "empty_answer" for row in group),
                "abstention_cues": sum(row["abstention_cue"] for row in group),
                "citation_present_rate_all_answers": len(cited) / len(group),
                "any_retrieval_support_rate_among_cited": (
                    sum(row["any_retrieval_support"] for row in cited) / len(cited)
                    if cited
                    else None
                ),
                "all_candidate_support_rate_among_cited": (
                    sum(row["all_candidate_retrieval_support"] for row in cited) / len(cited)
                    if cited
                    else None
                ),
                "candidate_count": len(candidates),
                "candidate_retrieval_match_rate": (
                    sum(item["retrieval_match"] for item in candidates) / len(candidates)
                    if candidates
                    else None
                ),
            }
        )
    return output


def paired_bootstrap(rows, metric="citation_present", B=10000, seed=42):
    """Within one run/category, sample paired prompt differences, not summary rows."""
    validate(rows, paired=True)
    if metric not in {
        "citation_present",
        "any_retrieval_support",
        "all_candidate_retrieval_support",
    }:
        raise ValueError("Unknown binary per-answer metric")
    if len({(row["run_id"], row["source_path"]) for row in rows}) != 1 or B < 1:
        raise ValueError("Require one run/category and a positive bootstrap count")
    pairs = defaultdict(dict)
    for row in rows:
        value = row[metric]
        if value is not None and type(value) not in {bool, int}:
            raise ValueError("Metric must be binary or null")
        if value is not None and value not in (0, 1):
            raise ValueError("Metric must be binary or null")
        pairs[str(row["prompt_id"])][row["variant"]] = value
    differences = [
        int(values["with_citation"]) - int(values["neutral"])
        for _, values in sorted(pairs.items())
        if all(value is not None for value in values.values())
    ]
    if len(differences) < 2:
        raise ValueError("Need at least two defined complete prompt pairs")
    rng = random.Random(seed)
    estimates = sorted(
        sum(rng.choices(differences, k=len(differences))) / len(differences) for _ in range(B)
    )

    def quantile(p):
        position = (len(estimates) - 1) * p
        low = math.floor(position)
        high = math.ceil(position)
        return estimates[low] + (estimates[high] - estimates[low]) * (position - low)

    return {
        "metric": metric,
        "contrast": "with_citation minus neutral",
        "units": "fraction",
        "estimate": sum(differences) / len(differences),
        "ci_low": quantile(0.025),
        "ci_high": quantile(0.975),
        "n_prompt_pairs": len(differences),
        "excluded_undefined_pairs": len(pairs) - len(differences),
        "B": B,
        "seed": seed,
    }


def transitions(merged):
    counts = Counter(
        f"{row['before']['answer_status']} -> {row['after']['answer_status']}" for row in merged
    )
    proxy_transitions = {}
    for metric in ("citation_present", "any_retrieval_support", "all_candidate_retrieval_support"):

        def label(value):
            return "not_applicable" if value is None else str(bool(value)).lower()

        proxy_transitions[metric] = dict(
            Counter(
                f"{label(row['before'][metric])} -> {label(row['after'][metric])}" for row in merged
            )
        )
    return {
        "responses": len(merged),
        "corrected": sum(row["was_corrected"] for row in merged),
        "answer_status_transitions": dict(counts),
        "proxy_transitions": proxy_transitions,
    }


def correction_bootstrap(merged, metric="citation_present", B=10000, seed=42):
    """Population effect: resample paired prompts, retaining both conditions together."""
    before = [row["before"] for row in merged]
    after = [row["after"] for row in merged]
    validate(before, paired=True)
    validate(after, paired=True)
    if len({(row["run_id"], row["source_path"]) for row in before}) != 1:
        raise ValueError("Require one run/category")
    if {key(row) for row in before} != {key(row) for row in after}:
        raise ValueError("Before/after population differs")
    if any(key(row["before"]) != key(row["after"]) for row in merged):
        raise ValueError("Misaligned before/after identity")
    if (
        metric
        not in {"citation_present", "any_retrieval_support", "all_candidate_retrieval_support"}
        or B < 1
    ):
        raise ValueError("Invalid metric or bootstrap count")
    by_prompt = defaultdict(list)
    for row in merged:
        old, new = row["before"][metric], row["after"][metric]
        if any(
            value is not None and (type(value) not in {bool, int} or value not in (0, 1))
            for value in (old, new)
        ):
            raise ValueError("Metric must be binary or null")
        by_prompt[str(row["before"]["prompt_id"])].append(
            None if old is None or new is None else int(new) - int(old)
        )
    effects = [
        sum(values) / len(values)
        for _, values in sorted(by_prompt.items())
        if all(value is not None for value in values)
    ]
    if len(effects) < 2:
        raise ValueError("Need at least two defined before/after prompt pairs")
    rng = random.Random(seed)
    samples = sorted(sum(rng.choices(effects, k=len(effects))) / len(effects) for _ in range(B))

    def quantile(p):
        position = (len(samples) - 1) * p
        low, high = math.floor(position), math.ceil(position)
        return samples[low] + (samples[high] - samples[low]) * (position - low)

    return {
        "metric": metric,
        "contrast": "after minus before; both variants retained per prompt",
        "units": "fraction",
        "estimate": sum(effects) / len(effects),
        "ci_low": quantile(0.025),
        "ci_high": quantile(0.975),
        "n_prompt_pairs": len(effects),
        "excluded_undefined_pairs": len(by_prompt) - len(effects),
        "B": B,
        "seed": seed,
    }
