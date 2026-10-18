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

from .association import associate_candidates

KEY = ("run_id", "source_path", "prompt_id", "variant")
VARIANTS = {"neutral", "with_citation"}
DETECTOR_VERSION = "structured-candidates-v2-conservative-association"


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
    return associate_candidates(text, matches)


def support_labels(flags):
    """Three-valued proxies: unresolved associations remain unknown, not failures."""
    if not flags:
        return None, None
    any_match = True if True in flags else (None if None in flags else False)
    all_match = False if False in flags else (None if None in flags else True)
    return any_match, all_match


def generation_issues(text):
    issues = []
    if not text.strip():
        issues.append("empty_answer")
    if re.search(r"<\|(?:user|assistant|system|im_start|im_end)\|>|\[/?INST\]", text):
        issues.append("conversation_role_marker")
    if re.match(r"\s*(?:can you|could you|please)\s+provide\b", text, re.I):
        issues.append("followup_request_heuristic")
    return issues


def claim_query(span):
    query = span["text"]
    for _, pattern in PATTERNS:
        query = pattern.sub(" ", query)
    return " ".join(query.split())


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
            claims = []
            for span in candidate["claim_spans"]:
                query = claim_query(span)
                evidence = retriever.search(query)
                score = max((item["score"] for item in evidence), default=None)
                claims.append(
                    {
                        "claim": span,
                        "retrieval_query": query,
                        "evidence": evidence,
                        "max_cosine_similarity": score,
                        "retrieval_match": score is not None and score >= threshold,
                    }
                )
            evidence = [item for claim in claims for item in claim["evidence"]]
            score = max((item["score"] for item in evidence), default=None)
            _, candidate_match = support_labels([claim["retrieval_match"] for claim in claims])
            judgments.append(
                {
                    **candidate,
                    "evidence": evidence,
                    "claim_evaluations": claims,
                    "max_cosine_similarity": score,
                    "retrieval_match": candidate_match,
                    "source_exists": None,
                    "source_supports_claim": None,
                }
            )
        matches = [item["retrieval_match"] for item in judgments]
        any_match, all_match = support_labels(matches)
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
                "generation_issues": generation_issues(answer),
                "candidates": judgments,
                "any_retrieval_support": any_match,
                "all_candidate_retrieval_support": all_match,
            }
        )
    return evaluated


def select_corrections(rows, *, include_missing=False):
    validate_evaluations(rows)
    selected = []
    for row in rows:
        reasons = []
        if row["all_candidate_retrieval_support"] is False:
            reasons.append("retrieval_mismatch")
        if any(c["association_status"] == "unresolved" for c in row["candidates"]):
            reasons.append("unresolved_claim_association")
        if any(issue != "empty_answer" for issue in row["generation_issues"]):
            reasons.append("generation_format_issue")
        if include_missing and row["variant"] == "with_citation" and not row["citation_present"]:
            reasons.append("requested_citation_missing")
        if reasons:
            selected.append({**row, "correction_reasons": reasons})
    return selected


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
            if (
                candidate["source_exists"] is not None
                or candidate["source_supports_claim"] is not None
            ):
                raise ValueError("Automated retrieval cannot establish source validity")
            claims = candidate["claim_evaluations"]
            if [c["claim"] for c in claims] != expected["claim_spans"]:
                raise ValueError("Evaluated claims differ from associated claims")
            for claim in claims:
                if claim["retrieval_query"] != claim_query(claim["claim"]):
                    raise ValueError("Retrieval query differs from associated claim")
                scores = [item["score"] for item in claim["evidence"]]
                if any(not math.isfinite(s) or not -1 <= s <= 1 for s in scores):
                    raise ValueError("Invalid retrieval score")
                maximum = max(scores, default=None)
                flag = maximum is not None and maximum >= threshold
                if (
                    claim["max_cosine_similarity"] != maximum
                    or claim["retrieval_match"] is not flag
                ):
                    raise ValueError("Claim judgment differs from evidence")
            if candidate["evidence"] != [item for c in claims for item in c["evidence"]]:
                raise ValueError("Candidate evidence differs from claim evidence")
            scores = [item["score"] for item in candidate["evidence"]]
            if any(not math.isfinite(score) or not -1 <= score <= 1 for score in scores):
                raise ValueError("Invalid retrieval score")
            maximum = max(scores, default=None)
            _, flag = support_labels([c["retrieval_match"] for c in claims])
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
        any_match, all_match = support_labels(flags)
        if row["generation_issues"] != generation_issues(answer):
            raise ValueError("Invalid generation quality flags")
        if row["any_retrieval_support"] is not any_match:
            raise ValueError("Invalid any-candidate support label")
        if row["all_candidate_retrieval_support"] is not all_match:
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
    merged = []
    for row in before:
        attempt = revisions.get(key(row))
        reasons = list(attempt["generation_issues"]) if attempt else []
        if attempt and row["citation_present"] and not attempt["citation_present"]:
            reasons.append("lost_detected_citations")
        if attempt and row["variant"] == "with_citation" and not attempt["citation_present"]:
            reasons.append("requested_citation_still_missing")
        if attempt and any(c["association_status"] == "unresolved" for c in attempt["candidates"]):
            reasons.append("unresolved_claim_association")
        if attempt and expected[key(row)].get("correction_diagnostics", {}).get("near_token_cap"):
            reasons.append("possibly_truncated")
        accepted = attempt is not None and not reasons
        merged.append(
            {
                "key": dict(zip(KEY, key(row))),
                "was_corrected": accepted,
                "correction_attempted": attempt is not None,
                "correction_accepted": accepted,
                "acceptance_scope": "format/association guards only; factuality unknown",
                "rejection_reasons": sorted(set(reasons)),
                "attempted_after": attempt,
                "before": row,
                "after": attempt if accepted else row,
            }
        )
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
        any_known = [
            r["any_retrieval_support"] for r in cited if r["any_retrieval_support"] is not None
        ]
        all_known = [
            r["all_candidate_retrieval_support"]
            for r in cited
            if r["all_candidate_retrieval_support"] is not None
        ]
        candidate_known = [
            c["retrieval_match"] for c in candidates if c["retrieval_match"] is not None
        ]
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
                "generation_format_issues": sum(bool(row["generation_issues"]) for row in group),
                "unresolved_candidate_count": len(candidates) - len(candidate_known),
                "any_retrieval_evaluable_answers": len(any_known),
                "all_retrieval_evaluable_answers": len(all_known),
                "evaluable_candidate_count": len(candidate_known),
                "citation_present_rate_all_answers": len(cited) / len(group),
                "any_retrieval_support_rate_among_cited": (
                    sum(any_known) / len(cited) if cited and len(any_known) == len(cited) else None
                ),
                "all_candidate_support_rate_among_cited": (
                    sum(all_known) / len(cited) if cited and len(all_known) == len(cited) else None
                ),
                "candidate_count": len(candidates),
                "candidate_retrieval_match_rate": (
                    sum(candidate_known) / len(candidates)
                    if candidates and len(candidate_known) == len(candidates)
                    else None
                ),
                "candidate_match_rate_evaluable_only": (
                    sum(candidate_known) / len(candidate_known) if candidate_known else None
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
        "correction_attempts": sum(row["correction_attempted"] for row in merged),
        "rejected_corrections": sum(
            row["correction_attempted"] and not row["correction_accepted"] for row in merged
        ),
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
