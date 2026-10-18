"""Explicit offline/online commands. Importing this module never loads a model."""

import argparse
import importlib.metadata
import json
import math
import platform
import re
import subprocess
from collections import defaultdict
from pathlib import Path

from .core import (
    build_manifest,
    correction_bootstrap,
    deduplicate_historical,
    digest,
    evaluate,
    extract_candidates,
    merge_evaluations,
    output_directory,
    paired_bootstrap,
    read_jsonl,
    select_corrections,
    summarize,
    transitions,
    validate,
    write_json,
    write_jsonl,
)
from .generation import (
    FixtureGenerator,
    HuggingFaceGenerator,
    attach_context,
    correct,
    infer,
)
from .retrieval import FixtureEncoder, IndexRetriever, SemanticEncoder, build_index


def make_encoder(args):
    return (
        FixtureEncoder()
        if args.mode == "fixture"
        else SemanticEncoder(args.embedding_model, args.embedding_revision)
    )


def make_generator(args):
    if args.mode == "fixture":
        if not args.fixture_responses:
            raise ValueError("Fixture mode requires --fixture-responses")
        return FixtureGenerator(read_jsonl(args.fixture_responses))
    return HuggingFaceGenerator(args.model, args.model_revision, args.max_new_tokens, args.device)


def provenance(args):
    versions = {}
    for name in ("numpy", "faiss-cpu", "transformers", "torch", "sentence-transformers"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    try:
        package_root = Path(__file__).resolve().parents[2]
        commit = subprocess.check_output(
            ["git", "-C", str(package_root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "-C", str(package_root), "status", "--porcelain"], text=True
            )
        )
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unknown", None
    inputs = {
        name: {"path": str(value), "sha256": digest(value)}
        for name, value in vars(args).items()
        if name
        in {
            "input",
            "manifest",
            "corpus",
            "fixture_responses",
            "before",
            "corrected",
            "expected_corrections",
        }
        and value is not None
    }
    return {
        "command": args.command,
        "mode": getattr(args, "mode", "analysis"),
        "code_commit": commit,
        "working_tree_dirty": dirty,
        "python": platform.python_version(),
        "packages": versions,
        "inputs": inputs,
        "package_source_sha256": {
            path.name: digest(path) for path in sorted(Path(__file__).resolve().parent.glob("*.py"))
        },
        "settings": {
            name: value
            for name, value in vars(args).items()
            if isinstance(value, (int, float, str, bool)) or value is None
        },
        "claim_scope": "software fixture only"
        if getattr(args, "mode", "") == "fixture"
        else "retrieval proxy; source existence and entailment remain unverified",
    }


def bootstrap_groups(rows, metric, B, seed, *, correction=False):
    groups = defaultdict(list)
    for row in rows:
        identity = row["before"] if correction else row
        groups[(identity["run_id"], identity["source_path"])].append(row)
    function = correction_bootstrap if correction else paired_bootstrap
    output = []
    for (run_id, category), values in sorted(groups.items()):
        output.append(
            {
                "run_id": run_id,
                "source_path": category,
                **function(values, metric=metric, B=B, seed=seed),
            }
        )
    return output


def full_run(args, directory):
    manifest = read_jsonl(args.manifest)
    planned = [{**row, "run_id": args.run_id} for row in manifest]
    validate(planned, paired=True)  # Reject duplicates before loading anything expensive.
    category_prompts = defaultdict(set)
    for row in planned:
        category_prompts[row["source_path"]].add(str(row["prompt_id"]))
    if any(len(prompt_ids) < 2 for prompt_ids in category_prompts.values()):
        raise ValueError("Full run requires at least two prompt pairs per category for bootstrap")
    encoder = make_encoder(args)
    generator = make_generator(args)
    retriever = build_index(args.corpus, directory / "index", encoder)
    write_json(directory / "generation-config.json", generator.metadata)
    write_jsonl(directory / "manifest.jsonl", planned)
    completions = infer(manifest, generator, args.run_id)
    write_jsonl(directory / "completions.jsonl", completions)
    baseline = evaluate(completions, retriever, args.threshold)
    write_jsonl(directory / "baseline-verification.jsonl", baseline)
    selected = select_corrections(baseline, include_missing=args.correct_missing_citations)
    write_jsonl(directory / "selected.jsonl", selected)
    contexts = attach_context(selected, retriever)
    write_jsonl(directory / "contexts.jsonl", contexts)
    corrected = correct(contexts, generator)
    write_jsonl(directory / "corrected-generations.jsonl", corrected)
    judgments = evaluate(corrected, retriever, args.threshold, "corrected_answer_text")
    write_jsonl(directory / "corrected-verification.jsonl", judgments)
    merged = merge_evaluations(baseline, judgments, corrected)
    write_jsonl(directory / "final-records.jsonl", merged)
    write_json(directory / "summary-before.json", summarize(baseline))
    write_json(directory / "summary-after.json", summarize([row["after"] for row in merged]))
    write_json(directory / "transitions.json", transitions(merged))
    write_json(
        directory / "bootstrap-conditions.json",
        bootstrap_groups(baseline, "citation_present", args.B, args.seed),
    )
    write_json(
        directory / "bootstrap-correction.json",
        bootstrap_groups(merged, "citation_present", args.B, args.seed, correction=True),
    )
    return {
        "responses": len(baseline),
        "corrected": len(corrected),
        "accepted_corrections": sum(r["correction_accepted"] for r in merged),
        "rejected_corrections": sum(
            r["correction_attempted"] and not r["correction_accepted"] for r in merged
        ),
        "population_preserved": True,
    }


def execute(args):
    if hasattr(args, "B") and args.B < 1:
        raise ValueError("B must be positive")
    if hasattr(args, "threshold") and (
        not math.isfinite(args.threshold) or not -1 <= args.threshold <= 1
    ):
        raise ValueError("Cosine threshold must be finite and between -1 and 1")
    if getattr(args, "mode", None) == "model":
        required = []
        if args.command in {"run", "infer", "correct"}:
            required.append(args.model_revision)
        if args.command in {"run", "build-index", "verify", "retrieve"}:
            required.append(args.embedding_revision)
        if any(not re.fullmatch(r"[0-9a-f]{40}", value or "") for value in required):
            raise ValueError("Model mode requires immutable 40-character model/embedding revisions")
    with output_directory(args.out, provenance(args)) as directory:
        command = args.command
        if command == "run":
            result = full_run(args, directory)
        elif command == "manifest":
            rows = build_manifest(args.input, args.categories, args.count, args.seed)
            write_jsonl(directory / "manifest.jsonl", rows)
            result = {
                "manifest_rows": len(rows),
                "seed": args.seed,
                "source_categories": args.categories,
            }
        elif command == "audit-history":
            counts = {}
            for name in (
                "tinyllama_completions.jsonl",
                "verification_results.jsonl",
                "corrected_completions.jsonl",
            ):
                path = args.repo / "artifacts" / name
                original_hash = digest(path)
                rows = read_jsonl(path)
                unique, duplicate_count = deduplicate_historical(rows)
                write_jsonl(directory / name, unique)
                if digest(path) != original_hash:
                    raise ValueError("Historical input changed during audit")
                counts[name] = {
                    "input_sha256": original_hash,
                    "raw_rows": len(rows),
                    "unique_records": len(unique),
                    "identical_duplicates": duplicate_count,
                    "distinct_prompt_ids": len({str(row["prompt_id"]) for row in unique}),
                }
            result = {
                "kind": "derived historical files; old labels not revalidated",
                "files": counts,
            }
        elif command == "build-index":
            retriever = build_index(args.input, directory / "index", make_encoder(args))
            result = retriever.metadata
        elif command == "infer":
            rows = read_jsonl(args.manifest)
            validate([{**row, "run_id": args.run_id} for row in rows], paired=True)
            write_jsonl(directory / "records.jsonl", infer(rows, make_generator(args), args.run_id))
            result = {"responses": len(rows)}
        elif command in {"verify", "retrieve", "correct", "select", "extract"}:
            rows = read_jsonl(args.input)
            validate(rows)
            if command == "verify":
                retriever = IndexRetriever(args.index, make_encoder(args))
                output = evaluate(rows, retriever, args.threshold, args.answer_field)
            elif command == "retrieve":
                output = attach_context(rows, IndexRetriever(args.index, make_encoder(args)))
            elif command == "correct":
                output = correct(rows, make_generator(args))
            elif command == "select":
                output = select_corrections(rows, include_missing=args.correct_missing_citations)
            else:
                output = [
                    {**row, "detected_candidates": extract_candidates(row[args.answer_field])}
                    for row in rows
                ]
            write_jsonl(directory / "records.jsonl", output)
            result = {"input_rows": len(rows), "output_rows": len(output)}
        elif command == "merge":
            output = merge_evaluations(
                read_jsonl(args.before),
                read_jsonl(args.corrected),
                read_jsonl(args.expected_corrections),
            )
            write_jsonl(directory / "records.jsonl", output)
            result = transitions(output)
        elif command == "summarize":
            rows = read_jsonl(args.input)
            if rows and "before" in rows[0]:
                rows = [row[args.phase] for row in rows]
            result = summarize(rows)
        elif command == "bootstrap":
            rows = read_jsonl(args.input)
            result = bootstrap_groups(
                rows, args.metric, args.B, args.seed, correction=args.correction
            )
        elif command == "plot":
            import matplotlib

            matplotlib.use("Agg")
            from matplotlib import pyplot as plt

            summaries = json.loads(args.input.read_text(encoding="utf-8"))
            if not isinstance(summaries, list):
                raise ValueError("Plot input must be a summary result.json array")
            figure, axis = plt.subplots(figsize=(8, 4))
            for row in summaries:
                value = row[args.metric]
                if value is not None:
                    axis.bar(f"{row['source_path']}\n{row['variant']}", 100 * value)
            axis.set_ylabel(f"{args.metric} (%)")
            axis.set_ylim(0, 100)
            figure.tight_layout()
            figure.savefig(directory / "proxy-metrics.png", dpi=150)
            plt.close(figure)
            result = {"metric": args.metric, "scope": "retrieval proxy, not fabrication rate"}
        else:
            raise ValueError("Unknown command")
        write_json(directory / "result.json", result)
    print(f"Completed {args.command}: {args.out}")


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    names = (
        "manifest",
        "run",
        "audit-history",
        "build-index",
        "infer",
        "extract",
        "verify",
        "select",
        "retrieve",
        "correct",
        "merge",
        "summarize",
        "bootstrap",
        "plot",
    )
    for name in names:
        item = commands.add_parser(name)
        item.add_argument(
            "--out", type=Path, required=True, help="NEW output directory; never appended"
        )
        if name in {
            "manifest",
            "build-index",
            "extract",
            "verify",
            "select",
            "retrieve",
            "correct",
            "summarize",
            "bootstrap",
            "plot",
        }:
            item.add_argument("--input", type=Path, required=True)
        if name in {"run", "infer"}:
            item.add_argument("--manifest", type=Path, required=True)
            item.add_argument("--run-id", required=True)
        if name in {"run", "infer", "build-index", "verify", "retrieve", "correct"}:
            item.add_argument("--mode", choices=("fixture", "model"), required=True)
        if name in {"run", "build-index", "verify", "retrieve"}:
            item.add_argument("--embedding-model", default="sentence-transformers/all-MiniLM-L6-v2")
            item.add_argument("--embedding-revision")
        if name in {"run", "infer", "correct"}:
            item.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
            item.add_argument("--model-revision")
            item.add_argument("--fixture-responses", type=Path)
            item.add_argument("--max-new-tokens", type=int, default=128)
            item.add_argument("--device", type=int, default=-1)
        if name in {"run", "verify"}:
            item.add_argument("--threshold", type=float, default=0.6)
        if name in {"verify", "retrieve"}:
            item.add_argument("--index", type=Path, required=True)
        if name in {"extract", "verify"}:
            item.add_argument(
                "--answer-field",
                choices=("answer_text", "corrected_answer_text"),
                default="answer_text",
            )
        if name in {"run", "bootstrap", "manifest"}:
            item.add_argument("--seed", type=int, default=42)
        if name in {"run", "bootstrap"}:
            item.add_argument("--B", type=int, default=1000)
        if name == "manifest":
            item.add_argument("--categories", nargs="+", required=True)
            item.add_argument("--count", type=int, required=True)
        if name == "run":
            item.add_argument("--corpus", type=Path, required=True)
        if name in {"run", "select"}:
            item.add_argument(
                "--correct-missing-citations",
                action="store_true",
                help="Opt in to correcting missing citations in the citation-requested condition; not a factuality label",
            )
        if name == "audit-history":
            item.add_argument("--repo", type=Path, default=Path("."))
        if name == "merge":
            for field in ("before", "corrected", "expected-corrections"):
                item.add_argument(f"--{field}", type=Path, required=True)
        if name == "summarize":
            item.add_argument("--phase", choices=("before", "after"), default="after")
        if name == "bootstrap":
            item.add_argument(
                "--metric",
                default="citation_present",
                choices=(
                    "citation_present",
                    "any_retrieval_support",
                    "all_candidate_retrieval_support",
                ),
            )
            item.add_argument("--correction", action="store_true")
        if name == "plot":
            item.add_argument(
                "--metric",
                default="citation_present_rate_all_answers",
                choices=(
                    "citation_present_rate_all_answers",
                    "any_retrieval_support_rate_among_cited",
                    "all_candidate_support_rate_among_cited",
                    "candidate_retrieval_match_rate",
                ),
            )
    return root


def main(argv=None):
    arguments = parser().parse_args(argv)
    try:
        execute(arguments)
    except (ValueError, FileExistsError, FileNotFoundError, KeyError, ImportError) as error:
        raise SystemExit(f"Refused: {error}") from error
