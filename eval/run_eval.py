from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from pydantic import ValidationError  # noqa: E402

from classifier import classify_document  # noqa: E402
from date_extractor import extract_date_mentions  # noqa: E402


def normalise(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"(\d+)(st|nd|rd|th)", r"\1", text)
    text = re.sub(r"[,.]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def dates_match(extracted: str, expected: str) -> bool:
    extracted_norm = normalise(extracted)
    expected_norm = normalise(expected)
    return expected_norm in extracted_norm or extracted_norm in expected_norm


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((pct / 100) * (len(ordered) - 1))))
    return ordered[index]


def evaluate_dates(documents: list[dict]) -> dict:
    rows = []
    total_expected = total_matched = total_extracted = total_extra = 0
    latencies = []

    for document in documents:
        start = time.perf_counter()
        extracted = extract_date_mentions(document["text"])
        latencies.append((time.perf_counter() - start) * 1000)

        expected = document.get("expected_dates") or []
        matched = 0
        missing = []
        for expected_date in expected:
            if any(dates_match(item, expected_date) for item in extracted):
                matched += 1
            else:
                missing.append(expected_date)

        extras = [
            item
            for item in extracted
            if not any(dates_match(item, expected_date) for expected_date in expected)
        ]

        total_expected += len(expected)
        total_matched += matched
        total_extracted += len(extracted)
        total_extra += len(extras)

        rows.append(
            {
                "id": document["id"],
                "expected": len(expected),
                "extracted": len(extracted),
                "matched": matched,
                "missing": missing,
                "extras": extras,
            }
        )

    recall = total_matched / total_expected if total_expected else 0.0
    precision = total_matched / total_extracted if total_extracted else 0.0
    return {
        "rows": rows,
        "recall": recall,
        "precision": precision,
        "extras": total_extra,
        "latency_ms_p50": percentile(latencies, 50),
        "latency_ms_p95": percentile(latencies, 95),
    }


def evaluate_doc_types(documents: list[dict]) -> dict:
    rows = []
    correct = total = 0

    for document in documents:
        expected = document.get("doc_type")
        if not expected:
            continue
        predicted = classify_document(document["text"])
        is_correct = predicted == expected
        correct += int(is_correct)
        total += 1
        rows.append(
            {
                "id": document["id"],
                "expected": expected,
                "predicted": predicted,
                "correct": is_correct,
            }
        )

    return {
        "rows": rows,
        "accuracy": correct / total if total else 0.0,
        "correct": correct,
        "total": total,
    }


def ensure_index() -> None:
    from build_chroma_index import main as build_index
    from rag import corpus_size

    if corpus_size() == 0:
        print("Legal index missing; building it before retrieval evaluation…")
        build_index()


def evaluate_retrieval(questions: list[dict], k: int = 3) -> dict:
    from rag import retrieve_legal_context

    ensure_index()
    retrieve_legal_context("warmup query about legal aid", k=k)  # warm up models

    rows = []
    hit1 = hitk = 0
    reciprocal_ranks = []
    latencies = []

    for question in questions:
        start = time.perf_counter()
        hits = retrieve_legal_context(
            question["query"], k=k, doc_type=question.get("doc_type")
        )
        latencies.append((time.perf_counter() - start) * 1000)

        expected_ids = set(question.get("expected_ids") or [])
        rank = None
        for index, hit in enumerate(hits, start=1):
            if hit["id"] in expected_ids:
                rank = index
                break

        if rank == 1:
            hit1 += 1
        if rank is not None:
            hitk += 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

        rows.append(
            {
                "id": question["id"],
                "rank": rank,
                "expected": sorted(expected_ids),
                "returned": [hit["id"] for hit in hits],
            }
        )

    total = len(questions)
    return {
        "rows": rows,
        "hit_at_1": hit1 / total if total else 0.0,
        "hit_at_k": hitk / total if total else 0.0,
        "mrr": sum(reciprocal_ranks) / total if total else 0.0,
        "latency_ms_p50": percentile(latencies, 50),
        "latency_ms_p95": percentile(latencies, 95),
    }


def evaluate_schema(cases: list[dict]) -> dict:
    from analyser import DocumentAnalysis

    rows = []
    correct = 0
    for case in cases:
        try:
            DocumentAnalysis(**case["payload"])
            passed = True
        except ValidationError:
            passed = False

        is_correct = passed == bool(case["should_validate"])
        correct += int(is_correct)
        rows.append(
            {
                "name": case["name"],
                "expected_valid": case["should_validate"],
                "actual_valid": passed,
                "correct": is_correct,
            }
        )

    return {
        "rows": rows,
        "correct": correct,
        "total": len(cases),
        "accuracy": correct / len(cases) if cases else 0.0,
    }


def print_report(report: dict) -> None:
    print("\n# Nyaya Mitra Evaluation\n")
    print(
        f"Provider: {os.environ.get('LLM_PROVIDER', 'ollama')} | "
        f"Date: {time.strftime('%Y-%m-%d %H:%M')}"
    )

    dates = report["dates"]
    print("\n## Date extraction")
    print(
        f"- Recall: **{dates['recall']:.1%}** | "
        f"Precision: **{dates['precision']:.1%}** | "
        f"Extra dates: {dates['extras']}"
    )
    print(
        f"- Latency: p50 **{dates['latency_ms_p50']:.1f} ms** | "
        f"p95 **{dates['latency_ms_p95']:.1f} ms**"
    )
    for row in dates["rows"]:
        print(
            f"  - `{row['id']}`: {row['matched']}/{row['expected']} dates"
            + (f" | missing: {row['missing']}" if row["missing"] else "")
            + (f" | extra: {row['extras']}" if row["extras"] else "")
        )

    doc_types = report["doc_types"]
    print("\n## Document classification")
    print(
        f"- Accuracy: **{doc_types['accuracy']:.1%}** "
        f"({doc_types['correct']}/{doc_types['total']})"
    )

    retrieval = report["retrieval"]
    print("\n## Retrieval")
    print(
        f"- Hit@1: **{retrieval['hit_at_1']:.1%}** | "
        f"Hit@3: **{retrieval['hit_at_k']:.1%}** | "
        f"MRR: **{retrieval['mrr']:.3f}**"
    )
    print(
        f"- Latency: p50 **{retrieval['latency_ms_p50']:.1f} ms** | "
        f"p95 **{retrieval['latency_ms_p95']:.1f} ms**"
    )
    for row in retrieval["rows"]:
        status = f"rank {row['rank']}" if row["rank"] else "MISS"
        print(f"  - `{row['id']}`: {status}")

    schema = report["schema"]
    print("\n## Schema validation")
    print(
        f"- Accuracy: **{schema['accuracy']:.1%}** "
        f"({schema['correct']}/{schema['total']})"
    )
    for row in schema["rows"]:
        print(
            f"  - `{row['name']}`: expected "
            f"{'valid' if row['expected_valid'] else 'invalid'}, "
            f"got {'valid' if row['actual_valid'] else 'invalid'}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate Nyaya Mitra")
    parser.add_argument("--k", type=int, default=3, help="retrieval cutoff")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of Markdown")
    args = parser.parse_args()

    with open(HERE / "eval_dataset.json", encoding="utf-8") as handle:
        dataset = json.load(handle)

    report = {
        "dates": evaluate_dates(dataset["documents"]),
        "doc_types": evaluate_doc_types(dataset["documents"]),
        "retrieval": evaluate_retrieval(dataset["retrieval_questions"], k=args.k),
        "schema": evaluate_schema(dataset["schema_cases"]),
    }

    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print_report(report)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
