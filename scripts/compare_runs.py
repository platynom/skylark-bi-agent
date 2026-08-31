"""Compare JSON outputs produced by scripts/bench_providers.py.

Usage:
    python scripts/compare_runs.py results/*.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def load_run(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    if document.get("schema_version") != 1 or not isinstance(document.get("cases"), list):
        raise ValueError(f"{path} is not a provider benchmark result")
    document["_path"] = str(path)
    return document


def configuration_name(run: dict[str, Any]) -> str:
    config = run["configuration"]
    return f"{config['provider']}/{config['model']}@{config['endpoint']}"


def _table(headers: list[str], rows: list[list[Any]]) -> None:
    rendered = [[str(value) for value in row] for row in rows]
    widths = [
        max(len(header), *(len(row[index]) for row in rendered))
        for index, header in enumerate(headers)
    ]
    print(" | ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in rendered:
        print(" | ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def summary_rows(runs: list[dict[str, Any]]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for run in runs:
        summary = run.get("summary") or {}
        counts = summary.get("counts") or {}
        latency = summary.get("latency_ms") or {}
        rows.append(
            [
                configuration_name(run),
                f"{summary.get('quality_score', 'n/a')}%",
                counts.get("FAIL_VALIDATOR", 0),
                counts.get("FAIL_PROVIDER", 0),
                latency.get("mean", 0),
                latency.get("median", 0),
                latency.get("p95", 0),
                "YES" if latency.get("p95_over_15s") else "no",
            ]
        )
    return rows


def disagreements(runs: list[dict[str, Any]]) -> list[tuple[str, str, list[str]]]:
    by_run = [
        {case["id"]: case for case in run.get("cases", [])}
        for run in runs
    ]
    common = set.intersection(*(set(mapping) for mapping in by_run)) if by_run else set()
    output: list[tuple[str, str, list[str]]] = []
    for case_id in sorted(common):
        signatures = [mapping[case_id].get("answer_signature") for mapping in by_run]
        if len(set(signatures)) <= 1:
            continue
        question = by_run[0][case_id].get("question", "")
        descriptions: list[str] = []
        for run, mapping in zip(runs, by_run):
            case = mapping[case_id]
            turn = case.get("turn") or {}
            result = turn.get("result")
            preview = json.dumps(result[:2] if isinstance(result, list) else result, default=str)
            if len(preview) > 180:
                preview = preview[:177] + "..."
            descriptions.append(
                f"{configuration_name(run)}: {case.get('status')} action={turn.get('action')} result={preview}"
            )
        output.append((case_id, question, descriptions))
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare provider benchmark result files.")
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args(argv)
    try:
        runs = [load_run(path) for path in args.files]
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if len(runs) < 2:
        print("ERROR: provide at least two benchmark JSON files", file=sys.stderr)
        return 2

    _table(
        ["configuration", "quality", "validator", "provider", "mean_ms", "median_ms", "p95_ms", "p95>15s"],
        summary_rows(runs),
    )
    differences = disagreements(runs)
    print(f"\nANSWER DISAGREEMENTS ({len(differences)})")
    for case_id, question, descriptions in differences:
        print(f"\n{case_id}  {question}")
        for description in descriptions:
            print(f"  {description}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
