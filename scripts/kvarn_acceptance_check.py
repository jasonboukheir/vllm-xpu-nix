#!/usr/bin/env python3
"""Aggregate the explicit artifacts required for Kvarn service acceptance."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

REQUIRED_COMPARISONS = 6
MINIMUM_SCORED_POSITIONS = 4096

SERVICE_GATE_ARGUMENTS = (
    ("b1_first", "b1-first-service-gate"),
    ("b1_restart", "b1-restart-service-gate"),
    ("near_first", "near-first-service-gate"),
    ("near_restart", "near-restart-service-gate"),
    ("b4", "b4-service-gate"),
)

ENGINE_LOG_SCAN_ARGUMENTS = (
    ("b1_first", "b1-first-engine-log-scan"),
    ("b1_restart", "b1-restart-engine-log-scan"),
    ("b4", "b4-engine-log-scan"),
)


def resolved(path: Path) -> str:
    return str(path.expanduser().resolve())


def load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("top-level JSON value must be an object")
    return value


def add_failure(
    failures: list[dict[str, str]],
    scope: str,
    message: str,
    *,
    path: str | None = None,
    role: str | None = None,
) -> None:
    failure = {"scope": scope, "message": message}
    if role is not None:
        failure["role"] = role
    if path is not None:
        failure["path"] = path
    failures.append(failure)


def comparison_record(
    path: Path, failures: list[dict[str, str]]
) -> tuple[dict[str, Any], int]:
    normalized_path = resolved(path)
    record: dict[str, Any] = {
        "path": normalized_path,
        "status": None,
        "acceptance_status": None,
        "decode_steps": None,
        "passed": False,
    }
    try:
        document = load_json_object(path)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError) as error:
        add_failure(
            failures,
            "comparison",
            f"cannot load comparison JSON: {error}",
            path=normalized_path,
        )
        return record, 0

    status = document.get("status")
    acceptance = document.get("acceptance")
    acceptance_status = (
        acceptance.get("status") if isinstance(acceptance, dict) else None
    )
    decode_steps = document.get("decode_steps")
    record.update(
        status=status,
        acceptance_status=acceptance_status,
        decode_steps=decode_steps,
    )

    passed = True
    if status != "passed":
        add_failure(
            failures,
            "comparison",
            f"status must be 'passed', got {status!r}",
            path=normalized_path,
        )
        passed = False
    if acceptance_status != "passed":
        add_failure(
            failures,
            "comparison",
            f"acceptance.status must be 'passed', got {acceptance_status!r}",
            path=normalized_path,
        )
        passed = False
    if (
        isinstance(decode_steps, bool)
        or not isinstance(decode_steps, int)
        or decode_steps <= 0
    ):
        add_failure(
            failures,
            "comparison",
            "decode_steps must be a positive integer",
            path=normalized_path,
        )
        decode_steps = 0
        passed = False

    record["passed"] = passed
    return record, decode_steps


def service_gate_record(
    role: str,
    path: Path,
    failures: list[dict[str, str]],
) -> dict[str, Any]:
    normalized_path = resolved(path)
    record: dict[str, Any] = {
        "role": role,
        "path": normalized_path,
        "status": None,
        "passed": False,
    }
    try:
        document = load_json_object(path)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError) as error:
        add_failure(
            failures,
            "service_gate",
            f"cannot load service-gate JSON: {error}",
            role=role,
            path=normalized_path,
        )
        return record

    status = document.get("status")
    record["status"] = status
    if status != "passed":
        add_failure(
            failures,
            "service_gate",
            f"status must be 'passed', got {status!r}",
            role=role,
            path=normalized_path,
        )
        return record

    record["passed"] = True
    return record


def engine_log_scan_record(
    role: str,
    path: Path,
    failures: list[dict[str, str]],
) -> tuple[dict[str, Any], int]:
    normalized_path = resolved(path)
    record: dict[str, Any] = {
        "role": role,
        "path": normalized_path,
        "status": None,
        "fatal_findings": None,
        "known_teardown_findings": None,
        "passed": False,
    }
    try:
        document = load_json_object(path)
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError) as error:
        add_failure(
            failures,
            "engine_log_scan",
            f"cannot load engine-log scan JSON: {error}",
            role=role,
            path=normalized_path,
        )
        return record, 0

    status = document.get("status")
    fatal = document.get("fatal_findings")
    known_teardown = document.get("known_teardown_findings")
    record.update(
        status=status,
        fatal_findings=len(fatal) if isinstance(fatal, list) else None,
        known_teardown_findings=(
            len(known_teardown) if isinstance(known_teardown, list) else None
        ),
    )

    passed = True
    if status != "passed":
        add_failure(
            failures,
            "engine_log_scan",
            f"status must be 'passed', got {status!r}",
            role=role,
            path=normalized_path,
        )
        passed = False
    if not isinstance(fatal, list):
        add_failure(
            failures,
            "engine_log_scan",
            "fatal_findings must be a list",
            role=role,
            path=normalized_path,
        )
        passed = False
    elif fatal:
        add_failure(
            failures,
            "engine_log_scan",
            f"fatal_findings must be empty, got {len(fatal)}",
            role=role,
            path=normalized_path,
        )
        passed = False
    if not isinstance(known_teardown, list):
        add_failure(
            failures,
            "engine_log_scan",
            "known_teardown_findings must be a list",
            role=role,
            path=normalized_path,
        )
        passed = False

    record["passed"] = passed
    return record, len(known_teardown) if isinstance(known_teardown, list) else 0


def repeated_paths(paths: list[Path]) -> list[str]:
    normalized = [resolved(path) for path in paths]
    return sorted({path for path in normalized if normalized.count(path) > 1})


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    failures: list[dict[str, str]] = []

    comparisons: list[dict[str, Any]] = []
    scored_positions = 0
    if len(args.comparison) != REQUIRED_COMPARISONS:
        add_failure(
            failures,
            "comparisons",
            f"expected {REQUIRED_COMPARISONS} files, got {len(args.comparison)}",
        )
    for duplicate in repeated_paths(args.comparison):
        add_failure(
            failures,
            "comparisons",
            "comparison paths must be distinct",
            path=duplicate,
        )
    for path in args.comparison:
        record, positions = comparison_record(path, failures)
        comparisons.append(record)
        scored_positions += positions
    if scored_positions < MINIMUM_SCORED_POSITIONS:
        add_failure(
            failures,
            "comparisons",
            (
                f"total decode_steps must be at least {MINIMUM_SCORED_POSITIONS}, "
                f"got {scored_positions}"
            ),
        )

    service_paths = [
        getattr(args, option.replace("-", "_"))
        for _role, option in SERVICE_GATE_ARGUMENTS
    ]
    for duplicate in repeated_paths(service_paths):
        add_failure(
            failures,
            "service_gates",
            "service-gate paths must be distinct",
            path=duplicate,
        )
    service_gates = [
        service_gate_record(
            role,
            getattr(args, option.replace("-", "_")),
            failures,
        )
        for role, option in SERVICE_GATE_ARGUMENTS
    ]

    scan_paths = [
        getattr(args, option.replace("-", "_"))
        for _role, option in ENGINE_LOG_SCAN_ARGUMENTS
    ]
    for duplicate in repeated_paths(scan_paths):
        add_failure(
            failures,
            "engine_log_scans",
            "engine-log scan paths must be distinct",
            path=duplicate,
        )
    engine_log_scans: list[dict[str, Any]] = []
    known_teardown_findings = 0
    for role, option in ENGINE_LOG_SCAN_ARGUMENTS:
        record, known_count = engine_log_scan_record(
            role,
            getattr(args, option.replace("-", "_")),
            failures,
        )
        engine_log_scans.append(record)
        known_teardown_findings += known_count

    return {
        "schema_version": 1,
        "status": "failed" if failures else "passed",
        "summary": {
            "comparison_files": len(comparisons),
            "required_comparison_files": REQUIRED_COMPARISONS,
            "scored_positions": scored_positions,
            "minimum_scored_positions": MINIMUM_SCORED_POSITIONS,
            "service_gates": len(service_gates),
            "engine_log_scans": len(engine_log_scans),
            "known_teardown_findings": known_teardown_findings,
        },
        "comparisons": comparisons,
        "service_gates": service_gates,
        "engine_log_scans": engine_log_scans,
        "failures": failures,
    }


def write_json_atomic(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    temporary.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--comparison",
        action="append",
        type=Path,
        required=True,
        help="thresholded comparison JSON; repeat exactly six times",
    )
    for _role, option in SERVICE_GATE_ARGUMENTS:
        parser.add_argument(f"--{option}", type=Path, required=True)
    for _role, option in ENGINE_LOG_SCAN_ARGUMENTS:
        parser.add_argument(f"--{option}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-tmp", action="store_true")
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    if not args.allow_tmp and output.is_relative_to(Path("/tmp")):
        parser.error("--output must be durable (outside /tmp)")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    document = evaluate(args)
    write_json_atomic(args.output, document)
    print(json.dumps(document, indent=2, sort_keys=True))
    return int(document["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
