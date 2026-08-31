#!/usr/bin/env python3
"""Classify fatal engine-log signatures for Kvarn service gates."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

FATAL_PATTERNS = {
    "non_finite": re.compile(r"(?i)(?<![a-z])(?:nan|infinity)(?![a-z])"),
    "explicit_non_finite": re.compile(r"(?i)non[- ]finite"),
    "device_failure": re.compile(r"(?i)device\s+(?:lost|fault)"),
    "segmentation_fault": re.compile(r"(?i)segmentation fault"),
    "out_of_memory": re.compile(r"(?i)out of memory"),
    "fatal_python": re.compile(r"(?i)fatal python error"),
    "runtime_error": re.compile(r"(?i)\bRuntimeError\b"),
    "assertion_error": re.compile(r"(?i)\bAssertionError\b"),
    "engine_dead_error": re.compile(r"(?i)\bEngineDeadError\b"),
    "error_level": re.compile(r"(?i)(?:^|\s)ERROR(?:\s|:|$)"),
}


def is_known_shutdown_traceback(lines: list[str], traceback_index: int) -> bool:
    before = lines[:traceback_index]
    nearby_before = lines[max(0, traceback_index - 3) : traceback_index]
    after = lines[traceback_index + 1 :]
    return (
        any("AsyncLLM output_handler failed" in line for line in nearby_before)
        and any("[shutdown] API server: shutdown triggered" in line for line in before)
        and any(
            "[shutdown] EngineCore: request processing complete" in line
            for line in before
        )
        and any(
            "EngineDeadError: EngineCore encountered an issue" in line for line in after
        )
        and any("Application shutdown complete" in line for line in after)
    )


def scan(lines: list[str]) -> dict[str, object]:
    fatal: list[dict[str, object]] = []
    known_teardown: list[dict[str, object]] = []
    known_teardown_lines: set[int] = set()
    for index, line in enumerate(lines):
        if "Traceback (most recent call last):" in line and is_known_shutdown_traceback(
            lines, index
        ):
            end = next(
                (
                    candidate
                    for candidate in range(index, len(lines))
                    if "Application shutdown complete" in lines[candidate]
                ),
                len(lines) - 1,
            )
            known_teardown_lines.update(range(max(0, index - 1), end + 1))
    for line_number, line in enumerate(lines, start=1):
        for name, pattern in FATAL_PATTERNS.items():
            if pattern.search(line) and line_number - 1 not in known_teardown_lines:
                fatal.append({"kind": name, "line": line_number, "text": line.rstrip()})

        if "Traceback (most recent call last):" not in line:
            continue
        finding = {"kind": "traceback", "line": line_number, "text": line.rstrip()}
        if is_known_shutdown_traceback(lines, line_number - 1):
            known_teardown.append(finding)
        else:
            fatal.append(finding)

    return {
        "schema_version": 1,
        "status": "failed" if fatal else "passed",
        "fatal_findings": fatal,
        "known_teardown_findings": known_teardown,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    document = scan(args.log.read_text(encoding="utf-8", errors="replace").splitlines())
    rendered = json.dumps(document, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return int(document["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
