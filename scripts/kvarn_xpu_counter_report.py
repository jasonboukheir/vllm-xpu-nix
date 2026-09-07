"""Validate saved unitrace samples and report observed, not inferred, metrics."""

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from scripts.kvarn_factory_run import sha256_file, write_json_atomic
from scripts.kvarn_xpu_profile_sources import verify_sources

METRICS = (
    "GPU_BUSY",
    "XVE_ACTIVE",
    "XVE_STALL",
    "XVE_THREADS_OCCUPANCY_ALL",
    "XVE_SHARED_FUNCTION_ACCESS_HOLD",
    "XVE_INST_EXECUTED_ALU2_ALL_UTILIZATION",
    "LOAD_STORE_CACHE_BYTE_READ",
    "LOAD_STORE_CACHE_BYTE_WRITE",
    "LOAD_STORE_CACHE_HIT",
    "LOAD_STORE_CACHE_ACCESS",
    "L3_HIT",
    "L3_MISS",
    "L3_STALL",
    "GPU_MEMORY_BYTE_READ",
    "GPU_MEMORY_BYTE_WRITE",
    "GPU_MEMORY_BYTE_READ_RATE",
    "GPU_MEMORY_BYTE_WRITE_RATE",
    "GPU_MEMORY_REQUEST_QUEUE_FULL",
)


def read_samples(path):
    with path.open() as stream:
        rows = list(
            csv.DictReader(
                (line for line in stream if line.strip()), skipinitialspace=True
            )
        )
    if not rows:
        raise ValueError("no actual numeric counter samples")
    required = {"Kernel", "GlobalInstanceId", "GpuTime[ns]", "QueryBeginTime[ns]"}
    if not required.issubset(rows[0]):
        raise ValueError("missing counter sample identity/time columns")
    seen = set()
    for row in rows:
        if None in row or any(value is None for value in row.values()):
            raise ValueError("malformed counter row")
        for key, value in row.items():
            if key != "Kernel" and not math.isfinite(float(value)):
                raise ValueError("nonfinite counter value")
        if float(row["GpuTime[ns]"]) <= 0:
            raise ValueError("nonpositive sample duration")
        key = (row["GlobalInstanceId"], row["QueryBeginTime[ns]"])
        if key in seen:
            raise ValueError("duplicate sample within a kernel")
        seen.add(key)
    return rows


def summarize(rows, events, definitions):
    kernels = {
        str(e["args"]["id"]): e
        for e in events
        if e.get("cat") == "gpu_op" and e.get("ph") == "X"
    }
    if not kernels:
        raise ValueError("no GPU kernel timeline")
    grouped = defaultdict(list)
    for row in rows:
        event = kernels.get(row["GlobalInstanceId"])
        if event is None or event["name"] != row["Kernel"]:
            raise ValueError("sample/kernel identity mismatch")
        grouped[row["Kernel"]].append(row)
    columns = {}
    for column in rows[0]:
        name = column.split("[")[0]
        if name in definitions:
            unit = column[len(name) + 1 : -1] if "[" in column else ""
            if unit != {"percent": "%"}.get(
                definitions[name]["units"], definitions[name]["units"]
            ):
                raise ValueError(f"metric unit mismatch: {name}")
            columns[name] = column
    if not {"GPU_BUSY", "XVE_ACTIVE", "XVE_STALL", "GPU_MEMORY_BYTE_READ"}.issubset(
        columns
    ):
        raise ValueError("required compute/stall/memory metric definitions missing")
    reports = []
    for name, samples in sorted(grouped.items()):
        duration = sum(float(row["GpuTime[ns]"]) for row in samples)
        metrics = {}
        for metric in METRICS:
            if metric not in columns:
                continue
            column = columns[metric]
            values = [float(row[column]) for row in samples]
            metrics[metric] = {
                **definitions[metric],
                "sample_min": min(values),
                "sample_max": max(values),
                "sample_mean": sum(values) / len(values),
                "duration_weighted_sample_mean": sum(
                    float(row[column]) * float(row["GpuTime[ns]"]) for row in samples
                )
                / duration,
            }
        reports.append(
            {
                "kernel": name,
                "sample_rows": len(samples),
                "sampled_instances": len({row["GlobalInstanceId"] for row in samples}),
                "traced_instances": sum(
                    event["name"] == name for event in kernels.values()
                ),
                "metrics": metrics,
            }
        )
    return {
        "sample_rows": len(rows),
        "traced_instances": len(kernels),
        "sampled_instances": len({row["GlobalInstanceId"] for row in rows}),
        "unsampled_instance_ids": sorted(
            set(kernels) - {row["GlobalInstanceId"] for row in rows}, key=int
        ),
        "timestamps_assigned_to_multiple_kernels": sum(
            count > 1
            for count in Counter(row["QueryBeginTime[ns]"] for row in rows).values()
        ),
        "reported_uncertainty_values": sorted(
            {float(row["ResultUncertainty[%]"]) for row in rows}
        )
        if "ResultUncertainty[%]" in rows[0]
        else None,
        "metric_definitions": definitions,
        "unsupported_requested_metrics": sorted(set(METRICS) - set(columns)),
        "kernels": reports,
    }


def validate_capture(capture_dir, definitions_path):
    manifest_path = capture_dir / "capture.json"
    manifest = json.loads(manifest_path.read_text())
    inputs = {
        manifest_path: sha256_file(manifest_path),
        definitions_path: sha256_file(definitions_path),
    }
    if not manifest.get("capture_completed") or not manifest.get("files"):
        raise ValueError("capture incomplete")
    for relative, digest in manifest["files"].items():
        path = capture_dir / relative
        if (
            Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not path.resolve().is_relative_to(capture_dir.resolve())
        ):
            raise ValueError("unsafe artifact path")
        if sha256_file(path) != digest:
            raise ValueError(f"capture input changed: {relative}")
        inputs[path] = digest
    verify_sources(capture_dir / "source-snapshot")
    definitions_record = json.loads(definitions_path.read_text())
    if definitions_record.get("returncode") != 0:
        raise ValueError("metric definitions probe failed")
    groups = [
        group
        for device in definitions_record["capabilities"]["devices"]
        if device["is_target"]
        for group in device["metric_groups"]
        if group["name"] == manifest["group"] and group["sampling_type_flags"] & 1
    ]
    if len(groups) != 1 or any(
        groups[0].get(key) != 0
        for key in ("status", "metrics_count_status", "metrics_get_status")
    ):
        raise ValueError("time-based metric definitions missing or ambiguous")
    definitions = {
        metric["name"]: metric
        for metric in groups[0]["metrics"]
        if metric["properties_status"] == 0
    }
    csvs = sorted(capture_dir.glob("collector/*/metrics/*.csv"))
    traces = sorted(capture_dir.glob("collector/*/chrome_trace.json"))
    excluded_traces = []
    device_traces = []
    for path in traces:
        raw = path.read_text()
        if not raw.strip():
            excluded_traces.append(
                {"path": str(path), "reason": "empty launcher trace"}
            )
            continue
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            # unitrace's short-lived `file` child leaves one metadata event without
            # the final array/object delimiters. Accept only that exact shape;
            # never repair or silently drop malformed operational events.
            data = json.loads(raw + "]}")
            if not data["traceEvents"] or any(
                e.get("ph") != "M" for e in data["traceEvents"]
            ):
                raise ValueError("malformed operational trace") from None
            excluded_traces.append(
                {
                    "path": str(path),
                    "reason": "truncated metadata-only auxiliary trace; raw preserved",
                }
            )
            continue
        if any(e.get("cat") == "gpu_op" for e in data["traceEvents"]):
            device_traces.append(path)
        else:
            excluded_traces.append({"path": str(path), "reason": "no GPU operations"})
    traces = device_traces
    if (
        len(csvs) != 1
        or len(traces) != 1
        or any(path not in inputs for path in (*csvs, *traces))
    ):
        raise ValueError("expected one hashed sample stream and one kernel timeline")
    workload = json.loads((capture_dir / "profiled.json").read_text())
    for label in ("before", "after"):
        other = json.loads((capture_dir / f"{label}.json").read_text())
        for key in (
            "workload",
            "contract",
            "iterations",
            "warmup",
            "seed",
            "hardware",
            "sinkhorn_module",
            "workload_source_sha256",
            "torch_version",
        ):
            if other[key] != workload[key]:
                raise ValueError(f"counter bracket mismatch: {key}")
        if not other["correctness_passed"]:
            raise ValueError("counter bracket correctness failed")
    if any(command.get("returncode") != 0 for command in manifest["commands"]):
        raise ValueError("counter capture command failed")
    for tool_key in ("unitrace", "injection_library"):
        identity = manifest[tool_key]
        path = Path(identity["path"])
        if sha256_file(path) != identity["sha256"]:
            raise ValueError(f"counter tool changed: {tool_key}")
        inputs[path] = identity["sha256"]
    result = summarize(
        read_samples(csvs[0]),
        json.loads(traces[0].read_text())["traceEvents"],
        definitions,
    )
    expected = {
        "gemm": {"gemm_kernel": workload["iterations"]},
        "sinkhorn": {
            "_sinkhorn_log_kernel": 2 * workload["iterations"],
            "_sinkhorn_pool_materialize_kernel": workload["iterations"],
        },
    }[workload["workload"]]
    events = [
        e
        for e in json.loads(traces[0].read_text())["traceEvents"]
        if e.get("cat") == "gpu_op" and e.get("ph") == "X"
    ]
    counts = Counter(e["name"].split("[")[0] for e in events)
    if counts != expected or not workload["correctness_passed"]:
        raise ValueError("measured kernel envelope/warmup guard mismatch")
    result.update(
        schema_version=1,
        artifact_kind="validated_xpu_counter_report",
        collection_validated=True,
        diagnostic_only=True,
        workload=workload,
        overhead=manifest["overhead"],
        tool=manifest["unitrace"],
        runtime_environment=manifest["environment"],
        raw_csv=str(csvs[0]),
        raw_trace=str(traces[0]),
        excluded_auxiliary_traces=excluded_traces,
        input_sha256={str(path): digest for path, digest in inputs.items()},
        limitations=[
            "Sampled device counters are not exclusive process counters; other GPU users would contaminate them. Qualification ran serially on the reserved GPU.",
            "Per-kernel metrics summarize assigned sample intervals, which may straddle kernel boundaries; no traffic totals or additive wall-time savings are inferred.",
            "Percentages may exceed 100 due to driver normalization; values are preserved, not clamped.",
            "Byte/event sample means are per sampled interval, not per whole kernel. Weighted means are intended for percentage/rate metrics.",
            "Uncertainty zero is the driver's report, not proof of perfect accuracy. Unsampled short kernels remain explicit.",
            "Only ComputeBasic is qualified here; instruction-level stall-PC sampling and other metric groups are not qualified.",
            "One off/on/off bracket is descriptive, not a statistical overhead guarantee. No bottleneck cause is inferred from counters alone.",
        ],
    )
    for path, digest in inputs.items():
        if sha256_file(path) != digest:
            raise ValueError("input changed during analysis")
    return result


def markdown(report):
    lines = [
        "# B70 hardware-counter validation",
        "",
        f"Actual samples validated: {report['sample_rows']:,} rows; {report['sampled_instances']}/{report['traced_instances']} traced kernel instances sampled.",
        "",
        f"Off/on/off measured-region ratio: {report['overhead']['profiled_over_baseline']:.4f}; baseline drift after/before: {report['overhead']['after_over_before']:.4f}.",
        "",
    ]
    for kernel in report["kernels"]:
        lines += [
            f"## {kernel['kernel']}",
            "",
            f"Coverage: {kernel['sampled_instances']}/{kernel['traced_instances']} instances, {kernel['sample_rows']} rows.",
            "",
            "| Measured metric | Unit | Sample mean | Min | Max |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
        for name, metric in kernel["metrics"].items():
            value = (
                metric["duration_weighted_sample_mean"]
                if metric["units"] in ("%", "percent", "GBpS")
                else metric["sample_mean"]
            )
            lines.append(
                f"| {name} | {metric['units']} | {value:.4f} | {metric['sample_min']:.4f} | {metric['sample_max']:.4f} |"
            )
        lines += [""]
    lines += [
        "Percentage/rate means are duration-weighted; bytes/events are arithmetic per-sample means, not whole-kernel totals.",
        "",
        "## Limitations",
        "",
        *[f"- {item}" for item in report["limitations"]],
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--definitions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.with_suffix(".md").exists():
        raise ValueError("choose fresh report output paths")
    report = validate_capture(args.capture_dir.resolve(), args.definitions.resolve())
    report["analyzer_source_sha256"] = sha256_file(Path(__file__))
    write_json_atomic(args.output, report)
    args.output.with_suffix(".md").write_text(markdown(report))
    print(f"Validated {report['sample_rows']} rows; {args.output}")


if __name__ == "__main__":
    main()
