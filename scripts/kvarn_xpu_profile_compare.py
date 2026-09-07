"""Compare matched diagnostic service traces; never establish performance parity."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from scripts.kvarn_factory_run import (
    ensure_durable_output,
    sha256_file,
    write_json_atomic,
)
from scripts.kvarn_perf_run import ensure_durable
from scripts.kvarn_xpu_profile import load_trace
from scripts.kvarn_xpu_profile_sources import verify_sources
from scripts.kvarn_xpu_trace import analyze_attribution


def family(name: str, kind: str) -> str:
    """Explicit name-based grouping; raw operation/caller tables are retained."""
    if kind != "kernel":
        return kind
    for pattern, label in (
        ("_sinkhorn_log_kernel", "Sinkhorn balancing"),
        ("_sinkhorn_pool_materialize_kernel", "Sinkhorn input staging"),
        ("kvarn_materialize_packed_kv", "Packed-cache materialization"),
        ("KVarNBalancedWriterKernel", "Packed-cache writer"),
        ("Hadamard", "Hadamard kernels"),
        ("cutlass::fmha::", "Dense attention (all variants)"),
    ):
        if pattern in name:
            return label
    return "GEMM" if name == "gemm_kernel" else "Other kernels"


def normalized_service_profile(profile: dict) -> dict:
    """Ignore only the trace output path in the already canonical arm profile."""
    canonical = json.loads(json.dumps(profile["canonical_matched_profile"]))
    argv = canonical["argv"]
    index = argv.index("--profiler-config") + 1
    config = json.loads(argv[index])
    config.pop("torch_profiler_dir", None)
    argv[index] = json.dumps(config, sort_keys=True)
    return canonical


def render_markdown(report: dict) -> str:
    """Compact, explicitly diagnostic view of the authoritative JSON report."""
    analyses = report["analyses"]
    workload = report["matched_workload"]
    lines = [
        "# Matched CPU/XPU profile comparison",
        "",
        "Diagnostic only: not service-performance or parity evidence.",
        "",
        f"Workload: `{json.dumps(workload, sort_keys=True)}`.",
        "",
        f"Selected captured steps: {analyses['reference']['coverage']['selected_step_indices']}.",
        "",
        "| Device duration sum | Auto (ms) | KVarN (ms) | Delta (ms) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in report["family_duration_sums"]:
        lines.append(
            f"| {row['family']} | {row['reference_ms']:.3f} | "
            f"{row['candidate_ms']:.3f} | {row['delta_ms']:+.3f} |"
        )
    lines += ["", "Duration sums may overlap; they are not request wall time.", ""]
    for arm, label in (("reference", "Auto"), ("candidate", "KVarN")):
        analysis = analyses[arm]
        coverage, window = analysis["coverage"], analysis["window"]
        lines += [
            f"## {label}",
            "",
            (
                f"Selected device busy union: {window['device_busy_union_us'] / 1000:.3f} ms. "
                f"All recorded device busy within the selected completion window: "
                f"{window['full_trace_device_busy_union_us'] / 1000:.3f} ms."
            ),
            "",
            (
                f"CPU origins resolved: {coverage['resolved_host_origin_fraction']:.2%}; "
                f"selected operations: {coverage['assigned_device_operations']}; "
                f"excluded guard operations: {coverage['excluded_guard_device_operations']}; "
                f"outside execution scopes: {coverage['resolved_outside_step_scopes']}."
            ),
            "",
            "Largest recorded runtime/driver CPU self-time totals:",
            "",
        ]
        for row in analysis["runtime_driver_by_cpu_caller"][:5]:
            lines.append(
                f"- `{row['name']}` under `{row['cpu_caller']}`: "
                f"{row['cpu_self_us'] / 1000:.3f} ms across {row['count']} calls."
            )
        lines.append("")
    if report.get("profiler_overhead"):
        lines += [
            "## Off/on/off profiler overhead",
            "",
            "Same-process bracketing measurements; one descriptive bracket per arm.",
            "",
            "| Arm | Unprofiled request baseline (ms) | Profiled request (ms) | Ratio | After/before baseline | Text matched |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
        for arm, label in (("reference", "Auto"), ("candidate", "KVarN")):
            overhead = report["profiler_overhead"][arm]
            delta = overhead["deltas"]["median_request_latency_ms"]
            active = overhead["measurements"]["profiled"]["median_request_latency_ms"]
            lines.append(
                f"| {label} | {delta['bracketing_baseline_mean']:.3f} | {active:.3f} | "
                f"{delta['profiled_over_baseline']:.3f} | {delta['after_over_before']:.3f} | "
                f"{overhead['matched_generated_texts']} |"
            )
        lines += [
            "",
            "These request timings include in-request trace stopping/export and must not be used for service-parity claims.",
            "",
        ]
    lines += ["## Interpretation limits", ""]
    lines.extend(f"- {item}" for item in report["limitations"])
    lines.append(
        "- Capture-boundary gaps may contain untraced queued work; guard steps reduce, but do not prove absence of, this effect."
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-run", type=Path, required=True)
    parser.add_argument("--candidate-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path)
    parser.add_argument("--first-step", type=int)
    parser.add_argument("--last-step", type=int)
    parser.add_argument(
        "--allow-legacy-provenance",
        action="store_true",
        help="Analyze historical captures without source snapshots; explicitly unqualified provenance.",
    )
    args = parser.parse_args()
    output = ensure_durable_output(args.output, allow_tmp=False)
    markdown_output = (
        ensure_durable(args.markdown_output, allow_tmp=False)
        if args.markdown_output
        else None
    )
    if markdown_output == output:
        raise ValueError("JSON and Markdown outputs must be different paths")
    for target in (output, markdown_output):
        if target and target.exists():
            raise ValueError(f"output already exists; choose a fresh path: {target}")
    runs = {}
    summaries = {}
    attributed = {}
    service_profiles = {}
    source_snapshots = {}
    hashes = {}
    for path in (Path(__file__), Path(__file__).with_name("kvarn_xpu_trace.py")):
        hashes[str(path.resolve())] = sha256_file(path)
    for arm, directory in (
        ("reference", args.reference_run),
        ("candidate", args.candidate_run),
    ):
        manifest_path = directory / "run.json"
        summary_path = directory / "profile-summary.json"
        service_path = directory / "diagnostic-service-profile.json"
        for path in (manifest_path, summary_path, service_path):
            hashes[str(path.resolve())] = sha256_file(path)
        manifest = json.loads(manifest_path.read_text())
        summary = json.loads(summary_path.read_text())
        if sha256_file(service_path) != manifest["service_profile_sha256"]:
            raise ValueError(f"{arm} service profile changed since capture")
        service_profiles[arm] = normalized_service_profile(
            json.loads(service_path.read_text())
        )
        if manifest["arm"] != arm or manifest["status"] != "valid_diagnostic":
            raise ValueError(f"invalid {arm} capture manifest")
        if manifest.get("source_provenance_qualified") is True:
            snapshot = Path(manifest["source_snapshot"])
            snapshot_manifest = snapshot / "manifest.json"
            if sha256_file(snapshot_manifest) != manifest["source_snapshot_sha256"]:
                raise ValueError(f"{arm} source snapshot manifest changed")
            hashes[str(snapshot_manifest.resolve())] = manifest[
                "source_snapshot_sha256"
            ]
            source_snapshots[arm] = verify_sources(snapshot)["files"]
            for relative, identity in source_snapshots[arm].items():
                hashes[str((snapshot / "files" / relative).resolve())] = identity[
                    "sha256"
                ]
        elif not args.allow_legacy_provenance:
            raise ValueError(
                f"{arm} lacks qualified source snapshot; use --allow-legacy-provenance for historical analysis"
            )
        trace = Path(manifest["kineto_trace"])
        hashes[str(trace.resolve())] = sha256_file(trace)
        if hashes[str(trace.resolve())] != manifest["kineto_trace_sha256"]:
            raise ValueError(f"{arm} trace changed since capture")
        if sha256_file(summary_path) != manifest["profile_summary_sha256"]:
            raise ValueError(f"{arm} summary changed since capture")
        runs[arm], summaries[arm] = manifest, summary
        attributed[arm] = analyze_attribution(
            load_trace(trace), first_step=args.first_step, last_step=args.last_step
        )
    if source_snapshots and len(source_snapshots) != 2:
        raise ValueError("unmatched source provenance qualification")
    if (
        source_snapshots
        and source_snapshots["reference"] != source_snapshots["candidate"]
    ):
        raise ValueError("unmatched profiling harness sources")
    if service_profiles["reference"] != service_profiles["candidate"]:
        raise ValueError(
            "unmatched canonical service profile (model, arguments, or environment)"
        )
    configs = {
        arm: {
            key: value
            for key, value in run["profiler_config"].items()
            if key != "torch_profiler_dir"
        }
        for arm, run in runs.items()
    }
    if configs["reference"] != configs["candidate"]:
        raise ValueError("unmatched capture setting: profiler_config")
    for key in ("workload", "profile_phase", "profile_start_step"):
        if runs["reference"][key] != runs["candidate"][key]:
            raise ValueError(f"unmatched capture setting: {key}")
    overheads = {}
    if bool(runs["reference"].get("profiler_overhead")) != bool(
        runs["candidate"].get("profiler_overhead")
    ):
        raise ValueError("unmatched overhead-bracket preconditioning")
    for arm, run in runs.items():
        if run.get("profiler_overhead"):
            overhead_path = Path(run["profiler_overhead"])
            hashes[str(overhead_path.resolve())] = sha256_file(overhead_path)
            if hashes[str(overhead_path.resolve())] != run["profiler_overhead_sha256"]:
                raise ValueError(f"{arm} overhead report changed since capture")
            overheads[arm] = json.loads(overhead_path.read_text())
            for item in overheads[arm]["inputs"].values():
                path = Path(item["path"]).resolve()
                hashes[str(path)] = sha256_file(path)
                if hashes[str(path)] != item["sha256"]:
                    raise ValueError(
                        f"{arm} overhead input changed since capture: {path}"
                    )
    for key in ("process_package", "candidate_closure_sha256", "device_name"):
        if summaries["reference"][key] != summaries["candidate"][key]:
            raise ValueError(f"unmatched runtime identity: {key}")
    ref, cand = attributed["reference"], attributed["candidate"]
    if [step["annotation"] for step in ref["steps"]] != [
        step["annotation"] for step in cand["steps"]
    ]:
        raise ValueError("captured token/sequence shapes do not match")
    for arm, analysis in attributed.items():
        if analysis["coverage"]["unresolved_host_origins"]:
            raise ValueError(f"{arm} has unresolved CPU/device correlations")
    if (
        ref["coverage"]["outside_step_operation_counts"]
        != cand["coverage"]["outside_step_operation_counts"]
    ):
        raise ValueError("unmatched excluded operations outside execution scopes")
    totals = {}
    for arm, analysis in attributed.items():
        grouped = defaultdict(float)
        for row in analysis["device_operations"]:
            grouped[family(row["device_operation"], row["kind"])] += row[
                "device_duration_sum_us"
            ]
        totals[arm] = grouped
    rows = [
        {
            "family": name,
            "reference_ms": totals["reference"][name] / 1000,
            "candidate_ms": totals["candidate"][name] / 1000,
            "delta_ms": (totals["candidate"][name] - totals["reference"][name]) / 1000,
        }
        for name in set(totals["reference"]) | set(totals["candidate"])
    ]
    rows.sort(key=lambda row: row["delta_ms"], reverse=True)
    for path, digest in hashes.items():
        if sha256_file(Path(path)) != digest:
            raise ValueError(f"input changed during analysis: {path}")
    result = {
        "artifact_kind": "matched_service_profile_comparison",
        "schema_version": 1,
        "diagnostic_only": True,
        "acceptance_eligible": False,
        "matched_workload": runs["reference"]["workload"],
        "profiler_config": runs["reference"]["profiler_config"],
        "profiled_steps": len(ref["steps"]),
        "selected_captured_step_indices": ref["coverage"]["selected_step_indices"],
        "source_hashes": hashes,
        "source_provenance_qualified": len(source_snapshots) == 2,
        "family_duration_sums": rows,
        "analyses": attributed,
        "profiler_overhead": overheads,
        "limitations": [
            *(
                []
                if len(source_snapshots) == 2
                else [
                    "Legacy capture sources were not archived; source provenance is unqualified."
                ]
            ),
            "One matched capture per arm, not a statistical performance comparison.",
            "Family labels are explicit kernel-name grouping, not a hardware bottleneck diagnosis.",
            "CPU wait time overlaps device execution and must not be added to device time.",
            "Only captured model-execution scopes are compared; sampling is outside those scopes.",
            "Other phases, context lengths, batches, and hardware counters require separate captures.",
        ],
    }
    write_json_atomic(output, result)
    if markdown_output:
        markdown_output.parent.mkdir(parents=True, exist_ok=True)
        markdown_output.write_text(render_markdown(result), encoding="utf-8")
    print(f"Matched {len(ref['steps'])} steps. Device duration sums, milliseconds:")
    for row in rows:
        print(
            f"{row['family']}: {row['reference_ms']:.3f} -> {row['candidate_ms']:.3f} ({row['delta_ms']:+.3f})"
        )


if __name__ == "__main__":
    main()
