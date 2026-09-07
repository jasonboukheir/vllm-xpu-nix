"""Record hardware-metric availability; enumeration is not collection proof."""

import argparse
import json
import os
import subprocess
from pathlib import Path

from scripts.kvarn_factory_run import (
    EXPECTED_DEVICE_NAME,
    ensure_durable_output,
    sha256_file,
    write_json_atomic,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe-binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--stream-group",
        help="Also open/close this time-based group; no workload or value collection.",
    )
    parser.add_argument(
        "--disable-metrics",
        action="store_true",
        help="Control probe: initialize the same runtime without enabling metrics.",
    )
    parser.add_argument(
        "--loader-debug",
        action="store_true",
        help="Retain dynamic-loader library search diagnostics for missing dependencies.",
    )
    args = parser.parse_args()
    if args.stream_group and args.disable_metrics:
        parser.error("--stream-group requires metrics enabled")
    output = ensure_durable_output(args.output, allow_tmp=False)
    if output.exists():
        parser.error("output exists; choose a fresh path")
    binary = args.probe_binary.resolve(strict=True)
    digest = sha256_file(binary)
    overrides = {
        "ZET_ENABLE_METRICS": "0" if args.disable_metrics else "1",
        "ZE_FLAT_DEVICE_HIERARCHY": "FLAT",
    }
    if args.loader_debug:
        overrides["LD_DEBUG"] = "libs"
    command = [str(binary), EXPECTED_DEVICE_NAME]
    if args.stream_group:
        command.append(args.stream_group)
    probe_status = "completed"
    launch_error = None
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
            env={**os.environ, **overrides},
        )
        returncode, stdout, stderr = result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired as error:
        probe_status = "timed_out"
        returncode = None
        # TimeoutExpired can carry bytes even when text=True was requested.
        stdout = error.stdout or ""
        stderr = error.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode("utf-8", errors="replace")
        launch_error = str(error)
    except OSError as error:
        probe_status = "launch_failed"
        returncode, stdout, stderr = None, "", ""
        launch_error = str(error)
    try:
        capabilities = json.loads(stdout)
    except json.JSONDecodeError:
        capabilities = None
    permission = Path("/proc/sys/dev/xe/observation_paranoid")
    report = {
        "artifact_kind": "xpu_metric_capability_probe",
        "schema_version": 1,
        "diagnostic_only": True,
        "collection_validated": False,
        "requested_stream_group": args.stream_group,
        "probe_status": probe_status,
        "launch_error": launch_error,
        "command": command,
        "timeout_seconds": 60,
        "returncode": returncode,
        "capabilities": capabilities,
        "stdout": stdout,
        "stderr": stderr,
        "probe_binary": str(binary),
        "probe_sha256": digest,
        "wrapper_sha256": sha256_file(Path(__file__)),
        "xe_observation_paranoid": permission.read_text().strip()
        if permission.exists()
        else None,
        "environment_overrides": overrides,
        "loader_search_path": os.environ.get("LD_LIBRARY_PATH"),
        "limitations": [
            "No GPU kernels are submitted and no metric values are collected.",
            "Optional stream open/close tests availability, not counter validity.",
            "Metric group enumeration does not prove collection permissions or counter validity.",
            "Zero groups alone does not distinguish missing libraries, runtime support, or permissions.",
            "No system settings are changed.",
        ],
    }
    if sha256_file(binary) != digest:
        raise RuntimeError("probe binary changed during execution")
    write_json_atomic(output, report)
    print(
        json.dumps(
            {
                "output": str(output),
                "probe_status": probe_status,
                "returncode": returncode,
                "collection_validated": False,
                "capabilities": capabilities,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
