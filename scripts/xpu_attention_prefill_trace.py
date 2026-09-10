"""Attest the bounded frozen-B70 pure-prefill dispatch change, without a GPU.

Kernel fingerprints are SHA256 of complete names from the frozen 23d8103
FP32-accumulating BF16 D256/Q256/SG32 paged causal specialization. A runtime or
compiler that names kernels differently must be reviewed, not silently accepted.
Device durations are profiler-on diagnostics, not comparative timing results.
"""

import hashlib
import json
from collections import Counter
from pathlib import Path

EXPECTED_KERNEL_SHA256 = {
    "prefill": "bf591ea0068cd054664ffccbff866f20c4daeb50fa59bbaabe985357f5a20498",
    "subtract": "6106cbf3eb17a0bf4fe0ff658be7fdc37dc831d391f65e868aff1e848f3236a6",
    "compare": "0e8901b42f4835680c156dc98d6cf14bd10ea7ba92b17111511780b8c627ab85",
    "decode": "f19cdcd7025e7c9569a6e9888af3585451bac98d78d37f41ebb6d7edc42ecf5e",
}
EXPECTED_CPU_OP = {
    "prefill": "_vllm_fa2_C::varlen_fwd",
    "subtract": "aten::sub",
    "compare": "aten::gt",
    "decode": "_vllm_fa2_C::varlen_fwd",
}
REMOVED = ("subtract", "compare", "decode")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def is_device_event(event):
    category = event.get("cat", "")
    return category == "kernel" or (
        category.startswith(("gpu_", "xpu_"))
        and category not in {"xpu_driver", "xpu_runtime", "gpu_user_annotation"}
    )


def device_signature(event):
    args = event.get("args", {})
    # Deliberately exclude timestamps, correlation IDs and queue addresses.
    return (
        event["cat"],
        event["name"],
        args.get("device"),
        args.get("Bytes"),
    )


def inspect_trace(path):
    path = Path(path)
    raw = path.read_bytes()
    events = json.loads(raw)["traceEvents"]
    devices = [e for e in events if is_device_event(e)]
    kernels = [e for e in devices if e["cat"] == "kernel"]
    prefill = [e for e in kernels if "XeFMHAFwdKernel" in e["name"]]
    if len(prefill) != 1:
        raise ValueError(f"{path}: expected exactly one native prefill event")
    cpu_ops = {}
    for event in events:
        external_id = event.get("args", {}).get("External id")
        if event.get("cat") == "cpu_op" and external_id is not None:
            if external_id in cpu_ops:
                raise ValueError(f"{path}: ambiguous CPU correlation ID")
            cpu_ops[external_id] = event["name"]
    matched = {role: [] for role in EXPECTED_KERNEL_SHA256}
    unrelated = []
    for event in devices:
        fingerprint = sha256(event["name"].encode())
        role = next(
            (
                role
                for role, expected in EXPECTED_KERNEL_SHA256.items()
                if event["cat"] == "kernel" and fingerprint == expected
            ),
            None,
        )
        if role is None:
            unrelated.append(event)
            continue
        if event.get("ph") != "X" or event.get("dur", 0) <= 0:
            raise ValueError(f"{path}: {role} is not a complete device event")
        external_id = event.get("args", {}).get("External id")
        if cpu_ops.get(external_id) != EXPECTED_CPU_OP[role]:
            raise ValueError(f"{path}: {role} has unexpected CPU provenance")
        matched[role].append(event)
    if len(matched["prefill"]) != 1:
        raise ValueError(f"{path}: unexpected prefill specialization")
    return {
        "path": str(path.resolve()),
        "sha256": sha256(raw),
        "device_events": devices,
        "matched": matched,
        "unrelated": unrelated,
    }


def validate_pair(control_path, candidate_path):
    """Return a successful attestation or raise on any dispatch mismatch."""
    control = inspect_trace(control_path)
    candidate = inspect_trace(candidate_path)
    for role in REMOVED:
        if len(control["matched"][role]) != 1:
            raise ValueError(f"control: expected exactly one {role} event")
        if candidate["matched"][role]:
            raise ValueError(f"candidate: {role} event was not removed")
    original_prefill = control["matched"]["prefill"][0]
    pure_prefill = candidate["matched"]["prefill"][0]
    if device_signature(original_prefill) != device_signature(pure_prefill):
        raise ValueError("prefill name or device changed")
    unrelated = Counter(device_signature(e) for e in control["unrelated"])
    if unrelated != Counter(device_signature(e) for e in candidate["unrelated"]):
        raise ValueError("unrelated device work changed")
    return {
        "pass": True,
        "method": __doc__,
        "trace_sha256": {
            "control": control["sha256"],
            "candidate": candidate["sha256"],
        },
        "prefill_name": original_prefill["name"],
        "prefill_name_sha256": EXPECTED_KERNEL_SHA256["prefill"],
        "device_event_counts": {
            "control": len(control["device_events"]),
            "candidate": len(candidate["device_events"]),
        },
        "removed": [
            {
                "role": role,
                "name": control["matched"][role][0]["name"],
                "name_sha256": EXPECTED_KERNEL_SHA256[role],
                "cpu_op": EXPECTED_CPU_OP[role],
                "device_us": control["matched"][role][0]["dur"],
            }
            for role in REMOVED
        ],
        "unrelated_device_events_unchanged": True,
        "unrelated_device_event_count": sum(unrelated.values()),
    }
