#!/usr/bin/env python3
"""Admission, pressure control, and incremental telemetry for quantization."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import NamedTuple


GIB = 1024**3
MIB = 1024**2


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def meminfo() -> dict[str, int]:
    result = {}
    with open("/proc/meminfo") as stream:
        for line in stream:
            key, raw = line.split(":", 1)
            result[key] = int(raw.strip().split()[0]) * 1024
    return result


def zfs_arc_reclaimable(path: Path = Path("/proc/spl/kstat/zfs/arcstats")) -> int:
    """Return ARC bytes above ZFS' non-reclaimable configured minimum.

    Linux does not include the out-of-tree ZFS ARC in ``MemAvailable`` even
    though its shrinker will release ARC down to ``c_min`` under pressure.
    Treating the whole ARC as resident application memory makes admission fail
    on otherwise idle ZFS hosts.  The minimum remains excluded so admission
    never relies on cache ZFS has promised to retain.
    """
    values: dict[str, int] = {}
    try:
        with path.open() as stream:
            for line in stream:
                fields = line.split()
                if len(fields) == 3 and fields[0] in {"size", "c_min"}:
                    values[fields[0]] = int(fields[2])
    except (OSError, ValueError):
        return 0
    if values.keys() != {"size", "c_min"}:
        return 0
    return max(0, values["size"] - values["c_min"])


def _cgroup_value(name: str) -> int | None:
    relative = Path(".")
    try:
        for line in Path("/proc/self/cgroup").read_text().splitlines():
            if line.startswith("0::"):
                relative = Path(line.split("::", 1)[1].lstrip("/"))
                break
    except OSError:
        pass
    path = Path("/sys/fs/cgroup") / relative / name
    try:
        raw = path.read_text().strip()
        return None if raw == "max" else int(raw)
    except (OSError, ValueError):
        return None


def solve_resources(storage: dict, resources: dict, estimates: dict, *, path: Path) -> dict:
    """Resolve aggregate memory/disk bounds or fail before model allocation."""
    memory = meminfo()
    available = memory.get("MemAvailable", 0)
    arc_reclaimable = zfs_arc_reclaimable()
    reserve = int(resources.get("host_mem_available_floor_gib", 24) * GIB)
    user_cap = int(resources.get("memory_max_gib", 76) * GIB)
    cgroup_max = _cgroup_value("memory.max")
    cgroup_current = _cgroup_value("memory.current") or 0
    # Admission may run after model load. Compare total required residency with
    # the total cgroup ceiling, and add already-accounted in-scope residency to
    # MemAvailable headroom so it is not subtracted twice.
    cgroup_budget = cgroup_max if cgroup_max else user_cap
    usable = min(
        user_cap,
        cgroup_budget,
        max(0, available + arc_reclaimable - reserve) + cgroup_current,
    )
    pinned = int(storage.get("pinned_staging_mib", 512) * MIB)
    queues = sum(int(storage.get(key, 512) * MIB) for key in (
        "read_queue_mib", "write_queue_mib", "reorder_queue_mib"
    ))
    required_parts = {
        "model_optimizer": int(estimates.get("model_optimizer_bytes", 0)),
        "activation_frontier": int(estimates.get("activation_frontier_bytes", 0)),
        "live_minibatch": int(estimates.get("live_minibatch_bytes", 0)),
        "queues": queues,
        "pinned_staging": pinned,
        "safety_margin": int(estimates.get("safety_margin_bytes", 4 * GIB)),
    }
    minimum = sum(required_parts.values())
    configured_lru = storage.get("pageable_lru_gib", "auto")
    # "auto" intentionally resolves to zero until trace telemetry demonstrates
    # reuse that beats the kernel page cache without harmful double caching.
    lru = 0 if configured_lru in (None, "auto") else int(configured_lru * GIB)
    if minimum + lru > usable:
        raise RuntimeError(
            f"host admission failed: required={minimum + lru} usable={usable} "
            f"reserve={reserve} parts={required_parts}"
        )
    disk = shutil.disk_usage(path)
    free_reserve = max(
        int(storage.get("min_free_gib", 200) * GIB),
        int(disk.total * storage.get("min_free_percent", 10) / 100),
    )
    disk_required = sum(int(value) for key, value in estimates.items() if key.endswith("_disk_bytes"))
    limit = storage.get("limit_gib")
    ceiling = int(limit * GIB) if limit is not None else max(0, disk.free - free_reserve)
    if disk_required > ceiling:
        raise RuntimeError(
            f"disk admission failed: required={disk_required} available={disk.free} "
            f"reserved={free_reserve} ceiling={ceiling}"
        )
    return {
        "memory": {"available": available, "zfs_arc_reclaimable": arc_reclaimable,
                   "reserve": reserve, "usable": usable,
                   "minimum": minimum, "pageable_lru": lru, "parts": required_parts},
        "disk": {"total": disk.total, "available": disk.free, "reserve": free_reserve,
                 "required_high_water": disk_required, "ceiling": ceiling},
        "effective_fallbacks": [],
    }


class PressureDecision(NamedTuple):
    state: str
    reason: str | None = None


class PressureController:
    """RUN -> DRAIN -> PAUSE -> STOP with recovery hysteresis."""

    def __init__(self, resources: dict):
        self.soft = int(resources.get("host_mem_available_floor_gib", 24) * GIB)
        self.hard = int(resources.get("host_mem_abort_floor_gib", 12) * GIB)
        self.resume = int(resources.get("host_mem_resume_gib", 28) * GIB)
        self.grace = float(resources.get("pressure_grace_seconds", 120))
        self.settle = float(resources.get("resume_settle_seconds", 30))
        self.state = "RUN"
        self.entered = time.monotonic()
        self.recovered_at: float | None = None
        self.transitions: list[dict] = []

    def sample(self, available: int, *, psi_full: float = 0.0, swap_growth: int = 0,
               cgroup_high_delta: int = 0, now: float | None = None) -> PressureDecision:
        now = time.monotonic() if now is None else now
        reason = None
        if available <= self.hard:
            target, reason = "STOP", "hard-memory-floor"
        elif available <= self.soft or psi_full > 0 or swap_growth > 0 or cgroup_high_delta > 0:
            if self.state == "RUN":
                target = "DRAIN"
            elif self.state == "DRAIN" and now - self.entered >= self.grace:
                target = "PAUSE"
            elif self.state == "PAUSE" and now - self.entered >= self.grace:
                target = "STOP"
            else:
                target = self.state
            reason = "soft-pressure"
            self.recovered_at = None
        elif available >= self.resume and self.state != "RUN":
            self.recovered_at = self.recovered_at or now
            target = "RUN" if now - self.recovered_at >= self.settle else self.state
            reason = "hysteresis-recovery"
        else:
            target = self.state
        if target != self.state:
            self.transitions.append({"at_monotonic": now, "from": self.state, "to": target,
                                     "reason": reason, "mem_available": available})
            self.state, self.entered = target, now
        return PressureDecision(self.state, reason)


class JsonlTelemetry:
    """One fsynced JSON record at a time; a crash loses no acknowledged event."""

    def __init__(self, path: Path, *, common: dict | None = None):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.common = common or {}

    def emit(self, event: str, **values) -> None:
        record = {**self.common, "timestamp": time.time(), "event": event, **values}
        with self.path.open("ab", buffering=0) as stream:
            stream.write(_json_line(record))
            os.fsync(stream.fileno())


def _json_line(value: dict) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode() + b"\n"
