#!/usr/bin/env python3
"""Checkpoint tensor inventory and export-gate validation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from safetensors import safe_open


PROTECTED_MARKERS = ("mtp.", ".mtp.", "visual.", ".visual.", "vision", "embed_tokens", "lm_head")
PROCESSOR_ASSETS = ("preprocessor_config.json", "processor_config.json", "tokenizer_config.json")


def tensor_inventory(root: Path) -> dict[str, dict]:
    root = Path(root)
    index_path = root / "model.safetensors.index.json"
    mapping = json.loads(index_path.read_text()).get("weight_map", {}) if index_path.exists() else {}
    shards = sorted(set(mapping.values())) if mapping else sorted(p.name for p in root.glob("*.safetensors"))
    result = {}
    for shard_name in shards:
        shard = root / shard_name
        if not shard.is_file():
            raise ValueError(f"indexed tensor shard is missing: {shard_name}")
        with safe_open(shard, framework="pt", device="cpu") as stream:
            for name in stream.keys():
                view = stream.get_slice(name)
                if name in result:
                    raise ValueError(f"duplicate tensor in checkpoint: {name}")
                result[name] = {"shape": list(view.get_shape()), "dtype": view.get_dtype(), "shard": shard_name}
    if mapping and set(mapping) != set(result):
        raise ValueError(
            f"safetensors index/header mismatch: missing={sorted(set(mapping)-set(result))[:20]} "
            f"unindexed={sorted(set(result)-set(mapping))[:20]}"
        )
    return result


def _protected(name: str) -> bool:
    return any(marker in name for marker in PROTECTED_MARKERS)


def validate_inventory(source: Path, output: Path, *, ignored: Iterable[str] = (),
                       kv_expected: Iterable[str] = ()) -> dict:
    source_tensors = tensor_inventory(source)
    output_tensors = tensor_inventory(output)
    missing = sorted(set(source_tensors) - set(output_tensors))
    protected_missing = [name for name in missing if _protected(name)]
    # Packed quantization legitimately replaces ordinary weight keys. Missing
    # protected tensors never receive that exemption.
    if protected_missing:
        raise ValueError(f"protected tensors disappeared: {protected_missing[:30]}")
    ignored = tuple(ignored)
    ignored_report = {}
    for name, source_meta in source_tensors.items():
        if any(rule == name or (rule.startswith("re:") and __import__("re").search(rule[3:], name)) for rule in ignored):
            actual = output_tensors.get(name)
            if actual is None or actual["dtype"] != source_meta["dtype"] or actual["shape"] != source_meta["shape"]:
                raise ValueError(f"ignored tensor was not preserved in source dtype/shape: {name}")
            ignored_report[name] = actual
    missing_kv = sorted(set(kv_expected) - set(output_tensors))
    if missing_kv:
        raise ValueError(f"calibrated KV scales do not map to output tensors: {missing_kv}")
    source_assets = {name for name in PROCESSOR_ASSETS if (source / name).is_file()}
    missing_assets = sorted(name for name in source_assets if not (output / name).is_file())
    if missing_assets:
        raise ValueError(f"processor assets disappeared: {missing_assets}")
    return {
        "status": "pass", "source_tensor_count": len(source_tensors),
        "output_tensor_count": len(output_tensors), "missing_replaced_tensor_count": len(missing),
        "protected_components": {
            marker: sum(marker in name for name in output_tensors) for marker in PROTECTED_MARKERS
        },
        "ignored_tensors": ignored_report, "kv_scale_tensors": sorted(kv_expected),
        "processor_assets": sorted(source_assets),
    }
