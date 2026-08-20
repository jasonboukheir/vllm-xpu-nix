#!/usr/bin/env python3
"""Graft unloaded MTP and vision tensors into a quantized checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

from compressed_tensors.utils.safetensors_load import (
    _fetch_and_save_prefix_tensors,
    get_weight_mappings,
    update_safetensors_index,
)
from safetensors import safe_open


EXTRAS = (
    ("mtp", "model_mtp.safetensors", r"re:^mtp.*"),
    # vLLM instantiates the root vision tower with the runtime prefix
    # ``visual`` even though checkpoint tensors are stored as ``model.visual``.
    ("model.visual", "model_vision.safetensors", r"re:^visual.*"),
)
PROCESSOR_ASSETS = ("preprocessor_config.json",)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json_atomic(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    with temporary.open("rb") as stream:
        os.fsync(stream.fileno())
    temporary.replace(path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def wrapper_config(source: dict, quantized_text: dict) -> dict:
    if not isinstance(source.get("text_config"), dict):
        raise ValueError("source config is not a conditional-generation wrapper")
    quantization = quantized_text.get("quantization_config")
    if not isinstance(quantization, dict):
        raise ValueError("quantized checkpoint has no quantization_config")

    result = dict(source)
    result["quantization_config"] = dict(quantization)
    if version := quantized_text.get("transformers_version"):
        result["transformers_version"] = version
    ignores = list(result["quantization_config"].get("ignore") or [])
    for _, _, pattern in EXTRAS:
        if pattern not in ignores:
            ignores.append(pattern)
    result["quantization_config"]["ignore"] = ignores
    return result


def copy_artifact(source: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    # Hard-link immutable artifact payloads when possible. The new config,
    # index, and graft shards are always separate files.
    shutil.copytree(source, output, copy_function=os.link)


def validate(output: Path, expected: dict[str, set[str]]) -> dict:
    index = read_json(output / "model.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict):
        raise ValueError("output index has no weight_map")

    report: dict[str, object] = {"shards": {}, "prefixes": {}}
    indexed_names = set(weight_map)
    for prefix, shard_name, _ in EXTRAS:
        expected_names = expected[prefix]
        actual_names = {name for name in indexed_names if name.startswith(prefix)}
        if actual_names != expected_names:
            missing = sorted(expected_names - actual_names)
            extra = sorted(actual_names - expected_names)
            raise ValueError(f"{prefix}: index mismatch; missing={missing}, extra={extra}")
        shard = output / shard_name
        with safe_open(shard, framework="pt", device="cpu") as tensors:
            header_names = set(tensors.keys())
            if header_names != expected_names:
                raise ValueError(f"{prefix}: shard header does not match source")
            dtypes = sorted({tensors.get_slice(name).get_dtype() for name in header_names})
        report["prefixes"][prefix] = {
            "count": len(actual_names),
            "dtypes": dtypes,
            "shard": shard_name,
        }

    referenced = set(weight_map.values())
    for shard_name in sorted(referenced):
        shard = output / shard_name
        if not shard.is_file() or shard.stat().st_size == 0:
            raise ValueError(f"missing or empty indexed shard: {shard_name}")
        expected_shard_names = {
            name for name, mapped_shard in weight_map.items() if mapped_shard == shard_name
        }
        with safe_open(shard, framework="pt", device="cpu") as tensors:
            header_names = set(tensors.keys())
        if header_names != expected_shard_names:
            missing = sorted(expected_shard_names - header_names)
            unindexed = sorted(header_names - expected_shard_names)
            raise ValueError(
                f"{shard_name}: index/header mismatch; "
                f"missing={missing}, unindexed={unindexed}"
            )
        report["shards"][shard_name] = {
            "size": shard.stat().st_size,
            "sha256": sha256(shard),
        }
    return report


def graft_in_place(source: Path, output: Path) -> dict:
    """Add required wrapper tensors and metadata to an existing artifact."""
    source_config = read_json(source / "config.json")
    quantized_config = read_json(output / "config.json")

    expected: dict[str, set[str]] = {}
    for prefix, shard_name, _ in EXTRAS:
        tensors = _fetch_and_save_prefix_tensors(
            str(source), prefix, str(output), shard_name
        )
        if not tensors:
            raise ValueError(f"source has no tensors with required prefix {prefix!r}")
        expected[prefix] = set(tensors)
        del tensors

    weight_map = {
        name: os.path.basename(path)
        for name, path in get_weight_mappings(str(output)).items()
    }
    for prefix, shard_name, _ in EXTRAS:
        weight_map.update({name: shard_name for name in expected[prefix]})
    total_size = sum((output / name).stat().st_size for name in set(weight_map.values()))
    update_safetensors_index(str(output), total_size, weight_map)
    write_json_atomic(output / "config.json", wrapper_config(source_config, quantized_config))

    copied_assets: dict[str, dict[str, object]] = {}
    for name in PROCESSOR_ASSETS:
        source_asset = source / name
        if not source_asset.is_file():
            raise ValueError(f"multimodal source is missing required processor asset: {name}")
        destination = output / name
        shutil.copy2(source_asset, destination)
        copied_assets[name] = {
            "size": destination.stat().st_size,
            "sha256": sha256(destination),
        }

    report = validate(output, expected)
    report.update(
        {
            "source": str(source),
            "output": str(output),
            "total_size": total_size,
            "processor_assets": copied_assets,
        }
    )
    write_json_atomic(output / "graft-manifest.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Graft source-precision MTP and vision tensors into a quantized model"
    )
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--artifact", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    source = args.source.resolve()
    artifact = args.artifact.resolve()
    output = args.output.resolve()
    copy_artifact(artifact, output)
    report = graft_in_place(source, output)
    report["artifact"] = str(artifact)
    write_json_atomic(output / "graft-manifest.json", report)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
