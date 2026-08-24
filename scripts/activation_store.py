#!/usr/bin/env python3
"""Crash-safe, immutable, disk-backed activation corpora.

The format deliberately uses one extent per logical record initially.  This
keeps useful-byte read amplification at 1.0 and gives us a correct baseline
before adding trace-driven coalescing.  Manifests are published last, so an
interrupted writer can never produce a consumable generation.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator, Sequence
import hashlib
import json
import os
from pathlib import Path
import shutil
import struct
import tempfile
import threading
import time
from typing import Any, NamedTuple

import torch


SCHEMA = 1
_HEADER = struct.Struct("<Q")
_PREAD_CHUNK_BYTES = 64 * 1024 * 1024


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic_json(path: Path, value: Any) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(json.dumps(value, indent=2, sort_keys=True).encode() + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_dir(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _encode_tree(value: Any) -> tuple[Any, list[torch.Tensor]]:
    tensors: list[torch.Tensor] = []
    identities: dict[int, int] = {}

    def visit(item: Any) -> Any:
        if isinstance(item, torch.Tensor):
            identity = id(item)
            if identity in identities:
                return {"type": "tensor_ref", "index": identities[identity]}
            tensor = item.detach().cpu()
            # The authoritative representation is contiguous. Original stride
            # and semantic layout remain in metadata for observability/parity.
            packed = tensor.contiguous()
            index = len(tensors)
            identities[identity] = index
            tensors.append(packed)
            return {
                "type": "tensor",
                "index": index,
                "dtype": str(tensor.dtype),
                "shape": list(tensor.shape),
                "stride": list(tensor.stride()),
                "layout": str(tensor.layout),
                "endianness": "little",
                "bytes": packed.numel() * packed.element_size(),
            }
        if isinstance(item, dict):
            return {"type": "dict", "items": [[visit(k), visit(v)] for k, v in item.items()]}
        if isinstance(item, tuple):
            return {"type": "tuple", "items": [visit(child) for child in item]}
        if isinstance(item, list):
            return {"type": "list", "items": [visit(child) for child in item]}
        if item is None or isinstance(item, (bool, str, int, float)):
            return {"type": "scalar", "value": item}
        raise TypeError(f"unsupported activation leaf: {type(item)!r}")

    return visit(value), tensors


def _decode_tree(tree: Any, tensors: list[torch.Tensor]) -> Any:
    kind = tree["type"]
    if kind in {"tensor", "tensor_ref"}:
        return tensors[tree["index"]]
    if kind == "scalar":
        return tree["value"]
    if kind == "dict":
        return {
            _decode_tree(pair[0], tensors): _decode_tree(pair[1], tensors)
            for pair in tree["items"]
        }
    values = [_decode_tree(child, tensors) for child in tree["items"]]
    if kind == "list":
        return values
    if kind == "tuple":
        return tuple(values)
    raise ValueError(f"unknown activation tree node: {kind!r}")


def _serialize(value: Any) -> tuple[bytes, dict]:
    tree, tensors = _encode_tree(value)
    metadata = {"tree": tree, "tensor_count": len(tensors)}
    metadata_bytes = _canonical(metadata)
    parts = [_HEADER.pack(len(metadata_bytes)), metadata_bytes]
    for tensor in tensors:
        raw = tensor.view(torch.uint8).numpy().tobytes()
        parts.append(_HEADER.pack(len(raw)))
        parts.append(raw)
    return b"".join(parts), metadata


def _deserialize(payload: bytes) -> Any:
    view = memoryview(payload)
    position = 0

    def take_size() -> int:
        nonlocal position
        if position + _HEADER.size > len(view):
            raise ValueError("truncated activation extent header")
        size = _HEADER.unpack(view[position : position + _HEADER.size])[0]
        position += _HEADER.size
        return size

    metadata_size = take_size()
    end = position + metadata_size
    if end > len(view):
        raise ValueError("truncated activation extent metadata")
    metadata = json.loads(bytes(view[position:end]))
    position = end
    tensor_nodes: dict[int, dict] = {}

    def find(node: Any) -> None:
        if isinstance(node, dict):
            if node.get("type") == "tensor":
                tensor_nodes[node["index"]] = node
            for child in node.values():
                find(child)
        elif isinstance(node, list):
            for child in node:
                find(child)

    find(metadata["tree"])
    tensors = []
    for index in range(metadata["tensor_count"]):
        size = take_size()
        end = position + size
        if end > len(view):
            raise ValueError("truncated activation tensor")
        node = tensor_nodes[index]
        dtype_name = node["dtype"].removeprefix("torch.")
        dtype = getattr(torch, dtype_name)
        # Clone detaches the returned tensor from the temporary bytes buffer.
        tensor = torch.frombuffer(bytearray(view[position:end]), dtype=dtype).clone()
        tensors.append(tensor.reshape(node["shape"]))
        position = end
    if position != len(view):
        raise ValueError("unexpected trailing activation bytes")
    return _decode_tree(metadata["tree"], tensors)


class CorpusHandle(NamedTuple):
    root: Path
    generation: str
    identity: str

    @property
    def path(self) -> Path:
        return self.root / self.generation


class ActivationStore:
    """Own immutable corpus generations and validate every consumed extent."""

    def __init__(self, root: Path, *, lru_bytes: int = 0):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lru_bytes = max(0, lru_bytes)
        self._cache: OrderedDict[tuple[str, int], tuple[Any, int]] = OrderedDict()
        self._cache_bytes = 0
        self._lock = threading.Lock()
        self.telemetry = {
            "logical_read_bytes": 0, "physical_read_bytes": 0, "pread_calls": 0,
            "logical_write_bytes": 0, "physical_write_bytes": 0,
            "lru_hit_bytes": 0, "lru_miss_bytes": 0, "lru_eviction_bytes": 0,
            "checksum_failures": 0,
        }

    def writer(self, name: str, *, identity: dict | None = None) -> "CorpusWriter":
        return CorpusWriter(self, name, identity or {})

    def open(self, handle: CorpusHandle, *, verify_all: bool = False) -> "DiskBackedBlockIO":
        manifest_path = handle.path / "manifest.json"
        if not manifest_path.is_file():
            raise ValueError(f"activation generation is unpublished: {handle.path}")
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("schema") != SCHEMA:
            raise ValueError("unsupported activation-store schema")
        actual_identity = hashlib.sha256(_canonical(manifest["identity"])).hexdigest()
        if actual_identity != handle.identity:
            raise ValueError("activation corpus identity mismatch")
        corpus = DiskBackedBlockIO(self, handle, manifest)
        if verify_all:
            corpus.verify()
        return corpus

    def _read(self, handle: CorpusHandle, record: dict) -> Any:
        key = (handle.generation, record["index"])
        with self._lock:
            cached = self._cache.pop(key, None)
            if cached is not None:
                self._cache[key] = cached
                self.telemetry["lru_hit_bytes"] += cached[1]
                self.telemetry["logical_read_bytes"] += cached[1]
                return cached[0]
        path = handle.path / record["extent"]
        fd = os.open(path, os.O_RDONLY)
        try:
            length = record["length"]
            offset = record.get("offset", 0)
            if length < 0 or offset < 0 or offset + length > os.fstat(fd).st_size:
                self.telemetry["checksum_failures"] += 1
                raise ValueError(f"truncated activation extent: {path}")
            payload = bytearray(length)
            payload_view = memoryview(payload)
            position = 0
            try:
                while position < length:
                    chunk = os.pread(
                        fd,
                        min(_PREAD_CHUNK_BYTES, length - position),
                        offset + position,
                    )
                    self.telemetry["pread_calls"] += 1
                    self.telemetry["physical_read_bytes"] += len(chunk)
                    if not chunk:
                        break
                    payload_view[position : position + len(chunk)] = chunk
                    position += len(chunk)
            finally:
                payload_view.release()
        finally:
            os.close(fd)
        self.telemetry["logical_read_bytes"] += record["length"]
        self.telemetry["lru_miss_bytes"] += record["length"]
        if position != record["length"]:
            self.telemetry["checksum_failures"] += 1
            raise ValueError(f"truncated activation extent: {path}")
        digest = hashlib.sha256(payload).hexdigest()
        if digest != record["sha256"]:
            self.telemetry["checksum_failures"] += 1
            raise ValueError(f"activation extent checksum mismatch: {path}")
        value = _deserialize(payload)
        if self.lru_bytes and record["length"] <= self.lru_bytes:
            with self._lock:
                self._cache[key] = (value, record["length"])
                self._cache_bytes += record["length"]
                while self._cache_bytes > self.lru_bytes:
                    _, (_, size) = self._cache.popitem(last=False)
                    self._cache_bytes -= size
                    self.telemetry["lru_eviction_bytes"] += size
        return value

    def collect_orphans(self) -> list[str]:
        removed = []
        for path in self.root.glob(".generation-*"):
            if path.is_dir():
                shutil.rmtree(path)
                removed.append(path.name)
        return removed


class CorpusWriter:
    def __init__(self, store: ActivationStore, name: str, identity: dict):
        self.store = store
        self.name = name
        self.identity = identity
        self.temporary = Path(tempfile.mkdtemp(prefix=".generation-", dir=store.root))
        self.records: list[dict] = []
        self.closed = False

    def append(self, value: Any) -> int:
        if self.closed:
            raise RuntimeError("activation corpus writer is closed")
        payload, metadata = _serialize(value)
        index = len(self.records)
        extent_name = f"extent-{index:06d}.bin"
        extent = self.temporary / extent_name
        digest = hashlib.sha256()
        with extent.open("wb") as stream:
            stream.write(payload)
            digest.update(payload)
            stream.flush()
            os.fsync(stream.fileno())
        record = {
            "index": index, "extent": extent_name, "offset": 0,
            "length": len(payload), "sha256": digest.hexdigest(),
            "tree": metadata["tree"],
        }
        self.records.append(record)
        self.store.telemetry["logical_write_bytes"] += len(payload)
        self.store.telemetry["physical_write_bytes"] += len(payload)
        return index

    def commit(self) -> CorpusHandle:
        if self.closed:
            raise RuntimeError("activation corpus writer is closed")
        identity_hash = hashlib.sha256(_canonical(self.identity)).hexdigest()
        manifest = {
            "schema": SCHEMA, "name": self.name, "identity": self.identity,
            "identity_sha256": identity_hash, "created_at": time.time(),
            "record_count": len(self.records), "records": self.records,
            "logical_bytes": sum(record["length"] for record in self.records),
        }
        manifest["manifest_sha256"] = hashlib.sha256(_canonical(manifest)).hexdigest()
        _atomic_json(self.temporary / "manifest.json", manifest)
        generation = f"{self.name}-{manifest['manifest_sha256'][:20]}"
        destination = self.store.root / generation
        if destination.exists():
            shutil.rmtree(self.temporary)
        else:
            os.replace(self.temporary, destination)
            _fsync_dir(self.store.root)
        self.closed = True
        return CorpusHandle(self.store.root, generation, identity_hash)

    def abort(self) -> None:
        if not self.closed:
            shutil.rmtree(self.temporary, ignore_errors=True)
            self.closed = True


class DiskBackedBlockIO(Sequence):
    """Lazy exact-range materialization while preserving requested order."""

    def __init__(self, store: ActivationStore, handle: CorpusHandle, manifest: dict):
        self.store, self.handle, self.manifest = store, handle, manifest

    def __len__(self) -> int:
        return len(self.manifest["records"])

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        return self.store._read(self.handle, self.manifest["records"][index])

    def iter_indices(self, indices: Sequence[int]) -> Iterator[Any]:
        # Delivery order is exactly caller/sampler order. Future readers may
        # reorder physical preads behind this boundary, never these results.
        for index in indices:
            yield self[index]

    def verify(self) -> None:
        for record in self.manifest["records"]:
            self.store._read(self.handle, record)

    def telemetry(self) -> dict:
        logical = self.store.telemetry["logical_read_bytes"]
        physical = self.store.telemetry["physical_read_bytes"]
        return {**self.store.telemetry, "useful_byte_amplification": physical / logical if logical else 0.0}
