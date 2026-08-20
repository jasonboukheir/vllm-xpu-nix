#!/usr/bin/env python3
"""Run a combined AutoRound weight + calibrated FP8 KV-cache recipe."""

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import re
import signal
import sys
import tempfile
import time
from unittest.mock import patch

import torch
from auto_round.calib_dataset import get_dataset
from compressed_tensors.utils import match_named_modules
from huggingface_hub import snapshot_download
from llmcompressor import oneshot
from llmcompressor.modifiers.autoround import AutoRoundModifier as UpstreamAutoRoundModifier
from llmcompressor.modifiers.autoround import base as autoround_base
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.observers import Observer
from llmcompressor.observers.min_max import StaticMinMaxObserver
from llmcompressor.pipelines.sequential import pipeline as sequential_pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from graft_checkpoint_extras import graft_in_place, read_json
from activation_store import ActivationStore, DiskBackedBlockIO
from artifact_integrity import tensor_inventory, validate_inventory
from quantization_resources import JsonlTelemetry, atomic_json, meminfo, solve_resources
from safetensors import safe_open
from safetensors.torch import save_file


def install_trace_watchdog(timeout_seconds: int) -> None:
    """Log and bound llm-compressor's CPU-only torch.fx tracing stage."""
    original = sequential_pipeline.trace_subgraphs

    def traced(*args, **kwargs):
        started = time.monotonic()
        print(
            f"Tracing sequential subgraphs on CPU (timeout: {timeout_seconds}s)...",
            flush=True,
        )

        def timeout_handler(_signum, _frame):
            raise TimeoutError(
                "torch.fx sequential-subgraph tracing exceeded "
                f"{timeout_seconds}s; no quantization blocks were started"
            )

        previous = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout_seconds)
        try:
            result = original(*args, **kwargs)
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, previous)
        print(
            f"Traced {len(result)} sequential subgraphs in "
            f"{time.monotonic() - started:.1f}s",
            flush=True,
        )
        return result

    sequential_pipeline.trace_subgraphs = traced


def canonical_json(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_calibration_data(value) -> str:
    """Hash the exact tokenized calibration inputs without materializing copies."""
    digest = hashlib.sha256()

    def update(item):
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().contiguous()
            digest.update(b"tensor\0")
            digest.update(str(tensor.dtype).encode() + b"\0")
            digest.update(canonical_json(list(tensor.shape)) + b"\0")
            digest.update(tensor.view(torch.uint8).numpy().tobytes())
        elif isinstance(item, dict):
            digest.update(b"dict\0")
            for key in sorted(item):
                digest.update(str(key).encode() + b"\0")
                update(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(b"list\0")
            for child in item:
                update(child)
        elif hasattr(item, "__iter__") and hasattr(item, "__len__"):
            digest.update(f"iterable:{type(item).__name__}\0".encode())
            for child in item:
                update(child)
        elif item is None or isinstance(item, (str, int, float, bool)):
            digest.update(canonical_json(item) + b"\0")
        else:
            raise TypeError(f"unsupported calibration value for hashing: {type(item)!r}")

    update(value)
    return digest.hexdigest()


def cpu_tree(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: cpu_tree(child) for key, child in value.items()}
    if isinstance(value, tuple):
        return tuple(cpu_tree(child) for child in value)
    if isinstance(value, list):
        return [cpu_tree(child) for child in value]
    return value


def tensor_diagnostics(value) -> dict:
    """Streaming-safe numerical summary for nested activation/state trees."""
    finite_count = nan_count = posinf_count = neginf_count = elements = 0
    minimum = maximum = None
    absolute_sum = square_sum = 0.0
    def visit(item):
        nonlocal finite_count, nan_count, posinf_count, neginf_count, elements
        nonlocal minimum, maximum, absolute_sum, square_sum
        if isinstance(item, torch.Tensor):
            tensor = item.detach().float().cpu()
            elements += tensor.numel()
            nan_count += torch.isnan(tensor).sum().item()
            posinf_count += torch.isposinf(tensor).sum().item()
            neginf_count += torch.isneginf(tensor).sum().item()
            finite = tensor[torch.isfinite(tensor)]
            finite_count += finite.numel()
            if finite.numel():
                low, high = finite.min().item(), finite.max().item()
                minimum = low if minimum is None else min(minimum, low)
                maximum = high if maximum is None else max(maximum, high)
                absolute_sum += finite.abs().sum().item()
                square_sum += finite.square().sum().item()
        elif isinstance(item, dict):
            for child in item.values(): visit(child)
        elif isinstance(item, (tuple, list)):
            for child in item: visit(child)
    visit(value)
    return {"elements": elements, "finite": finite_count, "nan": nan_count,
            "posinf": posinf_count, "neginf": neginf_count, "min": minimum,
            "max": maximum, "mean_abs": absolute_sum / finite_count if finite_count else None,
            "rms": (square_sum / finite_count) ** 0.5 if finite_count else None}


class DiskIntermediatesCache:
    """Drop-in llm-compressor frontier backed by immutable store records."""

    store: ActivationStore | None = None
    identity: dict = {}
    diagnostics: JsonlTelemetry | None = None

    def __init__(self, handles=None, offload_device="cpu", onload_device=None):
        self.handles = handles or []
        self.offload_device = torch.device(offload_device) if offload_device else None
        self.onload_device = onload_device

    @classmethod
    def configure(cls, store, identity, diagnostics):
        cls.store, cls.identity, cls.diagnostics = store, identity, diagnostics

    @classmethod
    def from_dataloader(cls, dataloader, model_device=torch.device("cpu"), offload_device=torch.device("cpu")):
        cache = cls(offload_device=offload_device, onload_device=model_device)
        for batch in dataloader:
            cache.append(batch)
        cls.diagnostics.emit("frontier_committed", role="intermediate_frontier", batches=len(cache))
        return cache

    @classmethod
    def empty(cls, num_batches, offload_device):
        cache = cls(offload_device=offload_device)
        for _ in range(num_batches): cache.append({})
        return cache

    def __len__(self):
        return len(self.handles)

    def _write(self, index, values):
        writer = self.store.writer(
            f"frontier-{index:06d}-{time.time_ns()}",
            identity={**self.identity, "role": "intermediate_frontier", "batch": index},
        )
        writer.append(cpu_tree(values))
        new_handle = writer.commit()
        old = self.handles[index] if index < len(self.handles) else None
        if index < len(self.handles): self.handles[index] = new_handle
        else: self.handles.append(new_handle)
        if old is not None:
            import shutil
            shutil.rmtree(old.path)

    def append(self, values):
        self._write(len(self.handles), values)

    def fetch(self, batch_index, input_names=None):
        values = self.store.open(self.handles[batch_index])[0]
        if input_names is not None:
            values = {key: value for key, value in values.items() if key in input_names}
        def onload(value):
            if isinstance(value, torch.Tensor) and self.onload_device is not None:
                return value.to(self.onload_device)
            if isinstance(value, dict): return {k: onload(v) for k, v in value.items()}
            if isinstance(value, tuple): return tuple(onload(v) for v in value)
            if isinstance(value, list): return [onload(v) for v in value]
            return value
        return onload(values)

    def update(self, batch_index, values):
        current = self.store.open(self.handles[batch_index])[0]
        current.update(values)
        self._write(batch_index, current)

    def delete(self, batch_index, consumed_names=None):
        current = self.store.open(self.handles[batch_index])[0]
        for name in list(current) if consumed_names is None else consumed_names:
            current.pop(name, None)
        self._write(batch_index, current)

    def iter(self, input_names=None):
        for index in range(len(self)): yield self.fetch(index, input_names)

    def iter_prefetch(self, input_names=None):
        yield from self.iter(input_names)

    def pin_memory(self, batch_index, input_names=None):
        # Authoritative storage is never pinned. The bounded staging layer owns
        # pinning; synchronous pageable transfer is the safe baseline fallback.
        return None

    def size(self):
        return {torch.device("cpu"): sum(h.path.joinpath("manifest.json").stat().st_size for h in self.handles)}


def collect_kv_scales(model) -> dict[str, torch.Tensor]:
    """Collect calibrated attention scales before Transformers serializes state."""
    scales = {}
    for name, module in model.named_modules():
        scheme = getattr(module, "quantization_scheme", None)
        if (
            scheme is None
            or scheme.weights is not None
            or scheme.input_activations is None
        ):
            continue
        for scale_name in ("k_scale", "v_scale"):
            value = getattr(module, scale_name, None)
            if value is None:
                raise RuntimeError(f"calibrated KV module is missing {scale_name}: {name}")
            value = value.detach().cpu().contiguous()
            if not torch.isfinite(value).all() or not (value > 0).all():
                raise RuntimeError(f"calibrated KV scale is invalid: {name}.{scale_name}")
            scales[f"{name}.{scale_name}"] = value
    if not scales:
        raise RuntimeError("FP8 KV calibration matched no cached-attention modules")
    return scales


def restore_kv_scales(model, scales: dict[str, torch.Tensor]) -> None:
    modules = dict(model.named_modules())
    for key, value in scales.items():
        module_name, scale_name = key.rsplit(".", 1)
        module = modules.get(module_name)
        if module is None:
            raise RuntimeError(f"calibrated KV module disappeared: {module_name}")
        if hasattr(module, scale_name):
            delattr(module, scale_name)
        module.register_parameter(
            scale_name, torch.nn.Parameter(value, requires_grad=False)
        )


def add_kv_scales_to_checkpoint(output: Path, runtime_scales: dict[str, torch.Tensor]) -> list[str]:
    """Ensure every KV scale tensor is indexed exactly once.

    Some Transformers composite models rename checkpoint keys while saving and
    omit newly registered attention parameters from the generated index even
    though they are present in the shard headers.  Prefer those already-written
    tensors; only create a tiny supplemental shard for genuinely missing ones.
    This avoids duplicate tensor names across shards, which makes the artifact
    inventory ambiguous and causes strict loaders to reject it.
    """
    index_path = output / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())
        weight_map = dict(index["weight_map"])
        metadata = dict(index.get("metadata", {}))
    else:
        weight_map = {}
        metadata = {}
        for shard in sorted(output.glob("*.safetensors")):
            if shard.name == "model-kv-scales.safetensors":
                continue
            with safe_open(shard, framework="pt") as stream:
                for key in stream.keys():
                    weight_map[key] = shard.name

    saved_keys = tuple(weight_map)
    header_locations: dict[str, list[str]] = {}
    for shard in sorted(output.glob("*.safetensors")):
        if shard.name == "model-kv-scales.safetensors":
            continue
        with safe_open(shard, framework="pt") as stream:
            for key in stream.keys():
                header_locations.setdefault(key, []).append(shard.name)
    checkpoint_scales = {}
    for runtime_name, value in runtime_scales.items():
        module_name, scale_name = runtime_name.rsplit(".", 1)
        suffix = module_name.removeprefix("model.")
        marker = f"{suffix}.q_proj."
        candidates = sorted(
            {
                key.split(".q_proj.", 1)[0]
                for key in saved_keys
                if marker in key
            }
        )
        if len(candidates) != 1:
            raise RuntimeError(
                f"cannot map KV scale {runtime_name!r} to exactly one checkpoint attention module"
            )
        checkpoint_scales[f"{candidates[0]}.{scale_name}"] = value

    shard_name = "model-kv-scales.safetensors"
    supplemental = {}
    for key, value in checkpoint_scales.items():
        locations = header_locations.get(key, [])
        if len(locations) > 1:
            raise RuntimeError(
                f"KV scale tensor is duplicated across base shards: {key}: {locations}"
            )
        if locations:
            weight_map[key] = locations[0]
        else:
            weight_map[key] = shard_name
            supplemental[key] = value
    if supplemental:
        save_file(supplemental, output / shard_name, metadata={"format": "pt"})
        metadata["total_size"] = int(metadata.get("total_size", 0)) + sum(
            value.numel() * value.element_size() for value in supplemental.values()
        )
    index_path.write_text(
        json.dumps({"metadata": metadata, "weight_map": weight_map}, indent=2, sort_keys=True)
        + "\n"
    )
    return sorted(checkpoint_scales)


class CheckpointDag:
    """Content-addressed block states with an atomic, hash-chained head."""

    schema = 1

    def __init__(self, root: Path, plan: dict, *, resume: bool,
                 activation_store: ActivationStore | None = None):
        self.plan = plan
        self.activation_store = activation_store
        self.run_hash = hashlib.sha256(canonical_json(plan)).hexdigest()
        self.root = root / self.run_hash
        self.nodes = self.root / "nodes"
        self.root.mkdir(parents=True, exist_ok=True)
        self.nodes.mkdir(exist_ok=True)
        self.lock_stream = (self.root / "lock").open("a+b")
        try:
            fcntl.flock(self.lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"checkpoint DAG is already in use: {self.root}") from exc
        plan_path = self.root / "plan.json"
        if plan_path.exists():
            if json.loads(plan_path.read_text()) != plan:
                self.close()
                raise RuntimeError("checkpoint plan hash collision or corrupt plan")
            if not resume:
                self.close()
                raise RuntimeError(
                    f"checkpoint DAG already exists: {self.root}; pass --resume to continue it"
                )
        else:
            self._atomic_json(plan_path, plan)
        head_path = self.root / "head.json"
        self.head = json.loads(head_path.read_text()) if head_path.exists() else {
            "schema": self.schema,
            "run_hash": self.run_hash,
            "completed": [],
            "head": None,
            "q_input_slot": None,
            "q_input_sha256": None,
            "rng_slot": None,
            "rng_sha256": None,
            "q_corpus": None,
            "q_corpus_history": [],
        }
        self._validate_head()
        print(f"Checkpoint DAG: {self.root} ({len(self.head['completed'])} blocks complete)")

    def _atomic_json(self, path: Path, value: dict) -> None:
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        try:
            with os.fdopen(fd, "w") as stream:
                json.dump(value, stream, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            self._fsync_dir(path.parent)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @staticmethod
    def _fsync_dir(path: Path) -> None:
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def close(self) -> None:
        if not self.lock_stream.closed:
            self.lock_stream.close()

    def _validate_head(self) -> None:
        parent = None
        for index, item in enumerate(self.head["completed"]):
            node_path = self.root / item["path"]
            manifest_path = node_path / "node.json"
            if not manifest_path.exists():
                raise RuntimeError(f"checkpoint node is missing: {node_path}")
            node = json.loads(manifest_path.read_text())
            expected = hashlib.sha256(canonical_json({k: v for k, v in node.items() if k != "node_hash"})).hexdigest()
            if node.get("node_hash") != expected or item.get("node_hash") != expected:
                raise RuntimeError(f"checkpoint node hash mismatch: {node_path}")
            if node["index"] != index or node["parent"] != parent:
                raise RuntimeError(f"broken checkpoint chain at: {node_path}")
            state_path = node_path / "state.pt"
            if sha256_file(state_path) != node["state_sha256"]:
                raise RuntimeError(f"checkpoint tensor hash mismatch: {state_path}")
            parent = expected
        if parent != self.head["head"]:
            raise RuntimeError("checkpoint head does not match its node chain")
        for kind in ("q_input", "rng"):
            slot = self.head.get(f"{kind}_slot")
            digest = self.head.get(f"{kind}_sha256")
            if slot and sha256_file(self.root / slot) != digest:
                raise RuntimeError(f"checkpoint {kind} snapshot hash mismatch")
        if self.head.get("q_corpus") and self.activation_store is not None:
            from activation_store import CorpusHandle
            self.activation_store.open(CorpusHandle(**self.head["q_corpus"]), verify_all=True)

    def completed_layer(self, index: int, layer_name: str) -> bool:
        completed = self.head["completed"]
        if index >= len(completed):
            return False
        if completed[index]["layer"] != layer_name:
            raise RuntimeError(
                f"checkpoint layer order differs at {index}: expected "
                f"{completed[index]['layer']!r}, loaded {layer_name!r}"
            )
        return True

    def restore(self, index: int, layer_name: str, module) -> object:
        item = self.head["completed"][index]
        node_path = self.root / item["path"]
        state = torch.load(node_path / "state.pt", map_location="cpu", weights_only=True)
        module.load_state_dict(state, strict=True)
        q_input = None
        if index == len(self.head["completed"]) - 1:
            if self.head.get("q_corpus") and self.activation_store is not None:
                from activation_store import CorpusHandle
                q_input = self.activation_store.open(CorpusHandle(**self.head["q_corpus"]), verify_all=True)[0]
            else:
                q_input = torch.load(
                    self.root / self.head["q_input_slot"], map_location="cpu", weights_only=True
                )
            rng = torch.load(
                self.root / self.head["rng_slot"], map_location="cpu", weights_only=False
            )
            random.setstate(rng["python"])
            torch.set_rng_state(rng["torch"])
            if torch.xpu.is_available() and rng.get("xpu"):
                torch.xpu.set_rng_state_all(rng["xpu"])
        print(f"Restored checkpoint block {index}: {layer_name}")
        return q_input

    def commit(self, index: int, layer_name: str, module, q_input) -> None:
        if index != len(self.head["completed"]):
            raise RuntimeError("checkpoint commits must be contiguous")
        temporary = Path(tempfile.mkdtemp(prefix=f".block-{index:04d}.", dir=self.nodes))
        state_path = temporary / "state.pt"
        torch.save(cpu_tree(module.state_dict()), state_path)
        with state_path.open("rb") as stream:
            os.fsync(stream.fileno())
        state_digest = sha256_file(state_path)
        node = {
            "schema": self.schema,
            "run_hash": self.run_hash,
            "index": index,
            "layer": layer_name,
            "parent": self.head["head"],
            "state_sha256": state_digest,
        }
        node_hash = hashlib.sha256(canonical_json(node)).hexdigest()
        node["node_hash"] = node_hash
        self._atomic_json(temporary / "node.json", node)
        node_name = f"{index:04d}-{node_hash[:16]}"
        node_path = self.nodes / node_name
        os.replace(temporary, node_path)
        self._fsync_dir(self.nodes)

        slot_number = index % 2
        q_slot = f"q-input-{slot_number}.pt"
        rng_slot = f"rng-{slot_number}.pt"
        q_corpus = None
        if self.activation_store is not None:
            writer = self.activation_store.writer(
                f"q-{self.run_hash[:12]}-{index:04d}",
                identity={"run_hash": self.run_hash, "block": index, "layer": layer_name,
                          "role": "propagated_q_input"},
            )
            writer.append(cpu_tree(q_input))
            q_corpus = writer.commit()._asdict()
        else:
            self._atomic_torch(self.root / q_slot, cpu_tree(q_input))
        rng = {
            "python": random.getstate(),
            "torch": torch.get_rng_state(),
            "xpu": torch.xpu.get_rng_state_all() if torch.xpu.is_available() else [],
        }
        self._atomic_torch(self.root / rng_slot, cpu_tree(rng))
        self.head["completed"].append({
            "index": index,
            "layer": layer_name,
            "node_hash": node_hash,
            "path": str(node_path.relative_to(self.root)),
        })
        q_history = list(self.head.get("q_corpus_history", []))
        if q_corpus:
            q_history.append(q_corpus)
        obsolete_q = q_history[:-2]
        q_history = q_history[-2:]
        self.head.update(
            head=node_hash,
            q_input_slot=None if q_corpus else q_slot,
            q_input_sha256=None if q_corpus else sha256_file(self.root / q_slot),
            q_corpus=q_corpus,
            q_corpus_history=q_history,
            rng_slot=rng_slot,
            rng_sha256=sha256_file(self.root / rng_slot),
        )
        self._atomic_json(self.root / "head.json", self.head)
        # Only after the new head is durable may generations older than the two
        # independently resumable q roots be reclaimed.
        for obsolete in obsolete_q:
            generation = obsolete["generation"]
            if Path(generation).name != generation:
                raise RuntimeError("invalid q corpus generation in checkpoint head")
            path = self.activation_store.root / generation
            if path.is_dir():
                import shutil
                shutil.rmtree(path)
                self._fsync_dir(path.parent)
        print(f"Committed checkpoint block {index}: {layer_name} ({node_hash[:16]})")

    def _atomic_torch(self, path: Path, value) -> None:
        fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        os.close(fd)
        try:
            torch.save(value, temporary)
            with open(temporary, "rb") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            self._fsync_dir(path.parent)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)


class AutoRoundModifier(UpstreamAutoRoundModifier):
    """Memory-bounded AutoRound with resumable sequential block boundaries.

    llm-compressor 0.13 captures decoder inputs on their execution device and
    later moves the complete calibration corpus to that device before invoking
    AutoRound.  That makes ``batch_size`` ineffective as a memory bound.  Keep
    the corpus on CPU and enable AutoRound's supported low-GPU-memory mode so it
    only onloads the minibatch being optimized.
    """

    low_gpu_mem_usage: bool = True

    def attach_activation_store(self, store: ActivationStore, identity: dict,
                                telemetry: JsonlTelemetry) -> None:
        object.__setattr__(self, "_activation_store", store)
        object.__setattr__(self, "_activation_identity", identity)
        object.__setattr__(self, "_diagnostics", telemetry)
        object.__setattr__(self, "_stored_corpora", {})

    def input_capture_hook(self, module, args, kwargs):
        if isinstance(self._all_module_input.get(module._tmp_name), DiskBackedBlockIO):
            # AutoRound's internal reference/prediction forwards re-enter the
            # llm-compressor hook after the authoritative capture is published.
            # The legacy list accumulated these unused duplicates; immutable
            # storage must neither mutate nor duplicate them.
            return
        if module._tmp_name not in self._all_module_input:
            self._all_module_input[module._tmp_name] = []
        self._all_module_input[module._tmp_name].append(
            (cpu_tree(args), cpu_tree(kwargs))
        )

    @staticmethod
    def _move_inputs_to(inputs, _device):
        """Leave the full corpus on CPU; AutoRound streams its minibatches."""
        return inputs

    def apply_autoround(self, state, modules):
        store = getattr(self, "_activation_store", None)
        if store is not None:
            # Capture hooks have finished and live model arguments are gone. Spill
            # retained snapshots now, then replace the list with a lazy Sequence.
            for name, values in list(self._all_module_input.items()):
                if isinstance(values, DiskBackedBlockIO):
                    continue
                writer = store.writer(
                    f"fp-{name.replace('.', '-')}",
                    identity={**self._activation_identity, "role": "fp_input", "module": name},
                )
                for value in values:
                    writer.append(value)
                handle = writer.commit()
                corpus = store.open(handle)
                self._all_module_input[name] = corpus
                self._stored_corpora[name] = handle._asdict()
                self._diagnostics.emit(
                    "activation_corpus_committed", component="decoder", layer=name,
                    role="fp_input", samples=len(corpus), logical_bytes=corpus.manifest["logical_bytes"],
                )
        original_autoround = autoround_base.AutoRound

        def memory_bounded_autoround(*args, **kwargs):
            kwargs["low_gpu_mem_usage"] = self.low_gpu_mem_usage
            return original_autoround(*args, **kwargs)

        # AutoRound is imported into llm-compressor's modifier module. Patch
        # only for this synchronous call, keeping the workaround tightly scoped
        # to the pinned 0.13 API until upstream exposes the option itself.
        with patch.object(autoround_base, "AutoRound", memory_bounded_autoround):
            return super().apply_autoround(state, modules)

    def _build_layer_config_for_autoround(self, wrapped_model) -> dict[str, dict]:
        """Ignore KV-only schemes when translating weight overrides to AutoRound.

        QuantizationModifier attaches an output-activation-only scheme to attention
        modules for calibrated KV caches. llm-compressor 0.13 (and current main as
        of 2026-08) assumes every scheme encountered here has weight arguments.
        Those parent attention schemes are not AutoRound weight targets and must
        remain on the model for KV scale export, but must not be translated.
        """
        default_scheme = self._get_default_quant_scheme()
        default_config = self._quant_scheme_to_autoround_config(default_scheme)
        layer_config = {}
        for name, module in wrapped_model.named_modules():
            quant_scheme = getattr(module, "quantization_scheme", None)
            if quant_scheme is None or quant_scheme.weights is None:
                continue
            layer_scheme = self._quant_scheme_to_autoround_config(quant_scheme)
            if layer_scheme != default_config:
                layer_config[name] = layer_scheme
        return layer_config

    def _postprocess_qparams(self, model, llmc_registered_qparams):
        """Keep calibrated KV metadata while AutoRound replaces weight metadata.

        llm-compressor 0.13's AutoRound cleanup treats every non-AutoRound module
        as an unquantized weight layer.  Attention parents carry a KV-only scheme,
        so that cleanup otherwise deletes their scheme and calibrated k/v scales.
        Preserve only those KV-only attributes around the upstream cleanup.
        """
        kv_metadata = []
        for module in model.modules():
            scheme = getattr(module, "quantization_scheme", None)
            if (
                scheme is None
                or scheme.weights is not None
                or scheme.input_activations is None
            ):
                continue
            qparams = {
                name: getattr(module, name)
                for name in ("k_scale", "v_scale")
                if hasattr(module, name)
            }
            kv_metadata.append((module, scheme, qparams))

        super()._postprocess_qparams(model, llmc_registered_qparams)

        for module, scheme, qparams in kv_metadata:
            module.quantization_scheme = scheme
            for name, value in qparams.items():
                if hasattr(module, name):
                    delattr(module, name)
                if isinstance(value, torch.nn.Parameter):
                    module.register_parameter(name, value)
                else:
                    module.register_buffer(name, value)

    def attach_checkpoint_dag(self, dag: CheckpointDag) -> None:
        object.__setattr__(self, "_checkpoint_dag", dag)
        object.__setattr__(self, "_checkpoint_index", 0)

    def on_sequential_epoch_end(self, state, event, modules, **kwargs):
        dag = getattr(self, "_checkpoint_dag", None)
        diagnostics = getattr(self, "_diagnostics", None)
        started = time.monotonic()
        if dag is None:
            result = super().on_sequential_epoch_end(state, event, modules, **kwargs)
            if diagnostics is not None:
                for module in modules or []:
                    if self._is_decoding_layer(module):
                        diagnostics.emit("block_complete", component="decoder",
                                         layer=module._tmp_name, wall_seconds=time.monotonic() - started,
                                         storage=getattr(self, "_activation_store").telemetry)
            return result
        decoding_layers = [module for module in (modules or []) if self._is_decoding_layer(module)]
        if not decoding_layers:
            return super().on_sequential_epoch_end(state, event, modules, **kwargs)
        if len(decoding_layers) != 1:
            raise RuntimeError("checkpointing requires exactly one decoder block per subgraph")
        module = decoding_layers[0]
        index = self._checkpoint_index
        layer_name = module._tmp_name
        if dag.completed_layer(index, layer_name):
            restored_q_input = dag.restore(index, layer_name, module)
            if restored_q_input is not None:
                self._q_input = restored_q_input
            self.post_autoround_cleanup()
        else:
            super().on_sequential_epoch_end(state, event, modules, **kwargs)
            dag.commit(index, layer_name, module, self._q_input)
        object.__setattr__(self, "_checkpoint_index", index + 1)
        if diagnostics is not None:
            diagnostics.emit("block_complete", component="decoder", layer=layer_name,
                             block=index, wall_seconds=time.monotonic() - started,
                             propagated_q=tensor_diagnostics(self._q_input),
                             storage=getattr(self, "_activation_store", None).telemetry)


@Observer.register("audit_static_minmax")
class AuditStaticMinMaxObserver(StaticMinMaxObserver):
    """Accumulating min/max plus a bounded deterministic value reservoir."""

    reservoir_per_observation = 4096
    reservoir_limit = 262144

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.audit_observations = 0
        self.audit_elements = 0
        self.audit_nan = 0
        self.audit_posinf = 0
        self.audit_neginf = 0
        self.audit_reservoir = []

    def update_statistics_from_observed(self, observed: torch.Tensor) -> None:
        super().update_statistics_from_observed(observed)
        detached = observed.detach().reshape(-1)
        self.audit_observations += 1
        self.audit_elements += detached.numel()
        self.audit_nan += torch.isnan(detached).sum().item()
        self.audit_posinf += torch.isposinf(detached).sum().item()
        self.audit_neginf += torch.isneginf(detached).sum().item()
        remaining = self.reservoir_limit - sum(x.numel() for x in self.audit_reservoir)
        if remaining <= 0 or detached.numel() == 0:
            return
        count = min(self.reservoir_per_observation, remaining, detached.numel())
        indices = torch.linspace(
            0, detached.numel() - 1, steps=count, device=detached.device
        ).long()
        self.audit_reservoir.append(detached.index_select(0, indices).float().cpu())


def collect_kv_observer_audit(
    model, *, observation_limit: int | None = None
) -> tuple[dict, dict[str, torch.Tensor]]:
    report = {}
    reservoirs = {}
    for module_name, module in model.named_modules():
        for kind in ("k", "v"):
            observer = getattr(module, f"{kind}_observer", None)
            if not isinstance(observer, AuditStaticMinMaxObserver):
                continue
            if not observer.has_statistics or observer.audit_observations == 0:
                continue
            chunks = observer.audit_reservoir
            if observation_limit is not None:
                chunks = chunks[:observation_limit]
            samples = torch.cat(chunks) if chunks else torch.empty(0)
            finite = samples[torch.isfinite(samples)]
            percentiles = None
            if finite.numel():
                values = torch.quantile(
                    finite, torch.tensor([0.0, 0.001, 0.01, 0.5, 0.99, 0.999, 1.0])
                ).tolist()
                percentiles = dict(zip(("p0", "p0_1", "p1", "p50", "p99", "p99_9", "p100"), values))
            key = f"{module_name}.{kind}"
            reservoirs[key] = samples
            report[key] = {
                "observation_count": min(observer.audit_observations, observation_limit)
                if observation_limit is not None else observer.audit_observations,
                "raw_observation_count": observer.audit_observations,
                "element_count": observer.audit_elements,
                "min": observer.min_vals.detach().float().cpu().reshape(-1).tolist(),
                "max": observer.max_vals.detach().float().cpu().reshape(-1).tolist(),
                "nan": observer.audit_nan,
                "posinf": observer.audit_posinf,
                "neginf": observer.audit_neginf,
                "reservoir_elements": samples.numel(),
                "percentiles": percentiles,
            }
    return report, reservoirs


class CalibratedKVModifier(QuantizationModifier):
    """Capture post-RoPE K and cache-ready V despite sequential ``use_cache=False``.

    llm-compressor's sequential pipeline intentionally removes the cache branch
    from its traced graph.  Consequently its normal QuantizedKVCache hooks never
    execute for this model.  Qwen3.5 projection hooks *do* remain in the graph, so
    reconstruct the exact cache tensors there: K projection -> K norm -> text
    RoPE, and V projection -> cache layout.  The hook feeds the ordinary K/V
    observers, retaining the same quantization lifecycle and serialized schema.
    Unsupported attention implementations fail closed instead of silently
    producing empty calibration data.
    """

    capture_samples: int = 1
    capture_batch_size: int = 1

    def on_initialize(self, state, **kwargs):
        result = super().on_initialize(state, **kwargs)
        object.__setattr__(self, "_capture_enabled", False)
        # These are declared Pydantic fields so the values survive recipe
        # serialization/cloning performed by llm-compressor.
        object.__setattr__(
            self,
            "_capture_observation_limit",
            math.ceil(self.capture_samples / self.capture_batch_size),
        )
        handles = []
        captured_modules = []

        for module_name, module in state.model.named_modules():
            if not (hasattr(module, "k_proj") and hasattr(module, "v_proj")):
                continue
            if not (hasattr(module, "k_norm") and hasattr(module, "head_dim")):
                continue
            config = getattr(module, "config", None)
            rope = getattr(config, "rope_parameters", None)
            num_kv_heads = getattr(config, "num_key_value_heads", None)
            if not isinstance(rope, dict) or num_kv_heads is None:
                continue
            if rope.get("rope_type", "default") != "default":
                raise RuntimeError(
                    f"post-RoPE KV capture does not support rope_type={rope.get('rope_type')!r} "
                    f"for {module_name}"
                )

            def capture_k(_projection, _inputs, output, *, parent=module):
                if not self._capture_enabled:
                    return
                if not hasattr(parent, "k_observer"):
                    raise RuntimeError("K observer disappeared during KV capture")
                batch, sequence, _ = output.shape
                head_dim = int(parent.head_dim)
                heads = int(parent.config.num_key_value_heads)
                key = parent.k_norm(output.view(batch, sequence, heads, head_dim)).transpose(1, 2)
                rope_parameters = parent.config.rope_parameters
                rotary_dim = int(head_dim * rope_parameters.get("partial_rotary_factor", 1.0))
                base = float(rope_parameters["rope_theta"])
                inv_freq = 1.0 / (
                    base
                    ** (
                        torch.arange(0, rotary_dim, 2, device=key.device, dtype=torch.float32)
                        / rotary_dim
                    )
                )
                positions = torch.arange(sequence, device=key.device, dtype=torch.float32)
                freqs = torch.outer(positions, inv_freq)
                emb = torch.cat((freqs, freqs), dim=-1)
                cos = emb.cos().to(key.dtype)[None, None, :, :]
                sin = emb.sin().to(key.dtype)[None, None, :, :]
                key_rot, key_pass = key[..., :rotary_dim], key[..., rotary_dim:]
                half = rotary_dim // 2
                rotated_half = torch.cat((-key_rot[..., half:], key_rot[..., :half]), dim=-1)
                cache_key = torch.cat((key_rot * cos + rotated_half * sin, key_pass), dim=-1)
                parent.k_observer(cache_key)

            def capture_v(_projection, _inputs, output, *, parent=module):
                if not self._capture_enabled:
                    return
                if not hasattr(parent, "v_observer"):
                    raise RuntimeError("V observer disappeared during KV capture")
                batch, sequence, _ = output.shape
                value = output.view(
                    batch,
                    sequence,
                    int(parent.config.num_key_value_heads),
                    int(parent.head_dim),
                ).transpose(1, 2)
                parent.v_observer(value)

            handles.append(module.k_proj.register_forward_hook(capture_k))
            handles.append(module.v_proj.register_forward_hook(capture_v))
            captured_modules.append(module_name)

        if not captured_modules:
            raise RuntimeError("no supported full-attention modules found for post-RoPE KV capture")
        object.__setattr__(self, "_capture_hook_handles", handles)
        object.__setattr__(self, "captured_modules", captured_modules)
        return result

    def on_calibration_start(self, state, event, **kwargs):
        result = super().on_calibration_start(state, event, **kwargs)
        object.__setattr__(self, "_capture_enabled", True)
        return result

    def on_calibration_end(self, state, event, **kwargs):
        audit, reservoirs = collect_kv_observer_audit(
            state.model, observation_limit=self._capture_observation_limit
        )
        object.__setattr__(self, "_capture_enabled", False)
        result = super().on_calibration_end(state, event, **kwargs)
        object.__setattr__(self, "calibration_audit", audit)
        object.__setattr__(self, "calibration_reservoirs", reservoirs)
        return result

    def on_finalize(self, state, **kwargs):
        for handle in getattr(self, "_capture_hook_handles", []):
            handle.remove()
        return super().on_finalize(state, **kwargs)


def calibrated_kv_cache_scheme() -> dict:
    """Static FP8 KV scheme whose statistics span the full calibration corpus."""
    return {
        "num_bits": 8,
        "type": "float",
        "strategy": "tensor",
        "dynamic": False,
        "symmetric": True,
        # compressed-tensors otherwise defaults static quantization to
        # memoryless_minmax, which replaces its range on every minibatch and
        # makes a 512-sample run depend only on the final batch.
        "observer": "audit_static_minmax",
    }


def snapshot_weight_quantization(model: torch.nn.Module) -> dict[str, dict]:
    """Capture the exact W4 modules before a KV-only calibration lifecycle.

    QuantizationModifier defaults to a ``Linear`` config group even when it was
    constructed only with ``kv_cache_scheme``.  Applying that empty group after
    AutoRound replaces every Linear's W4 scheme, producing a deceptively valid
    but almost entirely BF16 checkpoint.  Keep an explicit behavioral contract
    for the modules which must remain weight-quantized.
    """
    snapshot = {}
    for name, module in model.named_modules(remove_duplicate=True):
        scheme = getattr(module, "quantization_scheme", None)
        if scheme is None or scheme.weights is None:
            continue
        qparams = tuple(
            attr
            for attr in ("weight_scale", "weight_zero_point", "weight_g_idx")
            if hasattr(module, attr)
        )
        snapshot[name] = {
            "module": module,
            "scheme": scheme,
            "status": getattr(module, "quantization_status", None),
            "qparams": qparams,
        }
    if not snapshot:
        raise RuntimeError("AutoRound produced no weight-quantized modules")
    return snapshot


def restore_and_validate_weight_quantization(
    model: torch.nn.Module, snapshot: dict[str, dict]
) -> dict:
    """Restore W4 metadata and prove every original target is still compressible."""
    current = dict(model.named_modules(remove_duplicate=True))
    missing_modules = sorted(set(snapshot) - set(current))
    if missing_modules:
        raise RuntimeError(
            f"post-W4 KV pass removed weight modules: {missing_modules[:20]}"
        )

    repaired = []
    missing_qparams = []
    for name, saved in snapshot.items():
        module = current[name]
        scheme = getattr(module, "quantization_scheme", None)
        if scheme is not saved["scheme"]:
            repaired.append(name)
        module.quantization_scheme = saved["scheme"]
        if saved["status"] is not None:
            module.quantization_status = saved["status"]
        for attr in saved["qparams"]:
            if not hasattr(module, attr):
                missing_qparams.append(f"{name}.{attr}")

    actual = {
        name
        for name, module in model.named_modules(remove_duplicate=True)
        if (
            (scheme := getattr(module, "quantization_scheme", None)) is not None
            and scheme.weights is not None
        )
    }
    expected = set(snapshot)
    if missing_qparams or actual != expected:
        raise RuntimeError(
            "post-W4 KV pass damaged weight quantization: "
            f"missing_qparams={missing_qparams[:20]} "
            f"missing_modules={sorted(expected - actual)[:20]} "
            f"unexpected_modules={sorted(actual - expected)[:20]}"
        )
    return {
        "status": "pass",
        "expected_weight_modules": len(expected),
        "actual_weight_modules": len(actual),
        "metadata_repaired_modules": repaired,
        "module_names": sorted(expected),
    }


def validate_saved_weight_compression(
    output: Path, expected_weight_modules: int
) -> dict:
    """Fail unless each in-memory W4 target became a packed checkpoint tensor."""
    inventory = tensor_inventory(output)
    suffix_counts = {
        suffix: sum(name.endswith(f".{suffix}") for name in inventory)
        for suffix in ("weight_packed", "weight_scale", "weight_shape")
    }
    config = read_json(output / "config.json").get("quantization_config", {})
    groups = config.get("config_groups", {}) if isinstance(config, dict) else {}
    w4_groups = [
        group
        for group in groups.values()
        if isinstance(group, dict)
        and isinstance(group.get("weights"), dict)
        and group["weights"].get("num_bits") == 4
        and group.get("format") == "pack-quantized"
    ]
    failures = []
    for suffix, count in suffix_counts.items():
        if count != expected_weight_modules:
            failures.append(
                f"{suffix}: expected {expected_weight_modules}, observed {count}"
            )
    if not w4_groups:
        failures.append("saved quantization_config has no packed 4-bit weight group")
    if failures:
        raise RuntimeError("saved W4 compression coverage failed: " + "; ".join(failures))
    return {
        "status": "pass",
        "expected_weight_modules": expected_weight_modules,
        "packed_tensor_counts": suffix_counts,
        "w4_config_group_count": len(w4_groups),
        "checkpoint_bytes": sum(
            path.stat().st_size for path in output.glob("*.safetensors")
        ),
    }


def package_versions() -> dict:
    versions = {}
    for package in ("auto-round", "llmcompressor", "compressed-tensors", "transformers", "torch"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "unknown"
    return versions


def reference_identity(args, dataset, tokenizer, resolved_ignores) -> dict:
    tokenizer_state = {
        "class": tokenizer.__class__.__qualname__,
        "name_or_path": tokenizer.name_or_path,
        "vocab_size": len(tokenizer),
        "special_tokens": tokenizer.special_tokens_map,
        "chat_template": getattr(tokenizer, "chat_template", None),
    }
    return {
        "schema": 1,
        "role": "bf16_kv_reference",
        "source": {"repo": args.model, "revision": args.revision},
        "corpus": {
            "dataset": args.dataset,
            "exact_inputs_sha256": hash_calibration_data(dataset),
            "seed": args.seed,
            "samples": args.samples,
            "sequence_length": args.seqlen,
            "shuffle_calibration_samples": False,
            "collator": "truncation",
        },
        "tokenizer_sha256": hashlib.sha256(canonical_json(tokenizer_state)).hexdigest(),
        "tokenizer": tokenizer_state,
        "resolved_ignore": resolved_ignores,
        "dtype": getattr(args, "model_dtype", "unknown"),
        "capture": {"observer": "audit_static_minmax", "reservoir_limit": AuditStaticMinMaxObserver.reservoir_limit},
        "packages": package_versions(),
        "toolchain": {
            "nix_closure": os.environ.get("QUANTIZE_TOOLCHAIN_ID", "unknown"),
            "runner_sha256": sha256_file(Path(__file__)),
            "workspace_lock_sha256": args.workspace_lock_sha256,
        },
    }


def json_handle(handle) -> dict:
    return {"root": str(handle.root), "generation": handle.generation, "identity": handle.identity}


def fit_fp8_scale(samples: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """Choose a deterministic per-tensor E4M3 scale by reservoir MSE."""
    finite = samples.detach().float().reshape(-1)
    finite = finite[torch.isfinite(finite)]
    if finite.numel() == 0:
        raise RuntimeError("cannot fit FP8 scale without finite observations")
    absolute = finite.abs()
    bases = [absolute.max() / 448.0]
    for quantile in (0.99, 0.999, 0.9999):
        bases.append(torch.quantile(absolute, quantile) / 448.0)
    candidates = sorted({max(torch.finfo(torch.float32).tiny, float(base) * multiplier)
                         for base in bases for multiplier in (0.8, 0.9, 1.0, 1.1, 1.25)})
    total_power = finite.square().sum().item()
    evaluated = []
    for candidate in candidates:
        normalized = (finite / candidate).clamp(-448.0, 448.0)
        restored = normalized.to(torch.float8_e4m3fn).float() * candidate
        error = restored - finite
        mse = error.square().mean().item()
        signal = finite.square().mean().item()
        evaluated.append({
            "scale": candidate,
            "mse": mse,
            "mean_absolute_error": error.abs().mean().item(),
            "mean_relative_error": (error.abs() / finite.abs().clamp_min(1e-12)).mean().item(),
            "clipping_rate": (absolute > 448.0 * candidate).float().mean().item(),
            "sqnr_db": 10.0 * __import__("math").log10(signal / mse) if mse > 0 and signal > 0 else None,
        })
    best = min(evaluated, key=lambda item: (item["mse"], item["clipping_rate"], item["scale"]))
    return torch.tensor([best["scale"]], dtype=torch.float32), {
        "objective": "w4_kv_reservoir_mse",
        "elements": finite.numel(),
        "selected": best,
        "candidates": evaluated,
        "total_power": total_power,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--scheme", default="W4A16")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--seqlen", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dataset", default="NeelNanda/pile-10k")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kv-cache", choices=["none", "fp8"], default="none")
    parser.add_argument("--phase", choices=["bf16-reference", "quantize"], default="quantize")
    parser.add_argument("--reference-info-output")
    parser.add_argument("--reference-root")
    parser.add_argument("--bf16-reference")
    parser.add_argument("--enable-torch-compile", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--resolved-ignore-output")
    parser.add_argument("--checkpoint-root")
    parser.add_argument("--checkpoint-info-output")
    parser.add_argument("--workspace-lock-sha256")
    parser.add_argument("--activation-store-config", default="{}")
    parser.add_argument("--resource-config", default="{}")
    parser.add_argument("--diagnostics-dir")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--trace-timeout-seconds",
        type=int,
        default=600,
        help="fail if llm-compressor's CPU-only torch.fx trace exceeds this duration",
    )
    parser.add_argument(
        "--ignore-json",
        default="[]",
        help="JSON array of module-name fragments to preserve at source precision",
    )
    args = parser.parse_args()

    storage_config = json.loads(args.activation_store_config)
    resource_config = json.loads(args.resource_config)
    diagnostics_dir = Path(args.diagnostics_dir or Path(args.output_dir).parent / "eval")
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    telemetry = JsonlTelemetry(diagnostics_dir / "telemetry.jsonl", common={
        "model": args.model, "revision": args.revision, "dataset": args.dataset,
        "seed": args.seed,
    })

    if not torch.xpu.is_available():
        parser.error(
            "Intel XPU is unavailable; refusing to fall back to CPU for quantization"
        )
    if args.trace_timeout_seconds <= 0:
        parser.error("--trace-timeout-seconds must be positive")
    probe = torch.randn((1, 4, 16, 64), device="xpu", dtype=torch.bfloat16)
    torch.nn.functional.scaled_dot_product_attention(probe, probe, probe)
    torch.xpu.synchronize()
    print(f"XPU compute probe passed: {torch.xpu.get_device_name(0)}", flush=True)
    del probe
    install_trace_watchdog(args.trace_timeout_seconds)

    ignore_fragments = json.loads(args.ignore_json)
    if not isinstance(ignore_fragments, list) or not all(
        isinstance(item, str) and item for item in ignore_fragments
    ):
        parser.error("--ignore-json must be a JSON array of non-empty strings")
    for item in ignore_fragments:
        if item.startswith("re:"):
            try:
                re.compile(item[3:])
            except re.error as exc:
                parser.error(f"invalid ignore regex {item!r}: {exc}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        revision=args.revision,
        dtype="auto",
        trust_remote_code=False,
    )
    args.model_dtype = str(next(model.parameters()).dtype)
    resolved_ignores = {}
    for selector in ignore_fragments:
        matched = [name for name, _ in match_named_modules(model, [selector], [])]
        if not matched:
            parser.error(f"ignore selector matched no model modules: {selector!r}")
        resolved_ignores[selector] = matched
    print("Resolved full-precision modules:")
    print(json.dumps(resolved_ignores, indent=2))
    if args.resolved_ignore_output:
        Path(args.resolved_ignore_output).write_text(json.dumps(resolved_ignores, indent=2) + "\n")
    dataset = get_dataset(
        tokenizer=tokenizer,
        seqlen=args.seqlen,
        nsamples=args.samples,
        dataset_name=args.dataset,
        seed=args.seed,
    )
    sample_bytes = 0
    for item in dataset:
        def count_bytes(value):
            if isinstance(value, torch.Tensor):
                return value.numel() * value.element_size()
            if isinstance(value, dict):
                return sum(count_bytes(child) for child in value.values())
            if isinstance(value, (tuple, list)):
                return sum(count_bytes(child) for child in value)
            return 0
        sample_bytes += count_bytes(item)
    model_bytes = sum(value.numel() * value.element_size() for value in model.parameters())
    scratch = Path(storage_config.get("path") or Path(args.output_dir).parent / "activation-store")
    scratch.mkdir(parents=True, exist_ok=True)
    estimates = {
        # Transformers loads this checkpoint through mmap-backed safetensors;
        # sequential compression onloads the active block. Account reclaimable
        # model backing separately from bounded anonymous active-block state.
        "model_optimizer_bytes": 12 * 1024**3,
        "model_file_cache_bytes": model_bytes,
        "activation_frontier_bytes": sample_bytes,
        "live_minibatch_bytes": max(1, sample_bytes // max(1, args.samples)) * args.batch_size * 2,
        "safety_margin_bytes": 4 * 1024**3,
        "activation_disk_bytes": sample_bytes * 5,
        "checkpoint_disk_bytes": model_bytes + sample_bytes * 2,
    }
    admission = solve_resources(storage_config, resource_config, estimates, path=scratch)
    atomic_json(diagnostics_dir / "preflight.json", {
        "status": "pass", "storage": storage_config, "resources": resource_config,
        "estimates": estimates, "resolved": admission,
    })
    telemetry.emit("preflight_pass", admission=admission, meminfo=meminfo())
    activation_store = ActivationStore(scratch, lru_bytes=admission["memory"]["pageable_lru"])
    activation_store.collect_orphans()
    store_identity = {"model": args.model, "revision": args.revision,
                      "dataset_sha256": hash_calibration_data(dataset), "seed": args.seed,
                      "samples": args.samples, "seqlen": args.seqlen}
    DiskIntermediatesCache.configure(activation_store, store_identity, telemetry)
    reference_cache_path = None
    current_reference_identity = None
    bf16_reference = None
    bf16_reservoirs = {}
    if args.phase == "bf16-reference":
        current_reference_identity = reference_identity(args, dataset, tokenizer, resolved_ignores)
        reference_hash = hashlib.sha256(canonical_json(current_reference_identity)).hexdigest()
        if args.reference_root:
            reference_root = Path(args.reference_root)
            reference_root.mkdir(parents=True, exist_ok=True)
            reference_cache_path = reference_root / f"{reference_hash}.json"
            if reference_cache_path.is_file():
                cached = json.loads(reference_cache_path.read_text())
                if cached.get("status") != "complete" or cached.get("identity_sha256") != reference_hash:
                    raise RuntimeError("cached BF16 reference is incomplete or has the wrong identity")
                from activation_store import CorpusHandle
                handle_data = cached["corpus"]
                cached_store = ActivationStore(Path(handle_data["root"]))
                cached_store.open(CorpusHandle(Path(handle_data["root"]), handle_data["generation"], handle_data["identity"]), verify_all=True)
                if args.reference_info_output:
                    atomic_json(Path(args.reference_info_output), cached)
                telemetry.emit("bf16_reference_reused", identity_sha256=reference_hash)
                return
    elif args.kv_cache == "fp8":
        if not args.bf16_reference:
            parser.error("post-W4 FP8 KV calibration requires --bf16-reference")
        reference_path = Path(args.bf16_reference)
        bf16_reference = json.loads(reference_path.read_text())
        expected_identity = reference_identity(args, dataset, tokenizer, resolved_ignores)
        expected_hash = hashlib.sha256(canonical_json(expected_identity)).hexdigest()
        if bf16_reference.get("status") != "complete" or bf16_reference.get("identity_sha256") != expected_hash:
            raise RuntimeError("BF16 reference identity does not match the exact current corpus/toolchain")
        from activation_store import CorpusHandle
        handle_data = bf16_reference["corpus"]
        ref_store = ActivationStore(Path(handle_data["root"]))
        ref_corpus = ref_store.open(
            CorpusHandle(Path(handle_data["root"]), handle_data["generation"], handle_data["identity"]),
            verify_all=True,
        )
        bf16_reservoirs = ref_corpus[0]
        telemetry.emit("bf16_reference_verified", identity_sha256=expected_hash)
    dag = None
    if args.resume and not args.checkpoint_root:
        parser.error("--resume requires --checkpoint-root")
    if args.checkpoint_root:
        packages = package_versions()
        plan = {
            "schema": CheckpointDag.schema,
            "source": {"repo": args.model, "revision": args.revision},
            "algorithm": {
                "scheme": args.scheme,
                "iters": args.iters,
                "batch_size": args.batch_size,
                "sequence_length": args.seqlen,
                "samples": args.samples,
                "dataset": args.dataset,
                "dataset_sha256": hash_calibration_data(dataset),
                "seed": args.seed,
                "kv_cache": args.kv_cache,
                "torch_compile": args.enable_torch_compile,
                "low_gpu_mem_usage": True,
                "ignore": ignore_fragments,
                "resolved_ignore": resolved_ignores,
            },
            "packages": packages,
            "toolchain": {
                "nix_closure": os.environ.get("QUANTIZE_TOOLCHAIN_ID", "unknown"),
                "runner_sha256": sha256_file(Path(__file__)),
                "workspace_lock_sha256": args.workspace_lock_sha256,
            },
        }
        dag = CheckpointDag(
            Path(args.checkpoint_root), plan, resume=args.resume,
            activation_store=activation_store,
        )
        if args.checkpoint_info_output:
            Path(args.checkpoint_info_output).write_text(
                json.dumps({"run_hash": dag.run_hash, "path": str(dag.root)}, indent=2)
                + "\n"
            )
    autoround_modifier = AutoRoundModifier(
            targets=["Linear"],
            scheme=args.scheme,
            ignore=ignore_fragments,
            iters=args.iters,
            enable_torch_compile=args.enable_torch_compile,
            batch_size=args.batch_size,
        )
    autoround_modifier.attach_activation_store(
        activation_store,
        store_identity,
        telemetry,
    )
    if dag is not None:
        autoround_modifier.attach_checkpoint_dag(dag)
    kv_modifier = None
    if args.kv_cache == "fp8":
        kv_modifier = CalibratedKVModifier(
                # A KV-only modifier must not synthesize the default empty
                # Linear config group and overwrite AutoRound's W4 schemes.
                targets=[],
                kv_cache_scheme=calibrated_kv_cache_scheme(),
                capture_samples=args.samples,
                capture_batch_size=args.batch_size,
            )
    if args.phase == "bf16-reference":
        if kv_modifier is None:
            parser.error("--phase bf16-reference requires --kv-cache fp8")

    def run_oneshot(recipe):
        with patch.object(sequential_pipeline, "IntermediatesCache", DiskIntermediatesCache):
            oneshot(
                model=model,
                tokenizer=tokenizer,
                # Calibration is text-only. Passing the tokenizer as the processor
                # prevents llm-compressor from auto-loading Qwen's vision/video
                # processor (and an unnecessary Torchvision dependency). Its argument
                # resolver promotes tokenizer to processor internally; providing both
                # aliases is rejected.
                dataset=dataset,
                recipe=recipe,
                max_seq_length=args.seqlen,
                num_calibration_samples=args.samples,
                shuffle_calibration_samples=False,
                batch_size=args.batch_size,
                data_collator="truncation",
            )

    if args.phase == "bf16-reference":
        run_oneshot([kv_modifier])
    else:
        # Use two explicit lifecycle runs.  llm-compressor may clone modifiers
        # when composing heterogeneous pipelines, which makes diagnostics stored
        # on the caller's KV modifier inaccessible after a combined run.  The
        # first pass freezes the W4 model in memory; the second, observer-only
        # pass is then identical to the proven BF16 capture path and measures
        # the actual post-W4 cache tensors.
        run_oneshot([autoround_modifier])
        weight_snapshot = snapshot_weight_quantization(model)
        if kv_modifier is not None:
            run_oneshot([kv_modifier])
        weight_compression = restore_and_validate_weight_quantization(
            model, weight_snapshot
        )
        atomic_json(diagnostics_dir / "weight-compression.json", weight_compression)
    if args.phase == "bf16-reference":
        audit = getattr(kv_modifier, "calibration_audit", {})
        reservoirs = getattr(kv_modifier, "calibration_reservoirs", {})
        if not audit or not reservoirs:
            raise RuntimeError("BF16 KV reference capture produced no observations")
        identity = current_reference_identity or reference_identity(args, dataset, tokenizer, resolved_ignores)
        identity_hash = hashlib.sha256(canonical_json(identity)).hexdigest()
        writer = activation_store.writer(f"bf16-kv-reference-{identity_hash[:16]}", identity=identity)
        writer.append(reservoirs)
        handle = writer.commit()
        reference = {
            "schema": 1,
            "status": "complete",
            "identity": identity,
            "identity_sha256": identity_hash,
            "corpus": json_handle(handle),
            "layers": audit,
            "completed_at": time.time(),
        }
        atomic_json(diagnostics_dir / "bf16-reference.json", reference)
        if reference_cache_path is not None:
            atomic_json(reference_cache_path, reference)
        if args.reference_info_output:
            atomic_json(Path(args.reference_info_output), reference)
        telemetry.emit("bf16_reference_complete", identity_sha256=identity_hash, corpus=json_handle(handle))
        return
    if not args.no_save:
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        kv_scales = {}
        if kv_modifier:
            audit = getattr(kv_modifier, "calibration_audit", {})
            reservoirs = getattr(kv_modifier, "calibration_reservoirs", {})
            if not audit or not reservoirs:
                raise RuntimeError("post-W4 FP8 KV calibration produced no observations")
            fitted = {}
            for key, samples in reservoirs.items():
                scale, fit = fit_fp8_scale(samples)
                scale_key = key.rsplit(".", 1)[0] + f".{key.rsplit('.', 1)[1]}_scale"
                kv_scales[scale_key] = scale
                fitted[key] = fit
                reference_samples = bf16_reservoirs.get(key)
                if reference_samples is not None and reference_samples.numel() == samples.numel():
                    delta = samples.float() - reference_samples.float()
                    fitted[key]["bf16_reference"] = {
                        "mean_absolute_error": delta.abs().mean().item(),
                        "root_mean_square_error": delta.square().mean().sqrt().item(),
                        "max_absolute_error": delta.abs().max().item(),
                    }
            restore_kv_scales(model, kv_scales)
            merged = {}
            for key, value in kv_scales.items():
                layer, kind = key.rsplit(".", 1)
                audit_key = f"{layer}.{kind.removesuffix('_scale')}"
                merged.setdefault(layer, {})[kind] = {
                        "chosen_scale": value.detach().float().cpu().reshape(-1).tolist(),
                        **audit.get(audit_key, {}),
                        "fit": fitted.get(audit_key),
                        "error_by_token_position": None,
                    }
            atomic_json(diagnostics_dir / "kv-calibration.json", {
                "status": "complete", "phase": "post_w4",
                "reference": args.bf16_reference,
                "layers": merged,
            })
        model.save_pretrained(output, save_compressed=True)
        saved_weight_compression = validate_saved_weight_compression(
            output, weight_compression["expected_weight_modules"]
        )
        weight_compression.update(saved_weight_compression)
        atomic_json(diagnostics_dir / "weight-compression.json", weight_compression)
        checkpoint_scale_keys = (
            add_kv_scales_to_checkpoint(output, kv_scales) if kv_scales else []
        )
        tokenizer.save_pretrained(output)
        source_snapshot = Path(
            snapshot_download(repo_id=args.model, revision=args.revision)
        )
        source_config = read_json(source_snapshot / "config.json")
        if source_config.get("model_type") == "qwen3_5" and source_config.get(
            "vision_config"
        ):
            graft_in_place(source_snapshot, output)
        saved_config = json.loads((output / "config.json").read_text())
        quant_config = saved_config.get("quantization_config")
        if not isinstance(quant_config, dict):
            raise RuntimeError("saved checkpoint is missing quantization_config")
        if args.kv_cache == "fp8" and not quant_config.get("kv_cache_scheme"):
            raise RuntimeError("saved checkpoint is missing calibrated FP8 kv_cache_scheme")
        if args.kv_cache == "fp8" and len(checkpoint_scale_keys) != len(kv_scales):
            raise RuntimeError("saved checkpoint is missing calibrated FP8 KV scale tensors")
        integrity = validate_inventory(
            source_snapshot, output, ignored=ignore_fragments,
            kv_expected=checkpoint_scale_keys,
        )
        atomic_json(diagnostics_dir / "artifact-integrity.json", integrity)
        telemetry.emit("artifact_integrity_pass", report=integrity)
        output_names = tensor_inventory(output)
        components = {
            "full_attention_kv": "quantized" if args.kv_cache == "fp8" else "ignored",
            "gated_deltanet_recurrent_state": "unsupported",
            "mtp": "preserved" if any("mtp" in key for key in output_names) else "not_present",
            "vision": "preserved" if any("visual" in key or "vision" in key for key in output_names) else "not_present",
        }
        atomic_json(diagnostics_dir / "components.json", {
            "observed": ["decoder_activation_inputs", "propagated_quantized_inputs"],
            "components": components, "ignored_selectors": ignore_fragments,
        })
        atomic_json(diagnostics_dir / "export-gates.json", {
            "exportable": False,
            "gates": {
                "artifact_inventory": {"status": "pass", "report": "artifact-integrity.json"},
                "fixed_seed_kl_top_token": {"status": "pending"},
                "short_generation_repetition": {"status": "pending"},
                "context_length_sweep": {"status": "pending"},
                "vllm_loader_reload": {"status": "pending"},
            },
            "waivers": [],
        })
    if dag is not None:
        dag.close()


if __name__ == "__main__":
    main()
