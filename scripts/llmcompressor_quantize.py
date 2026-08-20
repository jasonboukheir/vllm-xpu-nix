#!/usr/bin/env python3
"""Run a combined AutoRound weight + calibrated FP8 KV-cache recipe."""

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
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
from llmcompressor.pipelines.sequential import pipeline as sequential_pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from graft_checkpoint_extras import graft_in_place, read_json
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
    """Add tiny KV tensors as a shard and index them without rewriting model bytes.

    Some Transformers composite models rename checkpoint keys while saving and
    omit newly registered attention parameters.  An explicit safetensors index
    lets us retain the large shard untouched while making the scales loadable.
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
    save_file(checkpoint_scales, output / shard_name, metadata={"format": "pt"})
    weight_map.update({key: shard_name for key in checkpoint_scales})
    metadata["total_size"] = int(metadata.get("total_size", 0)) + sum(
        value.numel() * value.element_size() for value in checkpoint_scales.values()
    )
    index_path.write_text(
        json.dumps({"metadata": metadata, "weight_map": weight_map}, indent=2, sort_keys=True)
        + "\n"
    )
    return sorted(checkpoint_scales)


class CheckpointDag:
    """Content-addressed block states with an atomic, hash-chained head."""

    schema = 1

    def __init__(self, root: Path, plan: dict, *, resume: bool):
        self.plan = plan
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
        self.head.update(
            head=node_hash,
            q_input_slot=q_slot,
            q_input_sha256=sha256_file(self.root / q_slot),
            rng_slot=rng_slot,
            rng_sha256=sha256_file(self.root / rng_slot),
        )
        self._atomic_json(self.root / "head.json", self.head)
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

    def input_capture_hook(self, module, args, kwargs):
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
        if dag is None:
            return super().on_sequential_epoch_end(state, event, modules, **kwargs)
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


class CalibratedKVModifier(QuantizationModifier):
    """Capture final KV scales before llm-compressor lifecycle cleanup."""

    def on_calibration_end(self, state, event, **kwargs):
        object.__setattr__(self, "calibrated_scales", collect_kv_scales(state.model))
        return super().on_calibration_end(state, event, **kwargs)


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
    parser.add_argument("--enable-torch-compile", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--resolved-ignore-output")
    parser.add_argument("--checkpoint-root")
    parser.add_argument("--checkpoint-info-output")
    parser.add_argument("--workspace-lock-sha256")
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
    dag = None
    if args.resume and not args.checkpoint_root:
        parser.error("--resume requires --checkpoint-root")
    if args.checkpoint_root:
        packages = {}
        for package in (
            "auto-round",
            "llmcompressor",
            "compressed-tensors",
            "transformers",
            "torch",
        ):
            try:
                packages[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                packages[package] = "unknown"
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
        dag = CheckpointDag(Path(args.checkpoint_root), plan, resume=args.resume)
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
    if dag is not None:
        autoround_modifier.attach_checkpoint_dag(dag)
    recipe = []
    kv_modifier = None
    if args.kv_cache == "fp8":
        # This is the ordering used by llm-compressor's combined AutoRound/KV
        # example. Capture calibrated KV scales first, then let AutoRound remain
        # the final modifier so compressed weight metadata drives serialization.
        kv_modifier = CalibratedKVModifier(
                kv_cache_scheme={
                    "num_bits": 8,
                    "type": "float",
                    "strategy": "tensor",
                    "dynamic": False,
                    "symmetric": True,
                }
            )
        recipe.append(kv_modifier)
    recipe.append(autoround_modifier)
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
    if not args.no_save:
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        kv_scales = getattr(kv_modifier, "calibrated_scales", {}) if kv_modifier else {}
        if kv_modifier:
            if not kv_scales:
                raise RuntimeError("FP8 KV calibration produced no captured scales")
            restore_kv_scales(model, kv_scales)
        model.save_pretrained(output, save_compressed=True)
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
    if dag is not None:
        dag.close()


if __name__ == "__main__":
    main()
