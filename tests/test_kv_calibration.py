import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import torch
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    QuantizationStatus,
)
from llmcompressor.observers import Observer
from safetensors.torch import save_file


SPEC = importlib.util.spec_from_file_location(
    "llmcompressor_quantize",
    Path(__file__).parents[1] / "scripts" / "llmcompressor_quantize.py",
)
quantize = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(quantize)


class KVCalibrationTests(unittest.TestCase):
    def test_post_kv_pass_preserves_w4_compression_targets(self):
        model = torch.nn.Sequential(torch.nn.Linear(4, 4, bias=False))
        layer = model[0]
        w4 = QuantizationScheme(
            targets=["Linear"],
            weights=QuantizationArgs(
                num_bits=4,
                type="int",
                strategy="group",
                group_size=2,
                symmetric=True,
            ),
        )
        layer.quantization_scheme = w4
        layer.quantization_status = QuantizationStatus.FROZEN
        layer.register_buffer("weight_scale", torch.ones(4, 2))
        snapshot = quantize.snapshot_weight_quantization(model)

        # Reproduce the old second-pass failure: an empty Linear scheme replaced
        # the W4 scheme and the recursive status reset removed its frozen state.
        layer.quantization_scheme = QuantizationScheme(targets=["Linear"])
        del layer.quantization_status

        report = quantize.restore_and_validate_weight_quantization(model, snapshot)

        self.assertIs(layer.quantization_scheme, w4)
        self.assertEqual(layer.quantization_status, QuantizationStatus.FROZEN)
        self.assertTrue(hasattr(layer, "weight_scale"))
        self.assertEqual(report["expected_weight_modules"], 1)
        self.assertEqual(report["actual_weight_modules"], 1)

    def test_saved_weight_gate_requires_packed_tensor_per_w4_module(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            save_file(
                {
                    "layer.weight_packed": torch.zeros(2, dtype=torch.int32),
                    "layer.weight_scale": torch.ones(1),
                    "layer.weight_shape": torch.tensor([4, 4]),
                },
                output / "model.safetensors",
            )
            (output / "config.json").write_text(json.dumps({
                "quantization_config": {
                    "config_groups": {
                        "group_0": {
                            "format": "pack-quantized",
                            "weights": {"num_bits": 4},
                        }
                    }
                }
            }))

            report = quantize.validate_saved_weight_compression(output, 1)
            self.assertEqual(report["packed_tensor_counts"]["weight_packed"], 1)
            with self.assertRaisesRegex(RuntimeError, "expected 2, observed 1"):
                quantize.validate_saved_weight_compression(output, 2)

    def test_scale_can_be_determined_by_first_of_512_samples(self):
        args = QuantizationArgs(**quantize.calibrated_kv_cache_scheme())
        observer = Observer.load_from_registry(
            args.observer,
            base_name="input",
            args=args,
        )

        # The first sample contains the corpus-wide FP8 endpoint. Every later
        # sample is deliberately smaller, so a last-batch-only observer yields
        # the wrong scale after processing the complete 512-sample corpus.
        observer(torch.tensor([[[-448.0, 448.0]]], dtype=torch.float32))
        for _ in range(511):
            observer(torch.tensor([[[-1.0, 1.0]]], dtype=torch.float32))

        scale = observer.get_qparams()["scale"]
        self.assertEqual(scale.numel(), 1)
        self.assertAlmostEqual(scale.item(), 1.0, places=6)

    def test_reference_identity_changes_with_exact_token_ids(self):
        args = SimpleNamespace(
            model="owner/model", revision="resolved-sha", dataset="dataset@revision",
            seed=42, samples=512, seqlen=2048, workspace_lock_sha256="lock",
            model_dtype="torch.bfloat16",
        )
        class FakeTokenizer:
            name_or_path = "owner/model"
            special_tokens_map = {"eos_token": "</s>"}
            chat_template = "{{ messages }}"
            def __len__(self): return 100

        first = [{"input_ids": torch.tensor([[1, 2, 3]]), "attention_mask": torch.ones(1, 3)}]
        changed = [{"input_ids": torch.tensor([[1, 9, 3]]), "attention_mask": torch.ones(1, 3)}]
        identity_a = quantize.reference_identity(args, first, FakeTokenizer(), {})
        identity_b = quantize.reference_identity(args, changed, FakeTokenizer(), {})
        self.assertNotEqual(identity_a["corpus"]["exact_inputs_sha256"], identity_b["corpus"]["exact_inputs_sha256"])

    def test_post_w4_scale_fit_uses_early_extreme(self):
        samples = torch.cat((torch.tensor([448.0]), torch.ones(511)))
        scale, report = quantize.fit_fp8_scale(samples)
        self.assertGreater(scale.item(), 1.0 / 448.0)
        self.assertEqual(report["objective"], "w4_kv_reservoir_mse")
        self.assertGreater(len(report["candidates"]), 1)

    def test_existing_serialized_scale_is_indexed_without_duplicate_shard(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            shard = "model-00001-of-00001.safetensors"
            checkpoint_key = "model.language_model.layers.3.self_attn.k_scale"
            q_proj = "model.language_model.layers.3.self_attn.q_proj.weight"
            save_file(
                {q_proj: torch.ones(1), checkpoint_key: torch.tensor([0.125])},
                output / shard,
            )
            (output / "model.safetensors.index.json").write_text(json.dumps({
                "metadata": {"total_size": 8},
                "weight_map": {q_proj: shard},
            }))

            keys = quantize.add_kv_scales_to_checkpoint(
                output, {"model.layers.3.self_attn.k_scale": torch.tensor([0.125])}
            )

            index = json.loads((output / "model.safetensors.index.json").read_text())
            self.assertEqual(keys, [checkpoint_key])
            self.assertEqual(index["weight_map"][checkpoint_key], shard)
            self.assertFalse((output / "model-kv-scales.safetensors").exists())


if __name__ == "__main__":
    unittest.main()
