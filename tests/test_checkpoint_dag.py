import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme
from safetensors import safe_open
from safetensors.torch import save_file


SCRIPT = Path(__file__).parents[1] / "scripts" / "llmcompressor_quantize.py"
SPEC = importlib.util.spec_from_file_location("llmcompressor_quantize", SCRIPT)
module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(module)


class CheckpointDagTests(unittest.TestCase):
    def test_checkpoint_manager_attaches_to_modifier(self):
        modifier = module.AutoRoundModifier(
            targets=["Linear"], scheme="W4A16", iters=5, batch_size=1
        )
        marker = object()
        modifier.attach_checkpoint_dag(marker)
        self.assertIs(modifier._checkpoint_dag, marker)
        self.assertEqual(modifier._checkpoint_index, 0)
        self.assertEqual(type(modifier).__name__, "AutoRoundModifier")

    def test_kv_only_attention_scheme_is_not_sent_to_autoround(self):
        modifier = module.AutoRoundModifier(
            targets=["Linear"], scheme="W4A16", iters=5, batch_size=1
        )
        wrapped = torch.nn.Module()
        wrapped.attention = torch.nn.Module()
        wrapped.attention.quantization_scheme = QuantizationScheme(
            targets=["attention"],
            input_activations=QuantizationArgs(
                num_bits=8,
                type="float",
                strategy="tensor",
                dynamic=False,
                symmetric=True,
            ),
        )
        self.assertEqual(modifier._build_layer_config_for_autoround(wrapped), {})

    def test_autoround_cleanup_preserves_kv_scheme_and_scales(self):
        modifier = module.AutoRoundModifier(
            targets=["Linear"], scheme="W4A16", iters=5, batch_size=1
        )
        wrapped = torch.nn.Module()
        wrapped.attention = torch.nn.Module()
        scheme = QuantizationScheme(
            targets=["attention"],
            input_activations=QuantizationArgs(
                num_bits=8,
                type="float",
                strategy="tensor",
                dynamic=False,
                symmetric=True,
            ),
        )
        wrapped.attention.quantization_scheme = scheme
        wrapped.attention.register_parameter(
            "k_scale", torch.nn.Parameter(torch.tensor(0.25), requires_grad=False)
        )
        wrapped.attention.register_parameter(
            "v_scale", torch.nn.Parameter(torch.tensor(0.5), requires_grad=False)
        )

        def destructive_cleanup(_self, model, _qparams):
            delattr(model.attention, "quantization_scheme")
            delattr(model.attention, "k_scale")
            delattr(model.attention, "v_scale")

        with patch.object(
            module.UpstreamAutoRoundModifier,
            "_postprocess_qparams",
            destructive_cleanup,
        ):
            modifier._postprocess_qparams(wrapped, {})

        self.assertIs(wrapped.attention.quantization_scheme, scheme)
        self.assertEqual(wrapped.attention.k_scale.item(), 0.25)
        self.assertEqual(wrapped.attention.v_scale.item(), 0.5)

    def test_kv_calibration_reapplies_only_kv_scheme_after_autoround(self):
        modifier = module.CalibratedKVModifier(
            kv_cache_scheme={
                "num_bits": 8,
                "type": "float",
                "strategy": "tensor",
                "dynamic": False,
                "symmetric": True,
            }
        )
        model = torch.nn.Module()
        state = SimpleNamespace(model=model)
        with (
            patch.object(module, "_apply_kv_cache_scheme") as apply_kv,
            patch.object(module.QuantizationModifier, "on_calibration_start") as start,
        ):
            modifier.on_calibration_start(state, object())

        apply_kv.assert_called_once_with(
            model,
            modifier.resolved_config.kv_cache_scheme,
            modifier.resolved_config.quantization_status,
        )
        start.assert_called_once()

    def test_kv_scales_are_added_as_an_indexed_checkpoint_shard(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            save_file(
                {
                    "model.language_model.layers.3.self_attn.q_proj.weight": torch.ones(2),
                    "model.language_model.layers.3.self_attn.k_proj.weight": torch.ones(2),
                },
                output / "model.safetensors",
            )
            keys = module.add_kv_scales_to_checkpoint(
                output,
                {
                    "model.layers.3.self_attn.k_scale": torch.tensor(0.25),
                    "model.layers.3.self_attn.v_scale": torch.tensor(0.5),
                },
            )
            self.assertEqual(
                keys,
                [
                    "model.language_model.layers.3.self_attn.k_scale",
                    "model.language_model.layers.3.self_attn.v_scale",
                ],
            )
            index = json.loads((output / "model.safetensors.index.json").read_text())
            self.assertEqual(
                index["weight_map"][keys[0]], "model-kv-scales.safetensors"
            )
            with safe_open(output / "model-kv-scales.safetensors", framework="pt") as stream:
                self.assertEqual(set(stream.keys()), set(keys))

    def test_commit_validate_and_restore(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = {"schema": 1, "source": {"revision": "a" * 40}}
            original = torch.nn.Linear(3, 2)
            expected = {key: value.detach().clone() for key, value in original.state_dict().items()}
            q_input = torch.arange(6, dtype=torch.float32).reshape(2, 3)

            dag = module.CheckpointDag(Path(directory), plan, resume=False)
            dag.commit(0, "model.layers.0", original, q_input)
            run_hash = dag.run_hash
            dag.close()

            restored = torch.nn.Linear(3, 2)
            with torch.no_grad():
                restored.weight.zero_()
                restored.bias.zero_()
            resumed = module.CheckpointDag(Path(directory), plan, resume=True)
            self.assertTrue(resumed.completed_layer(0, "model.layers.0"))
            restored_q_input = resumed.restore(0, "model.layers.0", restored)
            self.assertEqual(resumed.run_hash, run_hash)
            self.assertTrue(torch.equal(restored_q_input, q_input))
            for key, value in restored.state_dict().items():
                self.assertTrue(torch.equal(value, expected[key]))
            resumed.close()

    def test_changed_plan_has_a_different_dag(self):
        with tempfile.TemporaryDirectory() as directory:
            first = module.CheckpointDag(Path(directory), {"iters": 5}, resume=False)
            first.close()
            second = module.CheckpointDag(Path(directory), {"iters": 20}, resume=False)
            self.assertNotEqual(first.run_hash, second.run_hash)
            second.close()

    def test_existing_dag_requires_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            plan = {"iters": 1000}
            first = module.CheckpointDag(Path(directory), plan, resume=False)
            first.close()
            with self.assertRaisesRegex(RuntimeError, "pass --resume"):
                module.CheckpointDag(Path(directory), plan, resume=False)


if __name__ == "__main__":
    unittest.main()
