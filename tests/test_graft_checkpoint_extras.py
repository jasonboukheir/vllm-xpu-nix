import importlib.util
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "graft_checkpoint_extras", ROOT / "scripts" / "graft_checkpoint_extras.py"
)
graft = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(graft)


class WrapperConfigTests(unittest.TestCase):
    def test_requires_vision_processor_asset(self):
        self.assertEqual(graft.PROCESSOR_ASSETS, ("preprocessor_config.json",))

    def test_restores_wrapper_and_adds_full_precision_extras(self):
        source = {
            "architectures": ["Qwen3_5ForConditionalGeneration"],
            "model_type": "qwen3_5",
            "text_config": {"model_type": "qwen3_5_text"},
            "vision_config": {"model_type": "qwen3_5_vision"},
        }
        quantized = {
            "transformers_version": "5.14.1",
            "quantization_config": {"ignore": ["lm_head"]},
        }
        result = graft.wrapper_config(source, quantized)
        self.assertEqual(result["model_type"], "qwen3_5")
        self.assertEqual(
            result["architectures"], ["Qwen3_5ForConditionalGeneration"]
        )
        self.assertEqual(result["transformers_version"], "5.14.1")
        self.assertIn(r"re:^mtp.*", result["quantization_config"]["ignore"])
        self.assertIn(
            r"re:^visual.*", result["quantization_config"]["ignore"]
        )

    def test_rejects_text_only_source(self):
        with self.assertRaisesRegex(ValueError, "conditional-generation wrapper"):
            graft.wrapper_config(
                {"model_type": "qwen3_5_text"},
                {"quantization_config": {}},
            )


if __name__ == "__main__":
    unittest.main()
