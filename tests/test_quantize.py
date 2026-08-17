import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location("quantize", Path(__file__).parents[1] / "scripts" / "quantize.py")
quantize = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(quantize)


class QuantizeTests(unittest.TestCase):
    def test_normalize_repo(self):
        self.assertEqual(quantize.normalize_model("Qwen/Qwen3-8B"), ("Qwen/Qwen3-8B", None))

    def test_normalize_url_and_revision(self):
        self.assertEqual(
            quantize.normalize_model("https://huggingface.co/Qwen/Qwen3-8B/tree/main"),
            ("Qwen/Qwen3-8B", "main"),
        )

    def test_rejects_file_url(self):
        with self.assertRaises(SystemExit):
            quantize.normalize_model("https://huggingface.co/Qwen/Qwen3-8B/blob/main/config.json")

    def test_file_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "model.safetensors").write_bytes(b"weights")
            result = quantize.file_manifest(root)
            self.assertEqual(result[0]["path"], "model.safetensors")
            self.assertEqual(result[0]["size"], 7)

    def test_init_defaults_to_hf_account(self):
        with tempfile.TemporaryDirectory() as directory:
            args = quantize.parser().parse_args(["init", "Qwen/Qwen3-8B", "--workspace", directory])
            with patch.dict("os.environ", {}, clear=True):
                quantize.cmd_init(args)
            config = quantize.workspace_config(Path(directory) / "Qwen" / "Qwen3-8B")
            self.assertEqual(config["publish"]["repo"], "jasonboukheir/Qwen3-8B-W4A16-AutoRound")
            self.assertFalse(config["publish"]["private"])


if __name__ == "__main__":
    unittest.main()
