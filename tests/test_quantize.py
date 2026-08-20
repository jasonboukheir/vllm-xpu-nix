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
            self.assertEqual(config["quantization"]["ignore"], ["lm_head"])
            calibration = config["quantization"]["calibration"]
            self.assertEqual(calibration["storage"]["mode"], "disk")
            self.assertEqual(calibration["storage"]["pinned_staging_mib"], 512)
            self.assertEqual(calibration["resources"]["memory_high_gib"], 68)

    def test_ignore_rules_accept_exact_and_regex_selectors(self):
        config = {"quantization": {"ignore": ["lm_head", r"re:.*mtp\.fc$"]}}
        self.assertEqual(quantize.ignore_rules(config), ["lm_head", r"re:.*mtp\.fc$"])

    def test_ignore_rules_reject_commas(self):
        with self.assertRaises(SystemExit):
            quantize.ignore_rules({"quantization": {"ignore": ["lm_head,mtp.fc"]}})

    def test_test_subcommand_defaults_to_two_timing_points(self):
        args = quantize.parser().parse_args(["test"])
        self.assertIs(args.func, quantize.cmd_test)
        self.assertEqual(args.test_iters, [5, 20])
        self.assertEqual(args.test_calibration_samples, 32)
        self.assertFalse(args.full_calibration)

    def test_doctor_subcommand(self):
        args = quantize.parser().parse_args(["doctor"])
        self.assertIs(args.func, quantize.cmd_doctor)


if __name__ == "__main__":
    unittest.main()
