import importlib.util
from pathlib import Path
import unittest

import torch
from compressed_tensors.quantization import QuantizationArgs
from llmcompressor.observers import Observer


SPEC = importlib.util.spec_from_file_location(
    "llmcompressor_quantize",
    Path(__file__).parents[1] / "scripts" / "llmcompressor_quantize.py",
)
quantize = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(quantize)


class KVCalibrationTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
