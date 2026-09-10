"""Tests for full-prefill weighting and the boundaries of its prediction."""

import unittest

from scripts.xpu_attention_weight import anchor_quality, predict


class PredictionTests(unittest.TestCase):
    def setUp(self):
        self.anchors = [
            {"m": 2048, "k": 2048, "control_ms": 2},
            {"m": 2048, "k": 8192, "control_ms": 8},
            {"m": 2047, "k": 16383, "control_ms": 16},
        ]

    def test_all_layers_and_chunks_contribute(self):
        calls = [
            {"m": 2048, "k": 2048},
            {"m": 2048, "k": 5120},
            {"m": 2047, "k": 16383},
        ] * 16
        result = predict(calls, self.anchors, "control")
        self.assertEqual(
            result,
            {
                "ms": 368,
                "calls": 48,
                "interpolated_calls": 16,
                "contributing_anchors": [
                    {"m": 2048, "k": 2048},
                    {"m": 2048, "k": 8192},
                    {"m": 2047, "k": 16383},
                ],
            },
        )

    def test_unused_long_context_anchors_do_not_invalidate_16k(self):
        anchors = [
            {**a, "warmup_stable": True, "control_cv": 0.01} for a in self.anchors
        ] + [
            {
                "m": 2048,
                "k": 63488,
                "control_ms": 64,
                "warmup_stable": False,
                "control_cv": 0.5,
            }
        ]
        calls = [{"m": 2048, "k": k} for k in range(2048, 16383, 2048)] + [
            {"m": 2047, "k": 16383}
        ]
        result = predict(calls * 16, anchors, "control")
        self.assertEqual(
            anchor_quality(result, anchors),
            {"all_anchors_stable": True, "largest_control_cv": 0.01},
        )
        self.assertNotIn({"m": 2048, "k": 63488}, result["contributing_anchors"])

    def test_interpolation_uses_quality_of_both_endpoints(self):
        anchors = [
            {**self.anchors[0], "warmup_stable": True, "control_cv": 0.01},
            {**self.anchors[1], "warmup_stable": False, "control_cv": 0.2},
        ]
        result = predict([{"m": 2048, "k": 5120}], anchors, "control")
        self.assertEqual(
            anchor_quality(result, anchors),
            {"all_anchors_stable": False, "largest_control_cv": 0.2},
        )

    def test_unmeasured_tail_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unmeasured ragged"):
            predict([{"m": 1535, "k": 65023}], self.anchors, "control")

    def test_extrapolation_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "extrapolation forbidden"):
            predict([{"m": 2048, "k": 32768}], self.anchors, "control")

    def test_full_chunk_normalization(self):
        result = predict([{"m": 2048, "k": 16383}], self.anchors, "control")
        self.assertAlmostEqual(result["ms"], 16 * 2048 / 2047)
        self.assertEqual(result["contributing_anchors"], [{"m": 2047, "k": 16383}])


if __name__ == "__main__":
    unittest.main()
