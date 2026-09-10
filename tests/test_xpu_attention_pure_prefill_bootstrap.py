"""Check that weighted bootstrapping retains paired observations."""

import unittest

from scripts.xpu_attention_pure_prefill_bootstrap import estimate


class WeightedBootstrapTests(unittest.TestCase):
    def setUp(self):
        self.anchors = []
        for length, factor in ((2048, 1), (8192, 2)):
            self.anchors.append(
                {
                    "m": 2048,
                    "k": length,
                    "control_ms": 37 * factor,
                    "candidate_ms": 37 * factor * 0.8,
                    "warmup_stable": True,
                    "control_cv": 0.1,
                    "blocks": [
                        [
                            {"arm": "control", "ms": value * factor},
                            {"arm": "candidate", "ms": value * factor * 0.8},
                        ]
                        for value in (1, 10, 100)
                    ],
                }
            )

    def test_exact_paired_ratio_survives_anchor_interpolation(self):
        result = estimate(
            [{"m": 2048, "k": 5120}] * 16,
            self.anchors,
            draws=100,
            seed=11,
        )
        self.assertAlmostEqual(result["latency_change_percent"], -20)
        # Independent resampling of the two arms would give a broad interval
        # because control observations span 100x. Pairing keeps it exact.
        for bound in result["latency_change_percent_ci95"]:
            self.assertAlmostEqual(bound, -20)
        self.assertAlmostEqual(result["control"]["ms"], 888)

    def test_inconsistent_summary_cannot_enter_bootstrap(self):
        self.anchors[0]["control_ms"] = 1
        with self.assertRaisesRegex(ValueError, "summary disagrees"):
            estimate([{"m": 2048, "k": 2048}], self.anchors, draws=100, seed=11)


if __name__ == "__main__":
    unittest.main()
