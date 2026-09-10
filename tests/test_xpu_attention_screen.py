"""CPU checks for interpretation of paired complete-call attention samples."""

import unittest

from scripts.xpu_attention_screen import summarize


class BlockSummaryTests(unittest.TestCase):
    def test_symmetric_order_cancels_linear_drift(self):
        blocks = []
        for block in range(30):
            arms = ["control", "candidate"] if block % 2 else ["candidate", "control"]
            sequence = arms + arms[::-1]
            blocks.append(
                [
                    {"arm": arm, "ms": 10 + block + position}
                    for position, arm in enumerate(sequence)
                ]
            )
        result = summarize(blocks)
        self.assertEqual(result["speedup"], 1)
        self.assertEqual(result["speedup_ci95"], [1, 1])
        self.assertGreater(result["control_last_first_ratio"], 1)

    def test_regression_direction_and_interval(self):
        block = [
            {"arm": arm, "ms": 15 if arm == "candidate" else 10}
            for arm in ("control", "candidate", "candidate", "control")
        ]
        result = summarize([block] * 30)
        self.assertEqual(result["control_ms"], 10)
        self.assertEqual(result["candidate_ms"], 15)
        self.assertAlmostEqual(result["speedup"], 2 / 3)
        self.assertEqual(result["control_cv"], 0)
        self.assertEqual(result["control_last_first_ratio"], 1)

    def test_speedup_uses_total_cost_instead_of_averaging_ratios(self):
        def block(candidate_ms):
            return [
                {"arm": arm, "ms": candidate_ms if arm == "candidate" else 1}
                for arm in ("control", "candidate", "candidate", "control")
            ]

        result = summarize([block(1), block(9)] * 15)
        self.assertEqual(result["speedup"], 0.2)


if __name__ == "__main__":
    unittest.main()
