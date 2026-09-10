import unittest

from scripts.xpu_attention_pure_prefill import eligible


class EligibilityTest(unittest.TestCase):
    def test_captured_prefill(self):
        self.assertTrue(
            eligible(
                paged=True,
                cu_q=[0, 2047],
                kv_lengths=[16383],
                q_tokens=2047,
                max_q=2047,
                table_rows=1,
            )
        )

    def test_excluded_routing(self):
        base = {
            "paged": True,
            "cu_q": [0, 2048],
            "kv_lengths": [8192],
            "q_tokens": 2048,
            "max_q": 2048,
            "table_rows": 1,
        }
        for change in (
            {"paged": False},
            {
                "cu_q": [0, 2048, 2049],
                "kv_lengths": [8192, 17],
                "q_tokens": 2049,
                "table_rows": 2,
            },
            {
                "cu_q": [0, 2048, 4096],
                "kv_lengths": [8192, 8192],
                "q_tokens": 4096,
                "table_rows": 2,
            },
            {"q_tokens": 4096},
            {"cu_q": [0, 1024]},
            {"kv_lengths": [1024]},
            {"cu_q": [0, 1], "q_tokens": 1, "max_q": 1},
            {"cu_q": [0, 16], "q_tokens": 16, "max_q": 16},
        ):
            with self.subTest(change=change):
                self.assertFalse(eligible(**(base | change)))


if __name__ == "__main__":
    unittest.main()
