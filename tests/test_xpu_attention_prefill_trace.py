"""Reject incomplete, incorrectly attributed or broadened dispatch changes."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.xpu_attention_prefill_trace import sha256, validate_pair


class TracePairTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.control_path = Path(self.directory.name) / "control.json"
        self.candidate_path = Path(self.directory.name) / "candidate.json"
        # Synthetic names keep parser tests independent of large profiler
        # artifacts. Actual frozen fingerprints are checked on retained traces.
        names = {
            "prefill": "cutlass::XeFMHAFwdKernel<expected-policy>",
            "subtract": "subtract-kernel",
            "compare": "compare-kernel",
            "decode": "skipped-decode-kernel",
        }
        fingerprints = {role: sha256(name.encode()) for role, name in names.items()}
        patched = patch(
            "scripts.xpu_attention_prefill_trace.EXPECTED_KERNEL_SHA256",
            fingerprints,
        )
        patched.start()
        self.addCleanup(patched.stop)
        cpu = [
            self.event("cpu_op", "_vllm_fa2_C::varlen_fwd", 1),
            self.event("cpu_op", "aten::sub", 2),
            self.event("cpu_op", "aten::gt", 3),
        ]
        self.prefill = self.event("kernel", names["prefill"], 1)
        self.control = cpu + [
            self.event("kernel", names["subtract"], 2),
            self.event("kernel", names["compare"], 3),
            self.prefill,
            self.event("kernel", names["decode"], 1),
        ]
        self.candidate = [copy.deepcopy(cpu[0]), copy.deepcopy(self.prefill)]

    @staticmethod
    def event(category, name, external_id):
        return {
            "cat": category,
            "name": name,
            "ph": "X",
            "dur": 1.0,
            "args": {"External id": external_id, "device": 0},
        }

    def validate(self):
        self.control_path.write_text(json.dumps({"traceEvents": self.control}))
        self.candidate_path.write_text(json.dumps({"traceEvents": self.candidate}))
        return validate_pair(self.control_path, self.candidate_path)

    def test_exact_removal_passes(self):
        result = self.validate()
        self.assertTrue(result["pass"])
        self.assertEqual(result["device_event_counts"], {"control": 4, "candidate": 1})
        self.assertEqual(
            [r["role"] for r in result["removed"]], ["subtract", "compare", "decode"]
        )

    def test_unknown_prefill_policy_is_rejected(self):
        self.candidate[1]["name"] = "cutlass::XeFMHAFwdKernel<another-policy>"
        with self.assertRaisesRegex(ValueError, "unexpected prefill specialization"):
            self.validate()

    def test_duplicate_prefill_is_rejected(self):
        self.candidate.append(copy.deepcopy(self.prefill))
        with self.assertRaisesRegex(ValueError, "exactly one native prefill"):
            self.validate()

    def test_retained_decode_is_rejected(self):
        self.candidate.append(copy.deepcopy(self.control[-1]))
        with self.assertRaisesRegex(ValueError, "decode event was not removed"):
            self.validate()

    def test_wrong_cpu_origin_of_identical_add_functor_is_rejected(self):
        self.control[1]["name"] = "aten::add"
        with self.assertRaisesRegex(ValueError, "subtract has unexpected CPU"):
            self.validate()

    def test_unrelated_device_work_cannot_disappear(self):
        self.control.append(self.event("gpu_memcpy", "Memcpy HtoD", 4))
        with self.assertRaisesRegex(ValueError, "unrelated device work changed"):
            self.validate()

    def test_identical_unrelated_device_work_is_allowed(self):
        other = self.event("kernel", "unrelated-kernel", 4)
        self.control.append(other)
        self.candidate.append(copy.deepcopy(other))
        self.assertEqual(self.validate()["unrelated_device_event_count"], 1)

    def test_unrelated_device_work_count_cannot_change(self):
        other = self.event("kernel", "unrelated-kernel", 4)
        self.control.extend([other, copy.deepcopy(other)])
        self.candidate.append(copy.deepcopy(other))
        with self.assertRaisesRegex(ValueError, "unrelated device work changed"):
            self.validate()


if __name__ == "__main__":
    unittest.main()
