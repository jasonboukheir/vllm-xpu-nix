import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).parents[1] / "scripts" / "quantization_resources.py"
SPEC = importlib.util.spec_from_file_location("quantization_resources", SCRIPT)
resources = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(resources)


class ResourceTests(unittest.TestCase):
    def test_solver_accounts_for_all_bounded_pools(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            resources, "meminfo", return_value={"MemAvailable": 100 * resources.GIB}
        ), patch.object(resources, "_cgroup_value", return_value=None):
            result = resources.solve_resources(
                {"pinned_staging_mib": 512, "read_queue_mib": 128,
                 "write_queue_mib": 128, "reorder_queue_mib": 128,
                 "min_free_gib": 0, "min_free_percent": 0},
                {"host_mem_available_floor_gib": 24, "memory_max_gib": 76},
                {"model_optimizer_bytes": resources.GIB, "safety_margin_bytes": resources.GIB},
                path=Path(directory),
            )
            self.assertEqual(result["memory"]["parts"]["pinned_staging"], 512 * resources.MIB)
            self.assertLessEqual(result["memory"]["minimum"] + result["memory"]["pageable_lru"], result["memory"]["usable"])

    def test_pressure_state_machine_and_hysteresis(self):
        controller = resources.PressureController({
            "host_mem_available_floor_gib": 24, "host_mem_abort_floor_gib": 12,
            "host_mem_resume_gib": 28, "pressure_grace_seconds": 10,
            "resume_settle_seconds": 5,
        })
        self.assertEqual(controller.sample(20 * resources.GIB, now=0).state, "DRAIN")
        self.assertEqual(controller.sample(20 * resources.GIB, now=11).state, "PAUSE")
        self.assertEqual(controller.sample(30 * resources.GIB, now=12).state, "PAUSE")
        self.assertEqual(controller.sample(30 * resources.GIB, now=18).state, "RUN")
        self.assertEqual(controller.sample(10 * resources.GIB, now=19).state, "STOP")


if __name__ == "__main__":
    unittest.main()
