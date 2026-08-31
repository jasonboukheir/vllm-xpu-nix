import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "kvarn_scan_engine_log.py"
SPEC = importlib.util.spec_from_file_location("kvarn_scan_engine_log", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_clean_log_passes():
    result = MODULE.scan(["INFO request complete", "INFO shutdown complete"])
    assert result["status"] == "passed"
    assert result["fatal_findings"] == []
    assert result["known_teardown_findings"] == []


def test_known_vllm_shutdown_race_is_classified_without_hiding_it():
    result = MODULE.scan(
        [
            "INFO [shutdown] API server: shutdown triggered",
            "INFO [shutdown] EngineCore: request processing complete; teardown",
            "ERROR AsyncLLM output_handler failed.",
            "ERROR Traceback (most recent call last):",
            (
                "ERROR vllm.v1.engine.exceptions.EngineDeadError: "
                "EngineCore encountered an issue."
            ),
            "INFO Application shutdown complete.",
        ]
    )
    assert result["status"] == "passed"
    assert result["fatal_findings"] == []
    assert [item["kind"] for item in result["known_teardown_findings"]] == ["traceback"]


def test_engine_dead_traceback_before_shutdown_fails():
    result = MODULE.scan(
        [
            "ERROR AsyncLLM output_handler failed.",
            "ERROR Traceback (most recent call last):",
            (
                "ERROR vllm.v1.engine.exceptions.EngineDeadError: "
                "EngineCore encountered an issue."
            ),
        ]
    )
    assert result["status"] == "failed"
    kinds = {item["kind"] for item in result["fatal_findings"]}
    assert {"traceback", "engine_dead_error", "error_level"} <= kinds


def test_numerical_and_runtime_failures_fail():
    result = MODULE.scan(
        [
            "ERROR result contains NaN",
            "ERROR device lost",
            "Fatal Python error: Aborted",
            "RuntimeError: out of memory",
            "ERROR non-finite decoder result",
            "AssertionError: wrong token",
            "EngineDeadError without a traceback",
        ]
    )
    assert result["status"] == "failed"
    assert {item["kind"] for item in result["fatal_findings"]} == {
        "non_finite",
        "device_failure",
        "fatal_python",
        "out_of_memory",
        "runtime_error",
        "error_level",
        "explicit_non_finite",
        "assertion_error",
        "engine_dead_error",
    }
