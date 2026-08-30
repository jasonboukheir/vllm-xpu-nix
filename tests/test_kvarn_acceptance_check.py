import importlib.util
import json
import sys
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "scripts" / "kvarn_acceptance_check.py"
SPEC = importlib.util.spec_from_file_location("kvarn_acceptance_check", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def artifact_args(tmp_path, *, comparison_count=6):
    comparisons = []
    for index in range(comparison_count):
        comparisons.append(
            write_json(
                tmp_path / f"comparison-{index}.json",
                {
                    "status": "passed",
                    "acceptance": {"status": "passed"},
                    "decode_steps": 768,
                },
            )
        )

    services = {}
    for role, _option in MODULE.SERVICE_GATE_ARGUMENTS:
        services[role] = write_json(
            tmp_path / f"service-{role}.json",
            {"status": "passed"},
        )

    scans = {}
    for role, _option in MODULE.ENGINE_LOG_SCAN_ARGUMENTS:
        scans[role] = write_json(
            tmp_path / f"scan-{role}.json",
            {
                "status": "passed",
                "fatal_findings": [],
                "known_teardown_findings": [],
            },
        )

    argv = []
    for comparison in comparisons:
        argv.extend(("--comparison", str(comparison)))
    for role, option in MODULE.SERVICE_GATE_ARGUMENTS:
        argv.extend((f"--{option}", str(services[role])))
    for role, option in MODULE.ENGINE_LOG_SCAN_ARGUMENTS:
        argv.extend((f"--{option}", str(scans[role])))
    output = tmp_path / "acceptance.json"
    argv.extend(("--output", str(output), "--allow-tmp"))
    return argv, comparisons, services, scans, output


def test_complete_acceptance_bundle_passes_and_writes_same_json(tmp_path, capsys):
    argv, _comparisons, _services, scans, output = artifact_args(tmp_path)
    write_json(
        scans["b1_restart"],
        {
            "status": "passed",
            "fatal_findings": [],
            "known_teardown_findings": [{"kind": "traceback"}],
        },
    )

    assert MODULE.main(argv) == 0

    stdout = json.loads(capsys.readouterr().out)
    persisted = json.loads(output.read_text(encoding="utf-8"))
    assert stdout == persisted
    assert persisted["status"] == "passed"
    assert persisted["summary"] == {
        "comparison_files": 6,
        "required_comparison_files": 6,
        "scored_positions": 4608,
        "minimum_scored_positions": 4096,
        "service_gates": 5,
        "engine_log_scans": 3,
        "known_teardown_findings": 1,
    }
    assert persisted["failures"] == []


def test_failed_artifacts_and_insufficient_positions_exit_nonzero(tmp_path, capsys):
    argv, comparisons, services, scans, output = artifact_args(tmp_path)
    write_json(
        comparisons[0],
        {
            "status": "failed",
            "acceptance": {"status": "failed"},
            "decode_steps": 1,
        },
    )
    write_json(services["near_restart"], {"status": "running"})
    write_json(
        scans["b4"],
        {
            "status": "passed",
            "fatal_findings": [{"kind": "device_failure"}],
            "known_teardown_findings": [],
        },
    )

    assert MODULE.main(argv) == 1

    result = json.loads(capsys.readouterr().out)
    assert result == json.loads(output.read_text(encoding="utf-8"))
    assert result["status"] == "failed"
    assert result["summary"]["scored_positions"] == 3841
    assert {failure["scope"] for failure in result["failures"]} == {
        "comparison",
        "comparisons",
        "service_gate",
        "engine_log_scan",
    }
    assert any(
        "fatal_findings must be empty" in failure["message"]
        for failure in result["failures"]
    )


def test_missing_comparison_count_fails_machine_readably(tmp_path, capsys):
    argv, _comparisons, _services, _scans, output = artifact_args(
        tmp_path, comparison_count=5
    )

    assert MODULE.main(argv) == 1

    result = json.loads(capsys.readouterr().out)
    assert result == json.loads(output.read_text(encoding="utf-8"))
    assert result["summary"]["comparison_files"] == 5
    assert any(
        failure["message"] == "expected 6 files, got 5"
        for failure in result["failures"]
    )


def test_duplicate_comparison_paths_fail_machine_readably(tmp_path, capsys):
    argv, comparisons, _services, _scans, _output = artifact_args(tmp_path)
    argv[3] = str(comparisons[0])

    assert MODULE.main(argv) == 1

    result = json.loads(capsys.readouterr().out)
    assert result["summary"]["comparison_files"] == 6
    assert any(
        failure["message"] == "comparison paths must be distinct"
        for failure in result["failures"]
    )


def test_malformed_scan_schema_fails_closed(tmp_path, capsys):
    argv, _comparisons, _services, scans, _output = artifact_args(tmp_path)
    write_json(
        scans["b1_first"],
        {
            "status": "passed",
            "fatal_findings": None,
        },
    )

    assert MODULE.main(argv) == 1

    result = json.loads(capsys.readouterr().out)
    messages = [failure["message"] for failure in result["failures"]]
    assert "fatal_findings must be a list" in messages
    assert "known_teardown_findings must be a list" in messages
