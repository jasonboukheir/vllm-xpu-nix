from __future__ import annotations

import argparse

import pytest

from scripts import kvarn_correctness_run as correctness
from scripts import kvarn_perf_gate as gate
from scripts import kvarn_perf_run as perf
from scripts import kvarn_split_policy as policy
from scripts import kvarn_xpu_profile as profile


def _runtime_args() -> argparse.Namespace:
    return argparse.Namespace(
        native_split_policy="b70_q6_v2",
        native_splits={},
        native_layout="xe2_dpas",
        native_kernel_variant="q6_next_page_prefetch",
        native_output_dtype="bf16",
        native_frontend="reference",
        flush_index_materialization="per_layer",
        onednn_deterministic=True,
        max_model_len=65536,
        max_num_batched_tokens=2048,
        launcher_mode="runtime-factory",
        resolved_launchers={},
    )


def test_b70_wave_sweep_is_factory_only_and_records_evidence() -> None:
    contract = policy.factory_split_policy_contract("b70_wave_sweep")

    assert policy.NATIVE_SPLIT_POLICIES == ("fixed", "b70_q6", "b70_q6_v2")
    assert "b70_wave_sweep" not in policy.NATIVE_SPLIT_POLICIES
    assert contract["selector"] == "b70_wave_sweep"
    assert contract["selection_mode"] == "enumerate_all_candidates_no_winner"
    assert contract["candidate_num_kv_splits"] == [8, 16, 17, 24, 32]
    assert contract["winner"] is None
    assert contract["kernel_compatibility"] == {
        "kind": "exact_variant",
        "name": "q6_prefetch_record_cursor",
        "id": 18,
    }
    assert [item["sha256"] for item in contract["evidence"]] == [
        "eb307d22aba29adf68556013bf4bf1d8cf4e31e69e40967d35d56c04a0c07869",
        "ec92e73b7b1dd8aceae818dcfd32d5fff4024aa5059e5b9f52a4ff923df6c9aa",
        "034feed1e2d15a149573cd5f2cfec905be8bc4da24d6b8e6be2048c290a21a52",
    ]


def test_explicit_factory_contract_does_not_claim_a_winner() -> None:
    contract = policy.factory_split_policy_contract("explicit", [None, 8, 32])

    assert contract["selection_mode"] == "caller_explicit"
    assert contract["candidate_num_kv_splits"] == ["auto", 8, 32]
    assert contract["winner"] is None
    with pytest.raises(ValueError, match="requires split selections"):
        policy.factory_split_policy_contract("explicit")


def test_b70_q6_v2_contract_is_context_explicit_and_boundary_exact() -> None:
    contract = policy.split_policy_contract("b70_q6_v2")

    assert contract == {
        "schema_version": 1,
        "selector": "b70_q6_v2",
        "selection_axes": ["decode_batch_size", "context_tokens"],
        "supported_harness_batches": [1, 4],
        "scratch_max_splits": 32,
        "kernel_compatibility": {
            "kind": "exact_variants",
            "variants": [
                {"name": "q6_next_page_prefetch", "id": 12},
                {
                    "name": "q6_next_page_prefetch_split_reducer",
                    "id": 13,
                },
            ],
        },
        "rules": [
            {
                "batch": 1,
                "context_tokens_minimum": 1,
                "context_tokens_maximum_inclusive": None,
                "num_kv_splits": 32,
            },
            {
                "batch": 4,
                "context_tokens_minimum": 1,
                "context_tokens_maximum_inclusive": 49152,
                "num_kv_splits": 8,
            },
            {
                "batch": 4,
                "context_tokens_minimum": 49153,
                "context_tokens_maximum_inclusive": None,
                "num_kv_splits": 32,
            },
        ],
    }
    assert policy.nominal_splits_by_batch("b70_q6_v2") is None
    assert policy.effective_splits(
        "b70_q6_v2", batch=1, context_tokens=262144
    ) == 32
    assert policy.effective_splits(
        "b70_q6_v2", batch=4, context_tokens=49152
    ) == 8
    assert policy.effective_splits(
        "b70_q6_v2", batch=4, context_tokens=49153
    ) == 32


def test_b70_q6_v2_accepts_profiled_id12_and_id13_only() -> None:
    for kernel_variant in (
        "q6_next_page_prefetch",
        "q6_next_page_prefetch_split_reducer",
    ):
        policy.validate_kernel_compatibility(
            "b70_q6_v2",
            kernel_variant,
            q6_variants=perf.B70_Q6_KERNEL_VARIANTS,
        )
    with pytest.raises(ValueError, match="ID12.*ID13"):
        policy.validate_kernel_compatibility(
            "b70_q6_v2", "q6_scalar", q6_variants=perf.B70_Q6_KERNEL_VARIANTS
        )


def test_runtime_factory_resolves_v2_per_context_without_split_environment() -> None:
    args = _runtime_args()
    short = perf.PlannedRun(perf.Workload(49152, 4, 32, 4, 17), "candidate", 1)
    long = perf.PlannedRun(perf.Workload(49153, 4, 32, 4, 17), "candidate", 2)

    assert perf.native_splits_for_run(short, args) == 8
    assert perf.native_splits_for_run(long, args) == 32
    assert perf.native_max_splits_for_run(short, args) == 32
    assert perf.native_nominal_splits_by_batch(args) is None
    assert perf.runtime_factory_axes_for_run(short, args)["KVARN_FACTORY_SPLITS"] is None
    assert perf.runtime_factory_axes_for_run(short, args)[
        "KVARN_FACTORY_SPLIT_POLICY"
    ] == "b70_q6_v2"


@pytest.mark.parametrize(
    "kernel_variant",
    ["q6_next_page_prefetch", "q6_next_page_prefetch_split_reducer"],
)
def test_exploratory_perf_and_profile_clis_accept_v2_profiled_variants(
    tmp_path, monkeypatch: pytest.MonkeyPatch, kernel_variant: str
) -> None:
    candidate = tmp_path / "candidate"
    (candidate / "bin").mkdir(parents=True)
    for executable in ("vllm", "python"):
        (candidate / "bin" / executable).write_text("", encoding="utf-8")
    config = tmp_path / "config"
    config.mkdir()
    common = [
        "--candidate-env",
        str(candidate),
        "--allow-tmp",
        "--launcher-mode",
        "runtime-factory",
        "--native-layout",
        "xe2_dpas",
        "--native-kernel-variant",
        kernel_variant,
        "--native-split-policy",
        "b70_q6_v2",
        "--runtime-cache",
        str(tmp_path / "cache"),
        "--config-repo",
        str(config),
        "--config-ref",
        f"path:{config}",
    ]
    perf_args = perf.parse_args(
        [
            *common,
            "--exploratory",
            "--plan-only",
            "--context",
            "49152,49153",
            "--batch",
            "4",
            "--repeats",
            "2",
            "--output-dir",
            str(tmp_path / "perf"),
        ]
    )
    assert perf_args.native_splits == {}
    assert perf.native_nominal_splits_by_batch(perf_args) is None

    profile_args = profile.parse_args(
        [
            *common,
            "--context",
            "49153",
            "--batch",
            "4",
            "--output-dir",
            str(tmp_path / "profile"),
        ]
    )
    assert profile_args.native_splits == {}
    profile_run = perf.PlannedRun(
        perf.Workload(49153, 4, 96, 4, 17), "candidate", 1
    )
    assert perf.native_splits_for_run(profile_run, profile_args) == 32

    wrong_kernel = list(common)
    wrong_kernel[wrong_kernel.index(kernel_variant)] = "q6_scalar"
    with pytest.raises(SystemExit):
        perf.parse_args(
            [
                *wrong_kernel,
                "--exploratory",
                "--plan-only",
                "--context",
                "4096",
                "--batch",
                "4",
                "--repeats",
                "2",
                "--output-dir",
                str(tmp_path / "wrong-kernel"),
            ]
        )


def test_correctness_cli_records_v2_contract_without_nominal_map(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    candidate = tmp_path / "candidate"
    (candidate / "bin").mkdir(parents=True)
    for executable in ("vllm", "python"):
        (candidate / "bin" / executable).write_text("", encoding="utf-8")
    fixtures = tmp_path / "fixtures.json"
    fixtures.write_text("[]\n", encoding="utf-8")
    monkeypatch.setattr(
        correctness, "DEFAULT_FIXTURE_SHA256", correctness.sha256_file(fixtures)
    )
    factory = tmp_path / "factory.json"
    factory.write_text("{}\n", encoding="utf-8")
    config = tmp_path / "config"
    config.mkdir()

    args = correctness.parse_args(
        [
            "--candidate-env",
            str(candidate),
            "--factory-result",
            str(factory),
            "--fixtures",
            str(fixtures),
            "--allow-tmp",
            "--plan-only",
            "--launcher-mode",
            "runtime-factory",
            "--native-layout",
            "xe2_dpas",
            "--native-kernel-variant",
            "q6_next_page_prefetch",
            "--native-split-policy",
            "b70_q6_v2",
            "--runtime-cache",
            str(tmp_path / "cache"),
            "--config-repo",
            str(config),
            "--config-ref",
            f"path:{config}",
            "--output-dir",
            str(tmp_path / "correctness"),
        ]
    )

    assert args.native_splits == {}
    assert perf.native_nominal_splits_by_batch(args) is None
    assert perf.native_split_policy_contract(args)["scratch_max_splits"] == 32


def test_correctness_and_formal_gate_phase_schema_do_not_invent_nominal_map() -> None:
    args = _runtime_args()
    native_b4 = next(
        spec
        for spec in correctness.SERVICE_PLAN
        if spec.name == "native-65k-b4"
    )
    runtime_spec = correctness.service_spec_evidence(native_b4, args)
    gate_spec = gate._correctness_phase_spec(
        native_b4.name,
        args.native_layout,
        args.native_frontend,
        args.flush_index_materialization,
        args.native_kernel_variant,
        args.native_split_policy,
        args.native_splits,
    )

    assert runtime_spec == gate_spec
    assert runtime_spec["nominal_decode_splits"] is None
    assert gate_spec["nominal_decode_splits"] is None
    assert runtime_spec["native_split_policy_contract"] == gate_spec[
        "native_split_policy_contract"
    ]
    assert runtime_spec["max_decode_splits"] == gate_spec["max_decode_splits"] == 32
