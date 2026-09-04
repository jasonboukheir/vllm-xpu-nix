from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import kvarn_factory_run as factory


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _repo(path: Path) -> Path:
    path.mkdir()
    _git(path, "init", "--quiet")
    _git(path, "config", "user.name", "Factory Test")
    _git(path, "config", "user.email", "factory@example.invalid")
    (path / "tracked").write_text("clean\n", encoding="utf-8")
    _git(path, "add", "tracked")
    _git(
        path,
        "-c",
        "commit.gpgsign=false",
        "commit",
        "--quiet",
        "-m",
        "initial",
    )
    return path


def test_named_factory_ids_are_complete_and_stable() -> None:
    assert {name: spec.variant_id for name, spec in factory.VARIANTS.items()} == {
        "baseline": 0,
        "qk_i8u4": 1,
        "q6_scalar": 2,
        "q8_vector": 3,
        "q6_vector": 4,
        "page128": 5,
    }
    assert set(factory.VARIANTS_BY_ID) == set(range(6))
    assert all(spec.dpas_layout for spec in factory.VARIANTS.values())
    assert factory.DEFAULT_VARIANT_NAMES == tuple(list(factory.VARIANTS)[:5])


def test_variant_parser_accepts_names_and_all_but_not_numeric_aliases() -> None:
    assert [
        item.variant_id for item in factory.parse_variants("baseline,q8_vector")
    ] == [
        0,
        3,
    ]
    assert [item.variant_id for item in factory.parse_variants("all")] == list(range(5))
    assert factory.parse_variants("page128") == [factory.VARIANTS["page128"]]
    with pytest.raises(factory.FactoryError, match="unknown variant"):
        factory.parse_variants("3")


def test_matrix_expands_auto_and_explicit_split_sweeps() -> None:
    cases = factory.build_matrix(
        batches=[1, 4],
        contexts=[4096],
        splits=[None, 8],
        variants=[factory.VARIANTS["baseline"], factory.VARIANTS["q8_vector"]],
    )
    assert [
        (case.batch, case.requested_splits, case.variant.variant_id) for case in cases
    ] == [
        (1, 24, 0),
        (1, 24, 3),
        (1, 8, 0),
        (1, 8, 3),
        (4, 16, 0),
        (4, 16, 3),
        (4, 8, 0),
        (4, 8, 3),
    ]
    assert all(case.effective_splits == case.requested_splits for case in cases)
    assert factory.effective_split_count(64, 24) == 1
    with pytest.raises(factory.FactoryError, match="only B1 and B4"):
        factory.build_matrix(
            batches=[2],
            contexts=[4096],
            splits=[None],
            variants=[factory.VARIANTS["baseline"]],
        )


def test_split_parser_is_fail_closed() -> None:
    assert factory.parse_split_tokens("auto,16,24,16") == [None, 16, 24]
    for invalid in ("", "3", "16,", "environment"):
        with pytest.raises(factory.FactoryError):
            factory.parse_split_tokens(invalid)


def test_interleaved_order_is_rotating_and_palindromic() -> None:
    names = ["reference", "candidate"]
    assert factory.interleaved_order(names, 0) == (
        "reference",
        "candidate",
        "candidate",
        "reference",
    )
    assert factory.interleaved_order(names, 1) == (
        "candidate",
        "reference",
        "reference",
        "candidate",
    )
    three = factory.interleaved_order(["natural", "candidate", "auto"], 0)
    assert three == tuple(reversed(three))


def test_exact_b70_identity_is_mandatory() -> None:
    factory.validate_device_identity(
        available=True,
        count=1,
        selected_name=factory.EXPECTED_DEVICE_NAME,
    )
    with pytest.raises(factory.FactoryError, match="real XPU"):
        factory.validate_device_identity(
            available=False,
            count=0,
            selected_name="",
        )
    with pytest.raises(factory.FactoryError, match="expected exact device"):
        factory.validate_device_identity(
            available=True,
            count=1,
            selected_name="Intel(R) Data Center GPU Max 1550",
        )


class _ProbeResult:
    def __init__(self, value: float) -> None:
        self.value = value

    def cpu(self) -> _ProbeResult:
        return self

    def item(self) -> float:
        return self.value


class _ProbeTensor:
    device = SimpleNamespace(type="xpu")

    def __init__(self, value: float = 212.0) -> None:
        self.value = value

    def square(self) -> _ProbeTensor:
        return self

    def __add__(self, _value: int) -> _ProbeTensor:
        return self

    def sum(self) -> _ProbeResult:
        return _ProbeResult(self.value)


class _FakeXpu:
    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def device_count() -> int:
        return 1

    @staticmethod
    def get_device_name(_index: int) -> str:
        return factory.EXPECTED_DEVICE_NAME

    @staticmethod
    def synchronize() -> None:
        return None


class _FakeTorchProbe:
    xpu = _FakeXpu()
    float32 = "float32"

    @staticmethod
    def arange(*_args, **_kwargs) -> _ProbeTensor:
        return _ProbeTensor()


def test_xpu_preflight_requires_a_successful_tensor_operation() -> None:
    evidence = factory.preflight_xpu(_FakeTorchProbe())
    assert evidence["passed"] is True
    assert evidence["tensor_op_result"] == 212.0

    class BadProbe(_FakeTorchProbe):
        @staticmethod
        def arange(*_args, **_kwargs) -> _ProbeTensor:
            return _ProbeTensor(211.0)

    with pytest.raises(factory.FactoryError, match="invalid evidence"):
        factory.preflight_xpu(BadProbe())


def test_native_decode_invocation_uses_all_explicit_factory_args() -> None:
    calls: list[tuple] = []

    def operation(*args) -> None:
        calls.append(args)

    values = {
        name: object()
        for name in (
            "query",
            "cache",
            "block_table",
            "seq_lens",
            "block_to_slot",
            "tail_key",
            "tail_value",
            "temp_output",
            "exp_sums",
            "max_logits",
            "output",
        )
    }
    factory.invoke_native_decode(
        operation,
        **values,
        context=65_023,
        unrotate_output=True,
        num_kv_splits=24,
        kernel_variant=3,
        dpas_layout=True,
    )
    assert len(calls) == 1
    assert calls[0][-3:] == (24, 3, True)
    assert calls[0][-6:-3] == (factory.SOFTMAX_SCALE, True, False)


class _Packet:
    def __init__(self, schema: str) -> None:
        self.default = SimpleNamespace(_schema=schema)


def _fake_torch_ops(native_schema: str, *, fused: bool) -> SimpleNamespace:
    flash = SimpleNamespace(
        varlen_fwd=_Packet("varlen_fwd(...)"),
        kvarn_decode_with_scratch=_Packet(native_schema),
        kvarn_hadamard=_Packet("kvarn_hadamard(...)"),
        kvarn_hadamard_scatter=_Packet("kvarn_hadamard_scatter(... dpas_layout)"),
    )
    if fused:
        flash.kvarn_hadamard_qkv_scatter = _Packet(
            "kvarn_hadamard_qkv_scatter(... dpas_layout)"
        )
    return SimpleNamespace(
        ops=SimpleNamespace(
            load_library=lambda _path: None,
            _C_cache_ops=SimpleNamespace(
                reshape_and_cache_flash=_Packet("reshape_and_cache_flash(...)")
            ),
            _vllm_fa2_C=flash,
        )
    )


def test_operator_loader_requires_explicit_abi_and_detects_fused_symbol(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base.so"
    flash = tmp_path / "flash.so"
    torch_module = _fake_torch_ops(
        "kvarn_decode_with_scratch(... num_kv_splits, kernel_variant, dpas_layout)",
        fused=True,
    )
    operations, schemas = factory.load_operators(
        torch_module,
        base_library=base,
        flash_library=flash,
    )
    assert operations.fused_qkv_scatter is not None
    assert "kernel_variant" in schemas["kvarn_decode_with_scratch"]

    legacy = _fake_torch_ops("kvarn_decode_with_scratch(...)", fused=False)
    with pytest.raises(factory.FactoryError, match="explicit factory ABI"):
        factory.load_operators(
            legacy,
            base_library=base,
            flash_library=flash,
        )


def test_distinct_layout_cache_guard_rejects_alias() -> None:
    first = SimpleNamespace(data_ptr=lambda: 17)
    second = SimpleNamespace(data_ptr=lambda: 23)
    factory.ensure_distinct_storage(first, second)
    with pytest.raises(factory.FactoryError, match="caches alias"):
        factory.ensure_distinct_storage(first, first)


def test_fixture_helper_must_come_from_attested_repo(tmp_path: Path) -> None:
    expected = tmp_path / "benchmark" / "kvarn_utils.py"
    expected.parent.mkdir()
    expected.write_text("# fixture\n", encoding="utf-8")
    module = SimpleNamespace(__name__="benchmark.kvarn_utils", __file__=str(expected))
    factory._require_module_from_repo(
        module, tmp_path, Path("benchmark/kvarn_utils.py")
    )

    outside = tmp_path / "installed.py"
    outside.write_text("# wrong source\n", encoding="utf-8")
    module.__file__ = str(outside)
    with pytest.raises(factory.FactoryError, match="came from"):
        factory._require_module_from_repo(
            module, tmp_path, Path("benchmark/kvarn_utils.py")
        )


def test_build_and_source_provenance_is_exact(tmp_path: Path) -> None:
    attestation = factory.validate_build_attestation(
        label="flash",
        derivation="/nix/store/0123456789abcdfghijklmnpqrsvwxyz-kernels.drv",
        closure_sha256="a" * 64,
    )
    assert attestation["attestation_source"] == "explicit_cli"
    with pytest.raises(factory.FactoryError, match="Nix .drv"):
        factory.validate_build_attestation(
            label="flash",
            derivation="result",
            closure_sha256="a" * 64,
        )

    repo = _repo(tmp_path / "repo")
    clean_state = factory.repository_state("kernels", repo)
    assert len(clean_state["head"]) == 40
    factory.require_clean_repositories([clean_state])

    (repo / "dirty").write_text("dirty\n", encoding="utf-8")
    state = factory.repository_state("kernels", repo)
    assert state["dirty"] is True
    assert state["status_porcelain"] == ["?? dirty"]
    assert state["status_sha256"] == hashlib.sha256(b"?? dirty").hexdigest()
    with pytest.raises(factory.FactoryError, match="requires clean"):
        factory.require_clean_repositories([state])

    libraries = {
        "base": {"path": "/base", "sha256": "1" * 64, "size_bytes": 1, "mtime_ns": 2},
        "flash": {
            "path": "/flash",
            "sha256": "2" * 64,
            "size_bytes": 3,
            "mtime_ns": 4,
        },
    }
    identity = factory.evidence_identity(libraries, [state])
    assert len(identity) == 64
    changed = {**libraries, "flash": {**libraries["flash"], "sha256": "3" * 64}}
    assert factory.evidence_identity(changed, [state]) != identity


def test_nix_artifact_must_belong_to_attested_derivation_and_closure() -> None:
    output = "/nix/store/0123456789abcdfghijklmnpqrsvwxyz-kernels"
    library = Path(output) / "lib/python3.12/site-packages/kernels.so"
    derivation = "/nix/store/zyxwvutsrqpnmlkjihgfdcba98765432-kernels.drv"
    closure = [output, "/nix/store/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-torch"]
    expected_digest = factory.closure_digest(closure)
    attestation = factory.validate_build_attestation(
        label="flash",
        derivation=derivation,
        closure_sha256=expected_digest,
    )
    calls: list[tuple[str, ...]] = []

    def command_runner(command) -> str:
        call = tuple(command)
        calls.append(call)
        if call == ("nix-store", "-q", "--outputs", derivation):
            return output
        if call == ("nix-store", "-q", "--deriver", output):
            return derivation
        if call == ("nix-store", "-qR", output):
            return "\n".join(reversed(closure))
        raise AssertionError(call)

    verified = factory.verify_nix_artifact(
        label="flash",
        library=library,
        attestation=attestation,
        command_runner=command_runner,
    )
    assert verified["verified"] is True
    assert verified["output_path"] == output
    assert verified["closure_paths"] == sorted(closure)
    assert verified["closure_sha256"] == expected_digest
    assert len(calls) == 3

    def wrong_deriver(command) -> str:
        if "--outputs" in command:
            return output
        if "--deriver" in command:
            return "/nix/store/bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb-wrong.drv"
        return "\n".join(closure)

    with pytest.raises(factory.FactoryError, match="deriver mismatch"):
        factory.verify_nix_artifact(
            label="flash",
            library=library,
            attestation=attestation,
            command_runner=wrong_deriver,
        )

    bad_digest = {**attestation, "closure_sha256": "0" * 64}
    with pytest.raises(factory.FactoryError, match="closure digest mismatch"):
        factory.verify_nix_artifact(
            label="flash",
            library=library,
            attestation=bad_digest,
            command_runner=command_runner,
        )


def test_nix_artifact_rejects_non_store_library() -> None:
    with pytest.raises(factory.FactoryError, match="not in the Nix store"):
        factory.nix_store_root(Path("/tmp/kernel.so"))


def test_focused_xpu_kill_suite_is_bound_to_library_and_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "kernels"
    for selection in factory.FOCUSED_XPU_TESTS:
        relative = selection.partition("::")[0]
        source = repo / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        source.touch()
    library = tmp_path / "flash.so"
    library.touch()
    captured: dict = {}

    def passing_run(command, **kwargs):
        captured["command"] = command
        captured.update(kwargs)
        return SimpleNamespace(
            returncode=0,
            stdout=f"{factory.FOCUSED_XPU_MIN_PASSED} passed in 1.00s\n",
            stderr="",
        )

    monkeypatch.setattr(factory.subprocess, "run", passing_run)
    result = factory.run_focused_xpu_kill_suite(
        kernels_repo=repo, flash_library=library
    )
    factory.require_focused_xpu_kill_suite(result)
    assert result["passed"] is True
    assert result["passed_count"] == factory.FOCUSED_XPU_MIN_PASSED
    assert captured["cwd"] == repo.resolve()
    assert captured["env"]["VLLM_XPU_KERNELS_LIBRARY"] == str(library.resolve())
    assert "no:cacheprovider" in captured["command"]
    assert all(str(repo.resolve()) in item for item in captured["command"][7:])

    def skipped_run(_command, **_kwargs):
        return SimpleNamespace(
            returncode=0,
            stdout=f"{factory.FOCUSED_XPU_MIN_PASSED} passed, 1 skipped in 1.00s\n",
            stderr="",
        )

    monkeypatch.setattr(factory.subprocess, "run", skipped_run)
    skipped = factory.run_focused_xpu_kill_suite(
        kernels_repo=repo, flash_library=library
    )
    assert skipped["passed"] is False
    with pytest.raises(factory.FactoryError, match="kill suite failed"):
        factory.require_focused_xpu_kill_suite(skipped)


def test_fixture_and_fusion_results_cannot_claim_matched_parity() -> None:
    source = Path(factory.__file__).read_text(encoding="utf-8")
    assert 'logical_kv_payloads_matched_between_auto_and_kvarn": False' in source
    assert 'matched_parity_eligible": False' in source
    assert "candidate_separate_device_stage_over_auto" in source
    assert "candidate_fused_device_stage_over_auto" in source
    assert '"candidate_device_stage_over_auto"' not in source


def test_library_hash_and_atomic_durable_output(tmp_path: Path) -> None:
    library = tmp_path / "kernel.so"
    library.write_bytes(b"compiled-kernel")
    record = factory.stable_file_record(library)
    assert record["sha256"] == hashlib.sha256(b"compiled-kernel").hexdigest()

    output = factory.ensure_durable_output(tmp_path / "run.json", allow_tmp=True)
    factory.write_json_atomic(output, {"status": "running"})
    assert json.loads(output.read_text(encoding="utf-8")) == {"status": "running"}
    assert not list(tmp_path.glob(".run.json.*.tmp"))
    with pytest.raises(factory.FactoryError, match="outside /tmp"):
        factory.ensure_durable_output(Path("/tmp/factory-result.json"), allow_tmp=False)


def test_execution_failure_is_durable_and_nonzero(tmp_path: Path) -> None:
    output = tmp_path / "failed.json"
    derivation = "/nix/store/0123456789abcdfghijklmnpqrsvwxyz-test.drv"
    args = factory.parse_args(
        [
            "--base-library",
            str(tmp_path / "missing-base.so"),
            "--flash-library",
            str(tmp_path / "missing-flash.so"),
            "--base-derivation",
            derivation,
            "--base-closure-sha256",
            "1" * 64,
            "--flash-derivation",
            derivation,
            "--flash-closure-sha256",
            "2" * 64,
            "--output",
            str(output),
            "--allow-tmp",
        ]
    )

    assert factory.execute(args) == 2
    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["status"] == "failed"
    assert document["error"]["type"] == "FileNotFoundError"
    assert document["results"] == []


def test_scope_can_never_be_mistaken_for_service_parity() -> None:
    assert "not service-performance or parity evidence" in factory.SCOPE_WARNING
    source = Path(factory.__file__).read_text(encoding="utf-8")
    assert "KVARN_NATIVE_XPU" not in source
    assert "VLLM_XPU_KERNELS_LIBRARY" in source
