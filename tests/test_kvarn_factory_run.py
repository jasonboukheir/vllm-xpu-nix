from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import kvarn_factory_run as factory

PROJECT_REVISION = "1" * 40
VLLM_REVISION = "2" * 40
KERNELS_REVISION = "3" * 40


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
        "q6_cached_weights": 6,
        "q6_exact_rows": 7,
        "q6_cached_weights_exact_rows": 8,
        "q6_page_pair": 9,
        "q6_main_grf128": 10,
        "q6_split_reducer_specialized": 11,
        "q6_next_page_prefetch": 12,
    }
    assert set(factory.VARIANTS_BY_ID) == set(range(13))
    assert all(spec.dpas_layout for spec in factory.VARIANTS.values())
    assert factory.DEFAULT_VARIANT_NAMES == (
        "q6_scalar",
        "q6_vector",
        "q6_cached_weights",
        "q6_exact_rows",
        "q6_cached_weights_exact_rows",
    )
    assert factory.ALL_VARIANT_NAMES == (
        *factory.DEFAULT_VARIANT_NAMES,
        "q6_page_pair",
        "q6_main_grf128",
        "q6_split_reducer_specialized",
        "q6_next_page_prefetch",
    )


def test_variant_parser_accepts_names_and_all_but_not_numeric_aliases() -> None:
    assert [
        item.variant_id for item in factory.parse_variants("baseline,q8_vector")
    ] == [
        0,
        3,
    ]
    assert [item.variant_id for item in factory.parse_variants("all")] == [
        2,
        4,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
    ]
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
    assert all(case.output_dtype == "fp16" for case in cases)
    assert factory.effective_split_count(64, 24) == 1
    with pytest.raises(factory.FactoryError, match="only B1 and B4"):
        factory.build_matrix(
            batches=[2],
            contexts=[4096],
            splits=[None],
            variants=[factory.VARIANTS["baseline"]],
        )


def test_output_dtype_is_an_explicit_runtime_matrix_axis() -> None:
    assert factory.parse_output_dtypes("bf16,fp16,bf16") == ["bf16", "fp16"]
    cases = factory.build_matrix(
        batches=[1],
        contexts=[4096],
        splits=[32],
        variants=[factory.VARIANTS["q6_scalar"]],
        output_dtypes=["fp16", "bf16"],
    )
    assert [case.output_dtype for case in cases] == ["fp16", "bf16"]
    assert [case.as_dict()["output_dtype"] for case in cases] == ["fp16", "bf16"]
    with pytest.raises(factory.FactoryError, match="multi-split"):
        factory.build_matrix(
            batches=[1],
            contexts=[1],
            splits=[32],
            variants=[factory.VARIANTS["q6_scalar"]],
            output_dtypes=["bf16"],
        )
    with pytest.raises(factory.FactoryError, match="unsupported output dtype"):
        factory.parse_output_dtypes("tf32")


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
        write_bf16_output=True,
        num_kv_splits=24,
        kernel_variant=3,
        dpas_layout=True,
    )
    assert len(calls) == 1
    assert calls[0][-3:] == (24, 3, True)
    assert calls[0][-6:-3] == (factory.SOFTMAX_SCALE, True, True)


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


def test_natural_and_dpas_metadata_must_match_exactly() -> None:
    class FakeTorch:
        @staticmethod
        def equal(left, right) -> bool:
            return left == right

    natural_key = {"s_col_K": 1, "zp_K": 2, "s_row_K": 3}
    natural_value = {"s_col_V": 4, "s_row_V": 5, "zp_V": 6}
    factory._require_same_quantization_metadata(
        FakeTorch(),
        natural_key,
        natural_value,
        dict(natural_key),
        dict(natural_value),
    )
    with pytest.raises(factory.FactoryError, match="K metadata diverged"):
        factory._require_same_quantization_metadata(
            FakeTorch(),
            natural_key,
            natural_value,
            {**natural_key, "s_row_K": 99},
            dict(natural_value),
        )


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
    builds = {
        "package": {
            "derivation": "/nix/store/package.drv",
            "closure_sha256": "4" * 64,
            "output_path": "/nix/store/package",
            "actual_deriver": "/nix/store/package.drv",
        }
    }
    assert factory.evidence_identity(libraries, [state], builds) != identity


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


def test_nix_package_output_must_belong_to_attested_derivation_and_closure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = Path("/nix/store/0123456789abcdfghijklmnpqrsvwxyz-vllm")
    derivation = "/nix/store/zyxwvutsrqpnmlkjihgfdcba98765432-vllm.drv"
    closure = [str(output), "/nix/store/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-torch"]
    attestation = factory.validate_build_attestation(
        label="package",
        derivation=derivation,
        closure_sha256=factory.closure_digest(closure),
    )
    monkeypatch.setattr(Path, "resolve", lambda self, strict=False: self)

    def command_runner(command) -> str:
        call = tuple(command)
        if call == ("nix-store", "-q", "--outputs", derivation):
            return str(output)
        if call == ("nix-store", "-q", "--deriver", str(output)):
            return derivation
        if call == ("nix-store", "-qR", str(output)):
            return "\n".join(reversed(closure))
        raise AssertionError(call)

    verified = factory.verify_nix_output(
        label="package",
        output=output,
        attestation=attestation,
        command_runner=command_runner,
    )
    assert verified["verified"] is True
    assert verified["output_path"] == str(output)


def test_source_ownership_binds_package_and_kernel_outputs_to_correct_repos() -> None:
    package_output = "/nix/store/" + "p" * 32 + "-vllm"
    base_output = "/nix/store/" + "b" * 32 + "-base"
    flash_output = "/nix/store/" + "f" * 32 + "-flash"
    package = {
        "derivation": "/nix/store/" + "d" * 32 + "-vllm.g" + VLLM_REVISION[:7] + ".drv",
        "output_path": package_output,
        "closure_paths": [package_output, base_output, flash_output],
    }
    base = {
        "derivation": "/nix/store/"
        + "e" * 32
        + "-kernels.g"
        + KERNELS_REVISION[:7]
        + ".drv",
        "output_path": base_output,
    }
    flash = {
        "derivation": "/nix/store/"
        + "a" * 32
        + "-attention.g"
        + KERNELS_REVISION[:7]
        + ".drv",
        "output_path": flash_output,
    }
    expected = {
        "vllm-xpu-nix": PROJECT_REVISION,
        "vllm": VLLM_REVISION,
        "vllm-xpu-kernels": KERNELS_REVISION,
    }
    ownership = factory.verify_source_ownership(
        package=package, base=base, flash=flash, expected_revisions=expected
    )
    assert ownership["verified"] is True
    assert ownership["artifacts"]["base"]["repository"] == "vllm-xpu-kernels"
    with pytest.raises(factory.FactoryError, match="base source ownership mismatch"):
        factory.verify_source_ownership(
            package=package,
            base={
                **base,
                "derivation": base["derivation"].replace(
                    KERNELS_REVISION[:7], VLLM_REVISION[:7]
                ),
            },
            flash=flash,
            expected_revisions=expected,
        )
    with pytest.raises(factory.FactoryError, match="absent from"):
        factory.verify_source_ownership(
            package={**package, "closure_paths": [package_output, flash_output]},
            base=base,
            flash=flash,
            expected_revisions=expected,
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


def _valid_matched_manifest() -> dict:
    shape = [1, 128, factory.H_KV, factory.HEAD_DIM]
    tensor_bytes = 2
    for extent in shape:
        tensor_bytes *= extent
    source = {
        "path": "/source.py",
        "sha256": "e" * 64,
        "size_bytes": 1,
        "mtime_ns": 1,
    }
    manifest = {
        "fixture_mode": factory.MATCHED_FIXTURE_MODE,
        "logical_shape": shape,
        "logical_dtype": "torch.bfloat16",
        "generator": {
            "algorithm": "torch CPU Generator.randn request-major",
            "key_seed": factory.CORPUS_SEED,
            "value_seed": factory.CORPUS_SEED + 1,
            "generation_dtype": "torch.bfloat16",
        },
        "key_sha256": "a" * 64,
        "value_sha256": "b" * 64,
        "logical_bytes_hashed": {"key": tensor_bytes, "value": tensor_bytes},
        "auto_key_cache_sha256": "f" * 64,
        "auto_value_cache_sha256": "0" * 64,
        "natural_packed_cache_sha256": "c" * 64,
        "dpas_packed_cache_sha256": "d" * 64,
        "production_packer_sources": {
            "config": dict(source),
            "sinkhorn": dict(source),
            "store": dict(source),
        },
        "tail_mapping_by_context": {
            "128": {
                "validated": True,
                "block_to_slot_sha256": "1" * 64,
                "tail_key_sha256": "2" * 64,
                "tail_value_sha256": "3" * 64,
            }
        },
        "invariants": {
            "auto_populated_by_reshape_and_cache_flash": True,
            "natural_populated_by_production_packers": True,
            "dpas_populated_by_production_packers": True,
            "natural_and_dpas_share_sinkhorn_results": True,
            "sink_page_mapped_to_fp16_pool": True,
            "current_tail_mapped_to_fp16_pool": True,
            "partial_tail_valid_token_counts_verified": True,
            "all_setup_and_hashing_outside_timing": True,
        },
    }
    manifest["logical_corpus_sha256"] = factory._corpus_identity(manifest)
    return manifest


def test_matched_corpus_manifest_is_fail_closed() -> None:
    manifest = _valid_matched_manifest()
    factory.validate_matched_corpus_manifest(manifest)

    incomplete_hash = dict(manifest)
    incomplete_hash["logical_bytes_hashed"] = {"key": 1, "value": 1}
    incomplete_hash["logical_corpus_sha256"] = factory._corpus_identity(incomplete_hash)
    with pytest.raises(factory.FactoryError, match="every logical BF16 byte"):
        factory.validate_matched_corpus_manifest(incomplete_hash)

    unchecked_tail = {**manifest, "tail_mapping_by_context": {"128": {}}}
    with pytest.raises(factory.FactoryError, match="tail mappings"):
        factory.validate_matched_corpus_manifest(unchecked_tail)

    false_invariant = {
        **manifest,
        "invariants": {
            **manifest["invariants"],
            "natural_and_dpas_share_sinkhorn_results": False,
        },
    }
    with pytest.raises(factory.FactoryError, match="share_sinkhorn"):
        factory.validate_matched_corpus_manifest(false_invariant)


def test_tail_assignments_model_sink_and_partial_or_full_current_tail() -> None:
    one_page = factory.tail_page_assignments(
        batch=1, context=128, pages_per_request=512
    )
    assert one_page == [
        {
            "request": 0,
            "local_page": 0,
            "physical_page": 0,
            "pool_slot": 0,
            "roles": ["sink", "current_tail"],
            "valid_tokens": 128,
        }
    ]

    ragged = factory.tail_page_assignments(
        batch=2, context=65_023, pages_per_request=512
    )
    assert [
        (item["physical_page"], item["roles"], item["valid_tokens"]) for item in ragged
    ] == [
        (0, ["sink"], 128),
        (507, ["current_tail"], 127),
        (512, ["sink"], 128),
        (1019, ["current_tail"], 127),
    ]
    assert [item["pool_slot"] for item in ragged] == [0, 1, 2, 3]
    with pytest.raises(factory.FactoryError, match="exceeds"):
        factory.tail_page_assignments(batch=1, context=129, pages_per_request=1)


def test_matched_fixture_is_default_and_unmatched_is_explicit_diagnostic(
    tmp_path: Path,
) -> None:
    derivation = "/nix/store/0123456789abcdfghijklmnpqrsvwxyz-test.drv"
    common = [
        "--package-output",
        str(tmp_path / "package"),
        "--package-derivation",
        derivation,
        "--package-closure-sha256",
        "0" * 64,
        "--base-library",
        str(tmp_path / "base.so"),
        "--flash-library",
        str(tmp_path / "flash.so"),
        "--base-derivation",
        derivation,
        "--base-closure-sha256",
        "1" * 64,
        "--flash-derivation",
        derivation,
        "--flash-closure-sha256",
        "2" * 64,
        "--expected-vllm-xpu-nix-revision",
        PROJECT_REVISION,
        "--expected-vllm-revision",
        VLLM_REVISION,
        "--expected-kernels-revision",
        KERNELS_REVISION,
        "--output",
        str(tmp_path / "result.json"),
        "--allow-tmp",
    ]
    matched = factory.parse_args(common)
    assert matched.fixture_mode == factory.MATCHED_FIXTURE_MODE
    assert matched.auto_block_size == 64
    assert matched.output_dtype_values == ["bf16"]
    assert matched.warmup_rounds == 16
    assert matched.sample_rounds == 20
    with pytest.raises(SystemExit):
        factory.parse_args([*common, "--auto-block-size", "832"])
    diagnostic = factory.parse_args(
        [
            *common,
            "--fixture-mode",
            factory.UNMATCHED_FIXTURE_MODE,
            "--auto-block-size",
            "832",
        ]
    )
    assert diagnostic.fixture_mode == factory.UNMATCHED_FIXTURE_MODE
    assert diagnostic.auto_block_size == 832


def test_fixture_and_fusion_results_label_diagnostic_mode_clearly() -> None:
    source = Path(factory.__file__).read_text(encoding="utf-8")
    assert factory.MATCHED_FIXTURE_MODE in source
    assert factory.UNMATCHED_FIXTURE_MODE in source
    assert "production_sinkhorn_rtn_from_shared_logical_corpus" in source
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


def test_execution_failure_is_durable_and_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "failed.json"
    package = tmp_path / "package"
    package.mkdir()
    derivation = "/nix/store/0123456789abcdfghijklmnpqrsvwxyz-test.drv"
    args = factory.parse_args(
        [
            "--package-output",
            str(package),
            "--package-derivation",
            derivation,
            "--package-closure-sha256",
            "0" * 64,
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
            "--expected-vllm-xpu-nix-revision",
            PROJECT_REVISION,
            "--expected-vllm-revision",
            VLLM_REVISION,
            "--expected-kernels-revision",
            KERNELS_REVISION,
            "--output",
            str(output),
            "--allow-tmp",
        ]
    )

    monkeypatch.setattr(
        factory,
        "verify_nix_output",
        lambda **_kwargs: {"verified": True},
    )

    assert factory.execute(args) == 2
    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["status"] == "failed"
    assert document["error"]["type"] == "FileNotFoundError"
    assert document["results"] == []


def test_repository_revisions_require_full_exact_commits() -> None:
    repositories = [
        {"name": "vllm-xpu-nix", "head": PROJECT_REVISION},
        {"name": "vllm", "head": VLLM_REVISION},
        {"name": "vllm-xpu-kernels", "head": KERNELS_REVISION},
    ]
    expected = {
        "vllm-xpu-nix": PROJECT_REVISION,
        "vllm": VLLM_REVISION,
        "vllm-xpu-kernels": KERNELS_REVISION,
    }
    assert factory.require_expected_repository_revisions(repositories, expected) == (
        expected
    )
    with pytest.raises(factory.FactoryError, match="revision mismatch"):
        factory.require_expected_repository_revisions(
            repositories, {**expected, "vllm": "4" * 40}
        )


def test_runtime_environment_contract_exposes_prefix_contamination(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("KVARN_RTN_QUANTILE", "0.99")
    contract = factory.runtime_environment_contract()
    assert contract["prefixed_environment_clean"] is False
    assert contract["kvarn_or_vllm_prefixed_variables"] == {
        "KVARN_RTN_QUANTILE": "0.99"
    }


def test_scope_can_never_be_mistaken_for_service_parity() -> None:
    assert "not service-performance or parity evidence" in factory.SCOPE_WARNING
    source = Path(factory.__file__).read_text(encoding="utf-8")
    assert "KVARN_NATIVE_XPU" not in source
    assert "VLLM_XPU_KERNELS_LIBRARY" in source
