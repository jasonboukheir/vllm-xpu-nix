import base64
import csv
import hashlib
import importlib.util
import py_compile
from pathlib import Path

SCRIPT = Path(__file__).parents[1] / "nix" / "scripts" / "compose-kernel-package.py"


def load_module():
    spec = importlib.util.spec_from_file_location("compose_kernel_package", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_restamps_metadata_and_regenerates_record(tmp_path, monkeypatch):
    site = tmp_path / "site-packages"
    package = site / "vllm_xpu_kernels"
    package.mkdir(parents=True)
    (package / "_C.abi3.so").write_bytes(b"base")
    (package / "_vllm_fa2_C.abi3.so").write_bytes(b"fa2")
    source = package / "donor.py"
    source.write_text("VALUE = 1\n")
    cache = package / "__pycache__"
    cache.mkdir()
    bytecode = cache / "donor.cpython-312.pyc"
    py_compile.compile(
        str(source),
        cfile=str(bytecode),
        dfile="/nix/store/fake-base-glue/vllm_xpu_kernels/donor.py",
        doraise=True,
    )

    old_dist_info = site / "vllm_xpu_kernels-0.1.14.1+src.old.dist-info"
    old_dist_info.mkdir()
    (old_dist_info / "METADATA").write_text(
        "Metadata-Version: 2.4\nName: vllm-xpu-kernels\nVersion: 0.1.14.1+src.old\n"
    )
    (old_dist_info / "RECORD").write_text("stale\n")
    (old_dist_info / "WHEEL").write_text("Wheel-Version: 1.0\n")

    module = load_module()
    monkeypatch.setattr(
        module.sys,
        "argv",
        [
            str(SCRIPT),
            str(site),
            str(old_dist_info),
            "0.1.14.1+unstable.2026.09.04.gabc",
        ],
    )
    module.main()

    new_dist_info = site / "vllm_xpu_kernels-0.1.14.1+unstable.2026.9.4.gabc.dist-info"
    assert not old_dist_info.exists()
    assert not bytecode.exists()
    assert not cache.exists()
    assert (
        "Version: 0.1.14.1+unstable.2026.9.4.gabc\n"
        in (new_dist_info / "METADATA").read_text()
    )

    with (new_dist_info / "RECORD").open(newline="") as handle:
        rows = {row[0]: row[1:] for row in csv.reader(handle)}
    record_name = "vllm_xpu_kernels-0.1.14.1+unstable.2026.9.4.gabc.dist-info/RECORD"
    assert rows[record_name] == ["", ""]
    fa2_name = "vllm_xpu_kernels/_vllm_fa2_C.abi3.so"
    expected_digest = base64.urlsafe_b64encode(hashlib.sha256(b"fa2").digest())
    assert rows[fa2_name] == [
        f"sha256={expected_digest.rstrip(b'=').decode()}",
        "3",
    ]
    assert not any(name.endswith((".pyc", ".pyo")) for name in rows)
