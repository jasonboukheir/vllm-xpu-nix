import json
from pathlib import Path

ROOT = Path(__file__).parents[1]


def test_release_flake_has_no_kvarn_factory_surface() -> None:
    source = (ROOT / "flake.nix").read_text()
    assert "kvarn-factory" not in source
    assert "kvarnFactory" not in source
    assert "vllm-xpu-kvarn-validation" in source
    assert "vllm-xpu-kvarn-validation-env" in source


def test_only_unstable_release_package_enables_vision() -> None:
    source = (ROOT / "flake.nix").read_text()
    stable = source.split("vllm-xpu = mkVllm {", 1)[1].split("};", 1)[0]
    unstable = source.split("vllm-xpu-unstable = mkVllm {", 1)[1].split("};", 1)[0]
    assert "withTorchvision" not in stable
    assert "withTorchvision = true;" in unstable
    assert "96,false,false,false,false,false" in source


def test_release_refs_and_lock_are_xpu_v1_6() -> None:
    source = (ROOT / "flake.nix").read_text()
    assert source.count('ref = "refs/heads/releases/xpu-v1.6";') == 2
    assert "refs/heads/releases/xpu-v1.5" not in source
    nodes = json.loads((ROOT / "flake.lock").read_text())["nodes"]
    expected = {
        "vllm-xpu-unstable-src": "f9a7a62a1ae02d2b33385663c97049172f98f4c9",
        "vllm-xpu-kernels-unstable-src": "767dc3ddf3a614f765e34b566ce788cb79bb2798",
    }
    for name, revision in expected.items():
        assert nodes[name]["locked"]["ref"] == "refs/heads/releases/xpu-v1.6"
        assert nodes[name]["locked"]["rev"] == revision
