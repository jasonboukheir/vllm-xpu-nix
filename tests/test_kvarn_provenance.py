import argparse
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
SCRIPT = ROOT / "scripts" / "kvarn_provenance.py"
FIXTURES = ROOT / "fixtures" / "kvarn-long-generation.json"
SPEC = importlib.util.spec_from_file_location("kvarn_provenance", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def git(repo, *args):
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def make_repo(path, name):
    path.mkdir()
    git(path, "init", "--quiet")
    git(path, "config", "user.name", "Kvarn Test")
    git(path, "config", "user.email", "kvarn@example.invalid")
    (path / "tracked.txt").write_text(name, encoding="utf-8")
    git(path, "add", "tracked.txt")
    git(path, "-c", "commit.gpgsign=false", "commit", "--quiet", "-m", "initial")
    return path


def test_long_generation_fixtures_cover_required_categories():
    prompts = MODULE.load_prompt_fixtures(FIXTURES)

    assert {prompt["category"] for prompt in prompts} == {
        "adversarial",
        "code",
        "dialogue",
        "math",
        "reasoning",
    }
    assert all(prompt["max_tokens"] >= 2048 for prompt in prompts)
    assert all(len(prompt["prompt_sha256"]) == 64 for prompt in prompts)


def test_durable_output_rejects_tmp_without_override():
    with pytest.raises(ValueError, match="outside /tmp"):
        MODULE.ensure_durable_output(Path("/tmp/kvarn-results"), allow_tmp=False)

    assert MODULE.ensure_durable_output(
        Path("/tmp/kvarn-results"), allow_tmp=True
    ) == Path("/tmp/kvarn-results")


def test_manifest_records_repositories_command_environment_and_artifacts(tmp_path):
    packaging = make_repo(tmp_path / "packaging", "packaging")
    vllm = make_repo(tmp_path / "vllm", "vllm")
    kernels = make_repo(tmp_path / "kernels", "kernels")
    (kernels / "dirty.txt").write_text("dirty", encoding="utf-8")

    output = tmp_path / "durable-results"
    output.mkdir()
    artifact = output / "result.json"
    artifact.write_text('{"passed":true}\n', encoding="utf-8")
    argv_file = tmp_path / "argv.json"
    argv = ["vllm", "serve", "owner/model", "--enforce-eager"]
    argv_file.write_text(json.dumps(argv), encoding="utf-8")
    environment_file = tmp_path / "environment.json"
    environment_file.write_text(
        json.dumps(
            {
                "KVARN_NATIVE_XPU": "0",
                "SAFE_EXTRA": "yes",
                "HF_TOKEN": "must-not-be-recorded",
            }
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        output_dir=output,
        manifest_name="provenance.json",
        model="owner/model",
        model_revision="1" * 40,
        fixtures=FIXTURES,
        argv_file=argv_file,
        environment_file=environment_file,
        env=["SAFE_EXTRA"],
        artifact=[],
        vllm_xpu_nix=packaging,
        vllm=vllm,
        kernels=kernels,
        allow_tmp=True,
    )

    manifest_path = MODULE.write_manifest(args)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    repositories = {item["name"]: item for item in manifest["repositories"]}
    assert set(repositories) == {"vllm-xpu-nix", "vllm", "vllm-xpu-kernels"}
    assert not repositories["vllm"]["dirty"]
    assert repositories["vllm-xpu-kernels"]["dirty"]
    assert repositories["vllm-xpu-kernels"]["status_porcelain"] == ["?? dirty.txt"]
    assert all(len(item["head"]) == 40 for item in repositories.values())
    assert manifest["model"] == {"id": "owner/model", "revision": "1" * 40}
    assert manifest["command"]["argv"] == argv
    assert manifest["command"]["shell_rendered"].endswith("--enforce-eager")
    assert manifest["environment"]["values"]["KVARN_NATIVE_XPU"] == "0"
    assert manifest["environment"]["values"]["SAFE_EXTRA"] == "yes"
    assert "HF_TOKEN" not in manifest["environment"]["values"]
    assert manifest["artifacts"] == [
        {
            "modified_at": manifest["artifacts"][0]["modified_at"],
            "path": "result.json",
            "sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            "size_bytes": artifact.stat().st_size,
        }
    ]
    assert manifest["collection"]["finished_at"] >= manifest["collection"]["started_at"]
