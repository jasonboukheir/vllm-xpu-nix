"""Archive profiling/packaging sources and detect changes during capture."""

import hashlib
import json
from pathlib import Path


def _inputs(root):
    paths = set((root / "scripts").rglob("*.py"))
    paths.update((root / "scripts").rglob("*.cpp"))
    paths.update((root / "nix").rglob("*.nix"))
    paths.update(
        path for path in (root / "flake.nix", root / "flake.lock") if path.exists()
    )
    if not paths:
        raise ValueError("no profiling source inputs found")
    return sorted(paths)


def _digest(data):
    return {"sha256": hashlib.sha256(data).hexdigest(), "size_bytes": len(data)}


def _read(root, relative):
    path = root / relative
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"source path escapes snapshot root: {relative}")
    return path.read_bytes()


def snapshot_sources(root: Path, destination: Path) -> dict:
    root = root.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    entries = {}
    for path in _inputs(root):
        relative = path.relative_to(root).as_posix()
        data = _read(root, relative)
        target = destination / "files" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(data)
        entries[relative] = _digest(data)
    manifest = {"schema_version": 1, "source_root": str(root), "files": entries}
    with (destination / "manifest.json").open("x") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    verify_sources(destination, source_root=root)
    return manifest


def verify_sources(destination: Path, *, source_root: Path | None = None) -> dict:
    """Verify archives offline; optionally require the live source tree unchanged."""
    manifest = json.loads((destination / "manifest.json").read_text())
    if manifest.get("schema_version") != 1 or not manifest.get("files"):
        raise ValueError("invalid source snapshot manifest")
    for relative, expected in manifest["files"].items():
        if Path(relative).is_absolute() or ".." in Path(relative).parts:
            raise ValueError("invalid source snapshot path")
        if _digest(_read(destination / "files", relative)) != expected:
            raise ValueError(f"archived source changed: {relative}")
        if (
            source_root is not None
            and _digest(_read(source_root, relative)) != expected
        ):
            raise ValueError(f"source changed during capture: {relative}")
    if source_root is not None:
        current = {p.relative_to(source_root).as_posix() for p in _inputs(source_root)}
        if current != set(manifest["files"]):
            raise ValueError("source input set changed during capture")
    return manifest
