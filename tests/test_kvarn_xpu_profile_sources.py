import pytest

from scripts.kvarn_xpu_profile_sources import snapshot_sources, verify_sources


@pytest.mark.parametrize("change", [None, "content", "added", "archive"])
def test_snapshot_verifies_sources_and_detects_changes(tmp_path, change):
    root, snapshot = tmp_path / "repo", tmp_path / "snapshot"
    (root / "scripts").mkdir(parents=True)
    source = root / "scripts/run.py"
    source.write_text("before")
    initial = snapshot_sources(root, snapshot)
    assert verify_sources(snapshot, source_root=root) == initial
    if change == "content":
        source.write_text("after")
    elif change == "added":
        (root / "scripts/extra.py").write_text("new source")
    elif change == "archive":
        (snapshot / "files/scripts/run.py").write_text("changed archive")
    if change:
        with pytest.raises(ValueError, match="changed"):
            verify_sources(snapshot, source_root=root)
    else:
        assert verify_sources(snapshot) == initial
    with pytest.raises(FileExistsError):
        snapshot_sources(root, snapshot)


def test_source_symlink_outside_repository_rejected(tmp_path):
    root = tmp_path / "repo"
    (root / "scripts").mkdir(parents=True)
    outside = tmp_path / "outside.py"
    outside.write_text("outside")
    (root / "scripts/run.py").symlink_to(outside)
    with pytest.raises(ValueError, match="escapes"):
        snapshot_sources(root, tmp_path / "snapshot")
