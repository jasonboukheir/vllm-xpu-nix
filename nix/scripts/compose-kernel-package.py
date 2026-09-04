"""Scrub donor bytecode, restamp metadata, and regenerate RECORD."""

from __future__ import annotations

import base64
import csv
import hashlib
import re
import sys
from pathlib import Path

from packaging.version import Version


def scrub_cached_bytecode(site: Path) -> None:
    """Remove bytecode whose code objects retain the donor store path."""
    for pattern in ("*.pyc", "*.pyo"):
        for path in site.rglob(pattern):
            path.unlink()
    cache_dirs = sorted(
        site.rglob("__pycache__"), key=lambda path: len(path.parts), reverse=True
    )
    for path in cache_dirs:
        if path.is_dir() and not any(path.iterdir()):
            path.rmdir()


def main() -> None:
    site = Path(sys.argv[1])
    old_dist_info = Path(sys.argv[2])
    normalized_version = str(Version(sys.argv[3]))

    # baseGlue is copied into the composed package. Its cached bytecode embeds
    # the donor's absolute path in co_filename, which would retain the entire
    # donor output in the runtime closure. Source files remain importable.
    scrub_cached_bytecode(site)

    metadata = old_dist_info / "METADATA"
    text, count = re.subn(
        r"^Version: .*$",
        f"Version: {normalized_version}",
        metadata.read_text(),
        count=1,
        flags=re.MULTILINE,
    )
    if count != 1:
        raise SystemExit("expected exactly one Version field in METADATA")
    metadata.write_text(text)

    new_dist_info = site / f"vllm_xpu_kernels-{normalized_version}.dist-info"
    old_dist_info.rename(new_dist_info)
    record = new_dist_info / "RECORD"
    rows: list[tuple[str, str, str]] = []
    for path in sorted(path for path in site.rglob("*") if path.is_file()):
        relative = path.relative_to(site).as_posix()
        if path == record:
            rows.append((relative, "", ""))
            continue
        payload = path.read_bytes()
        digest = base64.urlsafe_b64encode(hashlib.sha256(payload).digest())
        encoded_digest = digest.rstrip(b"=").decode()
        rows.append((relative, f"sha256={encoded_digest}", str(len(payload))))
    with record.open("w", newline="") as handle:
        csv.writer(handle).writerows(rows)


if __name__ == "__main__":
    main()
