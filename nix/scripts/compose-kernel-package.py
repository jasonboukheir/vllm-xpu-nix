"""Restamp composed vllm-xpu-kernels metadata and regenerate RECORD."""

from __future__ import annotations

import base64
import csv
import hashlib
import re
import sys
from pathlib import Path

from packaging.version import Version


def main() -> None:
    site = Path(sys.argv[1])
    old_dist_info = Path(sys.argv[2])
    normalized_version = str(Version(sys.argv[3]))

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
