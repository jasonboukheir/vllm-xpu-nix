"""Inspect embedded Xe ELF metadata offline; never load a DSO or access XPU.

Run with the frozen runtime's Python, or any Python >= 3.10. Only the standard
library is needed. Output is JSON on stdout. Repeat --library LABEL=PATH to
inspect other artifacts. Missing spill metadata is not an occupancy measurement.
"""

import argparse
import hashlib
import json
import re
import struct
from pathlib import Path

ELF_HEADER = struct.Struct("<16sHHIQQQIHHHHHH")
SECTION_HEADER = struct.Struct("<IIQQQQIIQQ")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def embedded_elfs(data):
    """Find valid ELF64 little-endian section tables inside the host DSO."""
    for match in re.finditer(b"\x7fELF", data):
        offset = match.start()
        try:
            header = ELF_HEADER.unpack_from(data, offset)
            if header[0][4:6] != b"\x02\x01" or header[11] != 64:
                continue
            if not 0 < header[12] <= 5000 or header[13] >= header[12]:
                continue
            sections = [
                SECTION_HEADER.unpack_from(data, offset + header[6] + i * 64)
                for i in range(header[12])
            ]

            def contents(section, elf_offset=offset):
                start, size = elf_offset + section[4], section[5]
                if start + size > len(data):
                    raise ValueError("section exceeds container")
                return data[start : start + size]

            strings = contents(sections[header[13]])
            named = {strings[s[0] :].split(b"\0", 1)[0].decode(): s for s in sections}
            if ".ze_info" in named:
                yield offset, header[2], named, contents
        except (IndexError, UnicodeDecodeError, ValueError, struct.error):
            continue


def scalar(value):
    value = value.strip()
    if value in ("true", "false"):
        return value == "true"
    try:
        return int(value)
    except ValueError:
        return value


def selected_policy(name):
    return "prefill" if "XeFMHAFwdKernel" in name else None


def inspect_library(spec):
    label, requested = spec.split("=", 1)
    path = Path(requested).resolve(strict=True)
    data = path.read_bytes()
    kernels = []
    for offset, machine, sections, contents in embedded_elfs(data):
        metadata = contents(sections[".ze_info"]).decode()
        # Split top-level kernel records, avoiding similarly named records in
        # kernels_misc_info by requiring an execution_env in each selected entry.
        for entry in re.split(r"\n  - name:\s*", metadata)[1:]:
            name = entry.splitlines()[0]
            policy = selected_policy(name)
            if policy is None or "\n    execution_env:\n" not in entry:
                continue
            environment = entry.split("\n    execution_env:\n", 1)[1]
            environment = re.split(r"\n    \S", environment, maxsplit=1)[0]
            fields = {
                key: scalar(value)
                for key, value in re.findall(
                    r"^      (\w+):\s*(.+)$", environment, re.MULTILINE
                )
            }
            buffers = []
            if "\n    per_thread_memory_buffers:\n" in entry:
                region = entry.split("\n    per_thread_memory_buffers:\n", 1)[1]
                region = re.split(r"\n    \S", region, maxsplit=1)[0]
                for record in re.split(r"(?:^|\n)      - ", region):
                    if record.strip():
                        buffers.append(
                            {
                                key: scalar(value)
                                for key, value in re.findall(
                                    r"^\s*(\w+):\s*(.+)$", record, re.MULTILINE
                                )
                            }
                        )
            code = contents(sections[".text." + name])
            kernels.append(
                {
                    "policy": policy,
                    "kernel_name": name,
                    "embedded_elf_offset": offset,
                    "elf_machine": machine,
                    "code_bytes": len(code),
                    "code_sha256": sha256(code),
                    "execution_environment": fields,
                    "spill_size_reported": fields.get("spill_size"),
                    "per_thread_memory_buffers": buffers,
                }
            )
    if not kernels:
        raise ValueError(f"no matching Xe prefill kernels in {path}")
    return {
        "label": label,
        "requested_path": requested,
        "resolved_path": str(path),
        "sha256": sha256(data),
        "kernels": kernels,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", action="append", required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            {
                "extractor_sha256": sha256(Path(__file__).read_bytes()),
                "libraries": [inspect_library(s) for s in args.library],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
