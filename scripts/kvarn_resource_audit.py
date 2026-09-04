#!/usr/bin/env python3
"""Audit Kvarn IntelGT AOT code and resources without loading the library.

The input is opened read-only.  The report is written only to stdout, and no
device, driver, or vLLM process is touched.  A structural ambiguity exits 2;
an unambiguous resource-gate failure emits the complete JSON report and exits
3 so the measurements are still available for diagnosis.

The ``slm_size`` field is copied from ``.ze_info`` and covers compiler-declared
static SLM only.  SYCL ``work_group_scratch_size`` supplied at launch (including
ID21's 1 KiB paired-nibble LUT) is not represented by that field.

Admission requires unique signature-to-text bindings for all expected kernels,
nonempty text, ``grf_count == 256``, and zero spill and scratch bytes.  SLM and
relative text sizes are evidence, not admission gates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import struct
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ATTENTION_LIBRARY = "libattn_kernels_xe_2.so"
OFFLOAD_SECTION = "OFFLOAD_DEVICE_CODE"
EXPECTED_ZE_INFO_VERSION = "1.73"
ELF64_HEADER_SIZE = 64
ELF64_SECTION_SIZE = 64
ELFCLASS64 = 2
ELFDATA2LSB = 1
ET_REL = 1
ET_DYN = 3
EM_X86_64 = 62
EM_INTELGT = 0xCD
SHT_NOBITS = 8
INTELGT_IMAGE_ALIGNMENT = 16

# KVarNDecodeFwdMainloop template booleans, in source order:
# DpasPacked, VectorPackedLoads, QKInt8U4, ExactLiveRows, PagePair,
# NextPagePrefetch, SimdPackedUnpack, CurrentHalfVPrefetch,
# ReusePageRecordCursor, ReusePageMetadataCursor, PairedNibbleHalf2.
VARIANT_SIGNATURES = {
    18: {
        "name": "q6_prefetch_record_cursor",
        "signature": (
            True,
            False,
            False,
            False,
            False,
            True,
            False,
            True,
            True,
            False,
            False,
        ),
    },
    20: {
        "name": "q6_page_metadata_cursor",
        "signature": (
            True,
            False,
            False,
            False,
            False,
            True,
            False,
            True,
            True,
            True,
            False,
        ),
    },
    21: {
        "name": "q6_paired_nibble_half2",
        "signature": (
            True,
            True,
            False,
            False,
            False,
            True,
            False,
            True,
            True,
            False,
            True,
        ),
    },
}
SIGNATURE_TO_ID = {
    value["signature"]: variant_id
    for variant_id, value in VARIANT_SIGNATURES.items()
}
EXPECTED_SHARED_REDUCERS = (
    "generic",
    "split_2",
    "split_4",
    "split_8",
    "split_16",
    "split_32",
)


class AuditError(RuntimeError):
    """The artifact could not be audited without making an assumption."""


@dataclass(frozen=True)
class Section:
    name: str
    section_type: int
    offset: int
    size: int


@dataclass(frozen=True)
class Elf:
    data: bytes
    elf_type: int
    machine: int
    sections: tuple[Section, ...]
    extent: int

    def named_sections(self, name: str) -> list[Section]:
        return [section for section in self.sections if section.name == name]

    def section_data(self, section: Section) -> bytes:
        if section.section_type == SHT_NOBITS:
            raise AuditError(f"section {section.name!r} has no file bytes")
        return self.data[section.offset : section.offset + section.size]


@dataclass(frozen=True)
class KernelRecord:
    image_index: int
    mangled_name: str
    grf_count: int | None
    spill_size: int
    slm_size: int
    scratch_sizes: tuple[int, ...]
    text_bytes: int
    text_sha256: str


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def align_up(value: int, alignment: int) -> int:
    return (value + alignment - 1) // alignment * alignment


def _unpack(fmt: str, data: bytes, offset: int, label: str) -> tuple[Any, ...]:
    size = struct.calcsize(fmt)
    if offset < 0 or offset + size > len(data):
        raise AuditError(f"{label}: truncated structure at offset {offset}")
    return struct.unpack_from(fmt, data, offset)


def parse_elf(data: bytes, label: str) -> Elf:
    if len(data) < ELF64_HEADER_SIZE:
        raise AuditError(f"{label}: shorter than an ELF64 header")
    if data[:4] != b"\x7fELF":
        raise AuditError(f"{label}: missing ELF magic")
    if data[4] != ELFCLASS64 or data[5] != ELFDATA2LSB or data[6] != 1:
        raise AuditError(f"{label}: requires little-endian ELF64 version 1")

    elf_type, machine = _unpack("<HH", data, 16, label)
    (elf_version,) = _unpack("<I", data, 20, label)
    (section_offset,) = _unpack("<Q", data, 40, label)
    (header_size,) = _unpack("<H", data, 52, label)
    section_entry_size, section_count, string_table_index = _unpack(
        "<HHH", data, 58, label
    )
    if elf_version != 1 or header_size != ELF64_HEADER_SIZE:
        raise AuditError(f"{label}: unsupported ELF header version or size")
    if section_entry_size != ELF64_SECTION_SIZE:
        raise AuditError(
            f"{label}: expected {ELF64_SECTION_SIZE}-byte section entries, "
            f"got {section_entry_size}"
        )
    if section_count == 0 or string_table_index >= section_count:
        raise AuditError(f"{label}: extended or invalid section numbering")
    section_table_end = section_offset + section_entry_size * section_count
    if section_offset < header_size or section_table_end > len(data):
        raise AuditError(f"{label}: section table is outside the file")

    raw_sections: list[tuple[int, int, int, int]] = []
    extent = section_table_end
    for index in range(section_count):
        values = _unpack(
            "<IIQQQQIIQQ",
            data,
            section_offset + index * section_entry_size,
            f"{label} section {index}",
        )
        name_offset, section_type = values[:2]
        file_offset, size = values[4:6]
        if section_type != SHT_NOBITS:
            if file_offset + size > len(data):
                raise AuditError(f"{label}: section {index} exceeds file bytes")
            extent = max(extent, file_offset + size)
        raw_sections.append((name_offset, section_type, file_offset, size))

    string_header = raw_sections[string_table_index]
    if string_header[1] == SHT_NOBITS:
        raise AuditError(f"{label}: section-name table cannot be NOBITS")
    strings = data[string_header[2] : string_header[2] + string_header[3]]
    sections: list[Section] = []
    seen_names: set[str] = set()
    for index, (name_offset, section_type, file_offset, size) in enumerate(
        raw_sections
    ):
        if name_offset >= len(strings):
            raise AuditError(f"{label}: section {index} has an invalid name offset")
        name_end = strings.find(b"\0", name_offset)
        if name_end < 0:
            raise AuditError(f"{label}: section {index} name is unterminated")
        try:
            name = strings[name_offset:name_end].decode("utf-8")
        except UnicodeDecodeError as error:
            raise AuditError(f"{label}: section {index} name is not UTF-8") from error
        if name and name in seen_names:
            raise AuditError(f"{label}: duplicate section name {name!r}")
        if name:
            seen_names.add(name)
        sections.append(Section(name, section_type, file_offset, size))
    return Elf(data, elf_type, machine, tuple(sections), extent)


def sole_section(elf: Elf, name: str, label: str) -> Section:
    matches = elf.named_sections(name)
    if len(matches) != 1:
        raise AuditError(f"{label}: expected exactly one {name}, found {len(matches)}")
    return matches[0]


def parse_intelgt_notes(data: bytes, label: str) -> dict[str, dict[str, Any]]:
    notes: dict[str, dict[str, Any]] = {}
    cursor = 0
    while cursor < len(data):
        if len(data) - cursor < 12:
            raise AuditError(f"{label}: truncated IntelGT note header")
        name_size, description_size, note_type = _unpack(
            "<III", data, cursor, label
        )
        cursor += 12
        name_end = cursor + name_size
        if name_end > len(data):
            raise AuditError(f"{label}: truncated IntelGT note owner")
        owner = data[cursor:name_end].rstrip(b"\0")
        padded_name_end = align_up(name_end, 4)
        if any(data[name_end:padded_name_end]):
            raise AuditError(f"{label}: nonzero note-owner padding")
        cursor = padded_name_end
        description_end = cursor + description_size
        if description_end > len(data):
            raise AuditError(f"{label}: truncated IntelGT note description")
        description = data[cursor:description_end]
        padded_description_end = align_up(description_end, 4)
        if any(data[description_end:padded_description_end]):
            raise AuditError(f"{label}: nonzero note-description padding")
        cursor = padded_description_end
        if cursor > len(data):
            raise AuditError(f"{label}: truncated IntelGT note padding")
        if owner != b"IntelGT":
            raise AuditError(f"{label}: unexpected note owner {owner!r}")
        key = str(note_type)
        if key in notes:
            raise AuditError(f"{label}: duplicate IntelGT note type {note_type}")
        item: dict[str, Any] = {"description_hex": description.hex()}
        if len(description) == 4:
            item["value_u32"] = int.from_bytes(description, "little")
        else:
            try:
                item["text"] = description.rstrip(b"\0").decode("utf-8")
            except UnicodeDecodeError:
                pass
        notes[key] = item
    if "2" not in notes:
        raise AuditError(f"{label}: IntelGT architecture note is absent")
    return notes


def parse_ze_info_version(text: str, label: str) -> str:
    matches = re.findall(r"^version:\s+['\"]?([^'\"\s]+)['\"]?\s*$", text, re.M)
    if len(matches) != 1:
        raise AuditError(f"{label}: expected exactly one .ze_info version")
    if matches[0] != EXPECTED_ZE_INFO_VERSION:
        raise AuditError(
            f"{label}: expected .ze_info version {EXPECTED_ZE_INFO_VERSION}, "
            f"got {matches[0]}"
        )
    return matches[0]


def _numeric_execution_field(block: str, field: str, label: str) -> int | None:
    matches = re.findall(rf"^      {re.escape(field)}:\s+(\d+)\s*$", block, re.M)
    if len(matches) > 1:
        raise AuditError(f"{label}: duplicate execution_env field {field}")
    return int(matches[0]) if matches else None


def _scratch_sizes(block: str, label: str) -> tuple[int, ...]:
    lines = block.splitlines()
    try:
        start = lines.index("    per_thread_memory_buffers:") + 1
    except ValueError:
        return ()
    records: list[dict[str, str]] = []
    current: dict[str, str] | None = None
    for line in lines[start:]:
        stripped = line.lstrip(" ")
        indent = len(line) - len(stripped)
        if stripped and indent <= 4:
            break
        if indent == 6 and stripped.startswith("- type:"):
            if current is not None:
                records.append(current)
            current = {"type": stripped.split(":", 1)[1].strip()}
        elif current is not None and indent == 8 and ":" in stripped:
            key, value = stripped.split(":", 1)
            if key in current:
                raise AuditError(f"{label}: duplicate memory-buffer field {key}")
            current[key] = value.strip()
    if current is not None:
        records.append(current)
    scratch: list[int] = []
    for record in records:
        if record.get("type") != "scratch":
            continue
        value = record.get("size")
        if value is None or not value.isdecimal():
            raise AuditError(f"{label}: scratch buffer has no decimal size")
        scratch.append(int(value))
    return tuple(scratch)


def parse_kernel_records(
    elf: Elf, ze_info: str, image_index: int
) -> list[KernelRecord]:
    match = re.search(r"^kernels:\s*$", ze_info, re.M)
    if match is None:
        raise AuditError(f"IntelGT image {image_index}: .ze_info lacks kernels")
    kernel_text = ze_info[match.end() :]
    kernel_text = re.split(
        r"^kernels_misc_info:\s*$", kernel_text, maxsplit=1, flags=re.M
    )[0]
    headers = list(re.finditer(r"^  - name:\s+(\S+)\s*$", kernel_text, re.M))
    if not headers:
        raise AuditError(f"IntelGT image {image_index}: .ze_info has no kernels")
    records: list[KernelRecord] = []
    for position, header in enumerate(headers):
        end = (
            headers[position + 1].start()
            if position + 1 < len(headers)
            else len(kernel_text)
        )
        block = kernel_text[header.start() : end]
        name = header.group(1)
        label = f"IntelGT image {image_index} kernel {name}"
        text_section = sole_section(elf, f".text.{name}", label)
        text = elf.section_data(text_section)
        scratch_sizes = _scratch_sizes(block, label)
        if len(scratch_sizes) > 1:
            raise AuditError(f"{label}: more than one scratch buffer")
        records.append(
            KernelRecord(
                image_index=image_index,
                mangled_name=name,
                grf_count=_numeric_execution_field(block, "grf_count", label),
                spill_size=_numeric_execution_field(block, "spill_size", label) or 0,
                slm_size=_numeric_execution_field(block, "slm_size", label) or 0,
                scratch_sizes=scratch_sizes,
                text_bytes=len(text),
                text_sha256=sha256_bytes(text),
            )
        )
    return records


def split_template_arguments(value: str, template_name: str) -> list[str]:
    marker = f"{template_name}<"
    start = value.find(marker)
    if start < 0:
        raise AuditError(f"demangled name lacks {marker}")
    cursor = start + len(marker)
    argument_start = cursor
    depth = 1
    arguments: list[str] = []
    while cursor < len(value):
        char = value[cursor]
        if char == "<":
            depth += 1
        elif char == ">":
            depth -= 1
            if depth == 0:
                arguments.append(value[argument_start:cursor].strip())
                return arguments
        elif char == "," and depth == 1:
            arguments.append(value[argument_start:cursor].strip())
            argument_start = cursor + 1
        cursor += 1
    raise AuditError(f"unterminated {marker} in demangled name")


def mainloop_signature(demangled: str) -> tuple[bool, ...] | None:
    marker = "KVarNDecodeFwdMainloop<"
    starts = [match.start() for match in re.finditer(re.escape(marker), demangled)]
    if not starts:
        return None
    signatures: set[tuple[bool, ...]] = set()
    for start in starts:
        arguments = split_template_arguments(
            demangled[start:], "KVarNDecodeFwdMainloop"
        )
        if len(arguments) < 11 or any(
            argument not in {"true", "false"} for argument in arguments[-11:]
        ):
            raise AuditError(
                "KVarNDecodeFwdMainloop does not end in the expected 11 booleans"
            )
        signatures.add(tuple(argument == "true" for argument in arguments[-11:]))
    if len(signatures) != 1:
        raise AuditError(
            "one kernel name contains conflicting Kvarn mainloop signatures"
        )
    return signatures.pop()


def classify_variant_role(demangled: str) -> str:
    if "device_kernel<cutlass::fmha::kernel::ReduceSplitK<" in demangled:
        return "generic_split_reducer"
    if "device_kernel<cutlass::fmha::kernel::XeFMHAFwdSplitKVKernel<" in demangled:
        return "main"
    raise AuditError("target Kvarn signature belongs to an unknown kernel role")


def classify_shared_reducer(demangled: str) -> str | None:
    specialized = "KVarNReduceSplitOutputHadamardSpecializedKernel"
    generic = "KVarNReduceSplitOutputHadamardKernel"
    if f"{specialized}<" in demangled:
        arguments = split_template_arguments(demangled, specialized)
        if len(arguments) == 2 and arguments[1] != "64":
            return None
        if len(arguments) != 2 or not arguments[0].isdecimal():
            raise AuditError("unexpected specialized output-Hadamard reducer type")
        split = int(arguments[0])
        if split not in {2, 4, 8, 16, 32}:
            raise AuditError(f"unexpected specialized output-Hadamard split {split}")
        return f"split_{split}"
    if f"{generic}<" in demangled:
        arguments = split_template_arguments(demangled, generic)
        if len(arguments) == 1 and arguments[0] != "64":
            return None
        if arguments != ["64"]:
            raise AuditError("unexpected generic output-Hadamard reducer type")
        return "generic"
    return None


def resource_document(record: KernelRecord, demangled: str) -> dict[str, Any]:
    return {
        "image_index": record.image_index,
        "kernel_name": record.mangled_name,
        "kernel_name_sha256": sha256_bytes(record.mangled_name.encode()),
        "demangled_name": demangled,
        "grf_count": record.grf_count,
        "spill_size": record.spill_size,
        "scratch_bytes": record.scratch_sizes[0] if record.scratch_sizes else 0,
        "slm_size": record.slm_size,
        "text_bytes": record.text_bytes,
        "text_sha256": record.text_sha256,
    }


def demangle_records(
    records: Sequence[KernelRecord], demangle: Callable[[str], str]
) -> dict[str, str]:
    names = list(dict.fromkeys(record.mangled_name for record in records))
    batch_method = getattr(demangle, "many", None)
    if callable(batch_method):
        results = batch_method(names)
        missing = [name for name in names if name not in results]
        if missing:
            raise AuditError(
                f"batch demangler omitted {len(missing)} device kernel name(s)"
            )
        return {name: results[name] for name in names}
    return {name: demangle(name) for name in names}


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def audit_bytes(data: bytes, demangle: Callable[[str], str]) -> dict[str, Any]:
    outer = parse_elf(data, "attention library")
    if outer.elf_type != ET_DYN or outer.machine != EM_X86_64:
        raise AuditError("attention library must be an x86_64 ET_DYN ELF")
    offload_section = sole_section(outer, OFFLOAD_SECTION, "attention library")
    offload = outer.section_data(offload_section)
    if not offload:
        raise AuditError("OFFLOAD_DEVICE_CODE is empty")

    cursor = 0
    image_documents: list[dict[str, Any]] = []
    all_records: list[KernelRecord] = []
    compat_reference: dict[str, dict[str, Any]] | None = None
    while cursor < len(offload):
        index = len(image_documents)
        image = parse_elf(offload[cursor:], f"IntelGT image {index}")
        if image.elf_type != ET_REL or image.machine != EM_INTELGT:
            raise AuditError(f"IntelGT image {index}: expected IntelGT ET_REL ELF")
        # Intermediate images begin at a 16-byte boundary.  The section may
        # end exactly at the final image's ELF extent, with no terminal pad.
        # Treat only that exact condition as an exception to alignment.
        terminal_image = cursor + image.extent == len(offload)
        consumed_extent = (
            image.extent
            if terminal_image
            else align_up(image.extent, INTELGT_IMAGE_ALIGNMENT)
        )
        if cursor + consumed_extent > len(offload):
            raise AuditError(f"IntelGT image {index}: alignment exceeds offload bytes")
        padding = offload[cursor + image.extent : cursor + consumed_extent]
        if any(padding):
            raise AuditError(f"IntelGT image {index}: nonzero alignment padding")

        ze_section = sole_section(image, ".ze_info", f"IntelGT image {index}")
        try:
            ze_info = image.section_data(ze_section).rstrip(b"\0").decode("utf-8")
        except UnicodeDecodeError as error:
            raise AuditError(f"IntelGT image {index}: .ze_info is not UTF-8") from error
        ze_version = parse_ze_info_version(ze_info, f"IntelGT image {index}")
        compat_section = sole_section(
            image, ".note.intelgt.compat", f"IntelGT image {index}"
        )
        compat = parse_intelgt_notes(
            image.section_data(compat_section), f"IntelGT image {index} compat"
        )
        if compat_reference is None:
            compat_reference = compat
        elif compat != compat_reference:
            raise AuditError("IntelGT images have inconsistent compatibility notes")
        records = parse_kernel_records(image, ze_info, index)
        all_records.extend(records)
        text_sections = [
            section for section in image.sections if section.name.startswith(".text.")
        ]
        image_documents.append(
            {
                "index": index,
                "offload_offset": cursor,
                "elf_bytes": image.extent,
                "padding_bytes": consumed_extent - image.extent,
                "section_count": len(image.sections),
                "text_section_count": len(text_sections),
                "text_bytes": sum(section.size for section in text_sections),
                "ze_info_bytes": ze_section.size,
                "ze_info_sha256": sha256_bytes(image.section_data(ze_section)),
                "ze_info_version": ze_version,
                "compat_notes": compat,
                "kernel_count": len(records),
            }
        )
        cursor += consumed_extent
    if cursor != len(offload) or not image_documents:
        raise AuditError("OFFLOAD_DEVICE_CODE was not fully partitioned into images")

    seen_kernel_names: set[str] = set()
    variants: dict[int, dict[str, list[tuple[KernelRecord, str]]]] = {
        variant_id: {"main": [], "generic_split_reducer": []}
        for variant_id in VARIANT_SIGNATURES
    }
    shared: dict[str, list[tuple[KernelRecord, str]]] = {
        name: [] for name in EXPECTED_SHARED_REDUCERS
    }
    demangled_names = demangle_records(all_records, demangle)
    for record in all_records:
        # The compiler emits one Intel symbol-table pseudo-kernel in several
        # images; real entry-point names must otherwise remain globally unique.
        if record.mangled_name != "Intel_Symbol_Table_Void_Program":
            if record.mangled_name in seen_kernel_names:
                raise AuditError(f"duplicate device kernel {record.mangled_name}")
            seen_kernel_names.add(record.mangled_name)
        demangled = demangled_names[record.mangled_name]
        signature = mainloop_signature(demangled)
        if signature in SIGNATURE_TO_ID:
            variant_id = SIGNATURE_TO_ID[signature]
            role = classify_variant_role(demangled)
            variants[variant_id][role].append((record, demangled))
        shared_role = classify_shared_reducer(demangled)
        if shared_role is not None:
            shared[shared_role].append((record, demangled))

    variant_documents: dict[str, Any] = {}
    target_names: set[str] = set()
    for variant_id, metadata in VARIANT_SIGNATURES.items():
        entries: dict[str, Any] = {}
        for role, matches in variants[variant_id].items():
            if len(matches) != 1:
                raise AuditError(
                    f"ID{variant_id} {role}: expected exactly one kernel, "
                    f"found {len(matches)}"
                )
            record, demangled = matches[0]
            if record.mangled_name in target_names:
                raise AuditError("two Round6 IDs resolved to the same kernel name")
            target_names.add(record.mangled_name)
            if record.grf_count is None:
                raise AuditError(f"ID{variant_id} {role}: grf_count is absent")
            entries[role] = resource_document(record, demangled)
        variant_documents[str(variant_id)] = {
            "name": metadata["name"],
            "mainloop_signature": list(metadata["signature"]),
            "entries": entries,
        }

    shared_documents: dict[str, Any] = {}
    for role, matches in shared.items():
        if len(matches) != 1:
            raise AuditError(
                f"shared reducer {role}: expected exactly one kernel, "
                f"found {len(matches)}"
            )
        record, demangled = matches[0]
        if record.grf_count is None:
            raise AuditError(f"shared reducer {role}: grf_count is absent")
        shared_documents[role] = resource_document(record, demangled)

    violations: list[str] = []
    gated_entries = [
        (f"ID{variant_id} {role}", document)
        for variant_id, variant in variant_documents.items()
        for role, document in variant["entries"].items()
    ] + [
        (f"shared reducer {role}", document)
        for role, document in shared_documents.items()
    ]
    for label, document in gated_entries:
        if document["grf_count"] != 256:
            violations.append(f"{label} grf_count is not 256")
        if document["text_bytes"] <= 0:
            violations.append(f"{label} has no .text bytes")
        if document["spill_size"]:
            violations.append(f"{label} spills {document['spill_size']} bytes")
        if document["scratch_bytes"]:
            violations.append(f"{label} uses {document['scratch_bytes']} scratch bytes")

    id18_main = variant_documents["18"]["entries"]["main"]
    id20_main = variant_documents["20"]["entries"]["main"]
    id21_main = variant_documents["21"]["entries"]["main"]

    comparisons = {
        "id20_to_id18_main_text_ratio": _ratio(
            id20_main["text_bytes"], id18_main["text_bytes"]
        ),
        "id21_to_id18_main_text_ratio": _ratio(
            id21_main["text_bytes"], id18_main["text_bytes"]
        ),
        "id20_minus_id18_static_slm_bytes": (
            id20_main["slm_size"] - id18_main["slm_size"]
        ),
        "id21_minus_id18_static_slm_bytes": (
            id21_main["slm_size"] - id18_main["slm_size"]
        ),
    }
    return {
        "schema_version": 1,
        "status": "passed" if not violations else "failed",
        "admission_gates": [
            "unique expected signature-to-text bindings",
            "nonempty text",
            "grf_count == 256",
            "spill_size == 0",
            "scratch_bytes == 0",
        ],
        "limitations": [
            ".ze_info slm_size excludes runtime SYCL work_group_scratch_size; "
            "ID21's 1 KiB paired-nibble LUT must be verified from source/launch "
            "configuration rather than inferred from this artifact field"
        ],
        "offload": {
            "section_offset": offload_section.offset,
            "size_bytes": len(offload),
            "sha256": sha256_bytes(offload),
            "image_alignment": INTELGT_IMAGE_ALIGNMENT,
            "image_count": len(image_documents),
            "compat_notes": compat_reference,
        },
        "images": image_documents,
        "variants": variant_documents,
        "shared_service_reducers": shared_documents,
        "comparisons": comparisons,
        "violations": violations,
    }


class ExternalDemangler:
    def __init__(self, executable: Path):
        self.executable = executable.expanduser().resolve(strict=True)
        if not self.executable.is_file() or not os.access(self.executable, os.X_OK):
            raise AuditError(f"llvm-cxxfilt is not executable: {self.executable}")
        self.cache: dict[str, str] = {}

    def __call__(self, mangled: str) -> str:
        if mangled in self.cache:
            return self.cache[mangled]
        try:
            result = subprocess.run(
                [str(self.executable)],
                input=mangled + "\n",
                check=False,
                capture_output=True,
                text=True,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise AuditError(f"llvm-cxxfilt failed for {mangled}: {error}") from error
        output = result.stdout.strip()
        if result.returncode or not output or "\n" in output:
            raise AuditError(
                f"llvm-cxxfilt failed for {mangled}: return code {result.returncode}"
            )
        if "KVarN" in mangled and output == mangled:
            raise AuditError(f"llvm-cxxfilt could not demangle Kvarn kernel {mangled}")
        self.cache[mangled] = output
        return output

    def many(self, mangled_names: Sequence[str]) -> dict[str, str]:
        """Demangle in bounded batches; one process per AOT kernel is too costly."""
        result = dict(self.cache)
        pending = list(
            dict.fromkeys(name for name in mangled_names if name not in result)
        )
        for start in range(0, len(pending), 128):
            batch = pending[start : start + 128]
            try:
                completed = subprocess.run(
                    [str(self.executable)],
                    input="\n".join(batch) + "\n",
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
            except (OSError, subprocess.TimeoutExpired) as error:
                raise AuditError(f"llvm-cxxfilt batch failed: {error}") from error
            outputs = completed.stdout.splitlines()
            if completed.returncode or len(outputs) != len(batch):
                raise AuditError(
                    "llvm-cxxfilt batch returned "
                    f"{len(outputs)} lines for {len(batch)} names "
                    f"(return code {completed.returncode})"
                )
            for mangled, output in zip(batch, outputs, strict=True):
                output = output.strip()
                if not output or ("KVarN" in mangled and output == mangled):
                    raise AuditError(
                        f"llvm-cxxfilt could not demangle kernel {mangled}"
                    )
                self.cache[mangled] = output
                result[mangled] = output
        return result


def audit_file(library: Path, llvm_cxxfilt: Path) -> dict[str, Any]:
    requested = library.expanduser()
    try:
        resolved = requested.resolve(strict=True)
    except OSError as error:
        raise AuditError(
            f"cannot resolve attention library {requested}: {error}"
        ) from error
    if resolved.name != ATTENTION_LIBRARY:
        raise AuditError(f"library basename must be {ATTENTION_LIBRARY}")
    if not resolved.is_relative_to(Path("/nix/store")):
        raise AuditError(f"attention library is not in the Nix store: {resolved}")
    before = resolved.stat()
    if not stat.S_ISREG(before.st_mode):
        raise AuditError(f"attention library is not a regular file: {resolved}")
    try:
        data = resolved.read_bytes()
    except OSError as error:
        raise AuditError(
            f"cannot read attention library {resolved}: {error}"
        ) from error
    after = resolved.stat()
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after or len(data) != before.st_size:
        raise AuditError("attention library changed while it was being read")
    report = audit_bytes(data, ExternalDemangler(llvm_cxxfilt))
    report["artifact"] = {
        "requested_path": str(requested),
        "resolved_path": str(resolved),
        "size_bytes": len(data),
        "sha256": sha256_bytes(data),
    }
    return report


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--attention-library",
        "--shared-object",
        dest="attention_library",
        required=True,
        type=Path,
        help=(
            "resolved /nix/store path to libattn_kernels_xe_2.so; accepting the "
            "artifact directly avoids relying on floating-CA derivation lookup"
        ),
    )
    parser.add_argument("--llvm-cxxfilt", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = audit_file(args.attention_library, args.llvm_cxxfilt)
    except AuditError as error:
        print(f"Kvarn resource audit ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    if report["status"] != "passed":
        print(
            f"Kvarn resource audit FAILED: {len(report['violations'])} violation(s)",
            file=sys.stderr,
        )
        return 3
    print("Kvarn resource audit PASS", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
