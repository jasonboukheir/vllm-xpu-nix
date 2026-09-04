import importlib.util
import struct
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "kvarn_resource_audit", ROOT / "scripts" / "kvarn_resource_audit.py"
)
assert SPEC is not None and SPEC.loader is not None
audit = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = audit
SPEC.loader.exec_module(audit)


def make_elf(elf_type, machine, sections, *, force_extent_mod_16=None):
    """Create the strict ELF64 subset consumed by the audit parser."""
    sections = [*sections, (".shstrtab", b"")]
    names = bytearray(b"\0")
    name_offsets = {}
    for name, _data in sections:
        name_offsets[name] = len(names)
        names.extend(name.encode() + b"\0")
    sections[-1] = (".shstrtab", bytes(names))

    payload = bytearray(b"\0" * audit.ELF64_HEADER_SIZE)
    section_rows = [(0, 0, 0, 0)]
    for name, data in sections:
        payload.extend(b"\0" * (audit.align_up(len(payload), 8) - len(payload)))
        offset = len(payload)
        payload.extend(data)
        section_rows.append((name_offsets[name], 1, offset, len(data)))
    payload.extend(b"\0" * (audit.align_up(len(payload), 8) - len(payload)))
    if force_extent_mod_16 is not None:
        if force_extent_mod_16 not in {0, 8}:
            raise ValueError("synthetic ELF extent can only be shaped to 0 or 8 mod 16")
        if len(payload) % 16 != force_extent_mod_16:
            payload.extend(b"\0" * 8)
    section_offset = len(payload)
    for name_offset, section_type, offset, size in section_rows:
        payload.extend(
            struct.pack(
                "<IIQQQQIIQQ",
                name_offset,
                section_type,
                0,
                0,
                offset,
                size,
                0,
                0,
                1,
                0,
            )
        )
    ident = b"\x7fELF" + bytes([2, 1, 1]) + b"\0" * 9
    struct.pack_into(
        "<16sHHIQQQIHHHHHH",
        payload,
        0,
        ident,
        elf_type,
        machine,
        1,
        0,
        0,
        section_offset,
        0,
        audit.ELF64_HEADER_SIZE,
        0,
        0,
        audit.ELF64_SECTION_SIZE,
        len(section_rows),
        len(section_rows) - 1,
    )
    return bytes(payload)


def intelgt_note(note_type, description):
    owner = b"IntelGT\0"
    value = bytearray(struct.pack("<III", len(owner), len(description), note_type))
    value.extend(owner)
    value.extend(b"\0" * (audit.align_up(len(value), 4) - len(value)))
    value.extend(description)
    value.extend(b"\0" * (audit.align_up(len(value), 4) - len(value)))
    return bytes(value)


COMPAT_NOTES = intelgt_note(1, struct.pack("<I", 1274)) + intelgt_note(
    2, struct.pack("<I", 3081)
)


def ze_info(kernels):
    lines = ["---", "version: '1.73'", "kernels:"]
    for kernel in kernels:
        lines.extend(
            [
                f"  - name: {kernel['name']}",
                "    execution_env:",
                f"      grf_count: {kernel.get('grf', 256)}",
            ]
        )
        if kernel.get("spill", 0):
            lines.append(f"      spill_size: {kernel['spill']}")
        if kernel.get("slm", 0):
            lines.append(f"      slm_size: {kernel['slm']}")
        if kernel.get("scratch") is not None:
            lines.extend(
                [
                    "    per_thread_memory_buffers:",
                    "      - type: scratch",
                    f"        size: {kernel['scratch']}",
                    "        usage: private_space",
                ]
            )
    lines.extend(["kernels_misc_info:", "..."])
    return ("\n".join(lines) + "\n").encode()


def make_image(kernels, extra_padding_shape=0, *, force_extent_mod_16=None):
    sections = [
        (".ze_info", ze_info(kernels)),
        (".note.intelgt.compat", COMPAT_NOTES),
        *[
            (f".text.{kernel['name']}", bytes([index + 1]) * kernel["text"])
            for index, kernel in enumerate(kernels)
        ],
    ]
    if extra_padding_shape:
        sections.append((".synthetic_shape", b"x" * extra_padding_shape))
    return make_elf(
        audit.ET_REL,
        audit.EM_INTELGT,
        sections,
        force_extent_mod_16=force_extent_mod_16,
    )


def mainloop(signature):
    bools = ", ".join("true" if value else "false" for value in signature)
    return f"KVarNDecodeFwdMainloop<A<B, C>, D, E, 8, Q, K, V, {bools}>"


def demangled_main(signature):
    return (
        "void device_kernel<cutlass::fmha::kernel::XeFMHAFwdSplitKVKernel<"
        f"X, {mainloop(signature)}, Z>>()"
    )


def demangled_reducer(signature):
    return (
        "void device_kernel<cutlass::fmha::kernel::ReduceSplitK<"
        f"X, Wrapper<{mainloop(signature)}>>>()"
    )


def round6_fixture(*, omit=None, resources=None):
    omit = omit or set()
    resources = resources or {}
    kernel_groups = [[], []]
    demangled = {}
    for variant_id, metadata in audit.VARIANT_SIGNATURES.items():
        for role in ("main", "generic_split_reducer"):
            key = (variant_id, role)
            if key in omit:
                continue
            name = f"m_id{variant_id}_{role}"
            defaults = {
                "name": name,
                "text": 100 + (variant_id - 18) * 10 if role == "main" else 80,
                "slm": 0,
            }
            defaults.update(resources.get(key, {}))
            kernel_groups[0].append(defaults)
            demangled[name] = (
                demangled_main(metadata["signature"])
                if role == "main"
                else demangled_reducer(metadata["signature"])
            )

    shared_names = {
        "generic": "KVarNReduceSplitOutputHadamardKernel<64>",
        **{
            f"split_{split}": (
                f"KVarNReduceSplitOutputHadamardSpecializedKernel<{split}, 64>"
            )
            for split in (2, 4, 8, 16, 32)
        },
    }
    for index, (role, template) in enumerate(shared_names.items()):
        name = f"m_shared_{role}"
        kernel_groups[1].append({"name": name, "text": 48 + index})
        demangled[name] = f"void device_kernel<{template}>()"

    # Shape the first ELF to require real inter-image alignment padding; the
    # second remains an exact, unpadded terminal image.
    images = [
        make_image(
            group,
            extra_padding_shape=1 if index == 0 else 0,
            force_extent_mod_16=8 if index == 0 else None,
        )
        for index, group in enumerate(kernel_groups)
    ]
    offload = b"".join(
        image
        + (
            b"\0" * (audit.align_up(len(image), 16) - len(image))
            if index + 1 < len(images)
            else b""
        )
        for index, image in enumerate(images)
    )
    library = make_elf(
        audit.ET_DYN, audit.EM_X86_64, [(audit.OFFLOAD_SECTION, offload)]
    )
    return library, demangled, images


def test_round6_resource_audit_passes_and_enumerates_images():
    library, names, _images = round6_fixture()

    report = audit.audit_bytes(library, names.__getitem__)

    assert report["status"] == "passed"
    assert report["offload"]["image_count"] == 2
    assert report["images"][-1]["padding_bytes"] == 0
    assert report["offload"]["compat_notes"]["2"]["value_u32"] == 3081
    assert report["variants"]["18"]["entries"]["main"]["text_bytes"] == 100
    assert report["variants"]["20"]["entries"]["main"]["text_bytes"] == 120
    assert report["variants"]["21"]["entries"]["main"]["slm_size"] == 0
    assert report["comparisons"]["id20_to_id18_main_text_ratio"] == 1.2
    assert set(report["shared_service_reducers"]) == set(
        audit.EXPECTED_SHARED_REDUCERS
    )
    assert report["violations"] == []


def test_missing_variant_role_is_a_structural_error():
    library, names, _images = round6_fixture(omit={(20, "generic_split_reducer")})

    with pytest.raises(
        audit.AuditError,
        match="ID20 generic_split_reducer: expected exactly one kernel, found 0",
    ):
        audit.audit_bytes(library, names.__getitem__)


def test_resource_failures_remain_machine_readable():
    resources = {
        (20, "main"): {"spill": 32, "scratch": 64},
        (21, "main"): {"grf": 128},
    }
    library, names, _images = round6_fixture(resources=resources)

    report = audit.audit_bytes(library, names.__getitem__)

    assert report["status"] == "failed"
    assert "ID20 main spills 32 bytes" in report["violations"]
    assert "ID20 main uses 64 scratch bytes" in report["violations"]
    assert "ID21 main grf_count is not 256" in report["violations"]


def test_empty_reference_text_fails_without_dividing_by_zero():
    library, names, _images = round6_fixture(
        resources={(18, "main"): {"text": 0}}
    )

    report = audit.audit_bytes(library, names.__getitem__)

    assert report["status"] == "failed"
    assert "ID18 main has no .text bytes" in report["violations"]
    assert report["comparisons"]["id20_to_id18_main_text_ratio"] is None


def test_nonzero_inter_image_padding_fails_closed():
    library, names, images = round6_fixture()
    first = images[0]
    padding = audit.align_up(len(first), 16) - len(first)
    assert padding == 8
    outer = audit.parse_elf(library, "synthetic outer")
    section = audit.sole_section(outer, audit.OFFLOAD_SECTION, "synthetic outer")
    mutated = bytearray(library)
    mutated[section.offset + len(first)] = 0xA5

    with pytest.raises(audit.AuditError, match="nonzero alignment padding"):
        audit.audit_bytes(bytes(mutated), names.__getitem__)


def test_note_padding_and_scratch_records_are_parsed_strictly():
    block = """  - name: example
    per_thread_memory_buffers:
      - type: global
        size: 4096
      - type: scratch
        size: 2048
"""
    assert audit._scratch_sizes(block, "example") == (2048,)

    malformed = bytearray(intelgt_note(7, b"abc"))
    malformed[-1] = 1
    with pytest.raises(audit.AuditError, match="nonzero note-description padding"):
        audit.parse_intelgt_notes(bytes(malformed), "notes")


def test_template_parser_preserves_nested_arguments_and_detects_conflicts():
    signature = audit.VARIANT_SIGNATURES[18]["signature"]
    value = demangled_main(signature)
    args = audit.split_template_arguments(value, "KVarNDecodeFwdMainloop")
    assert args[0] == "A<B, C>"
    assert audit.mainloop_signature(value) == signature

    conflicting = value + " " + mainloop(audit.VARIANT_SIGNATURES[20]["signature"])
    with pytest.raises(audit.AuditError, match="conflicting Kvarn"):
        audit.mainloop_signature(conflicting)


def test_cli_accepts_both_explicit_artifact_option_spellings():
    expected = Path("/nix/store/example/lib/libattn_kernels_xe_2.so")
    first = audit.parse_args(
        ["--attention-library", str(expected), "--llvm-cxxfilt", "/bin/true"]
    )
    second = audit.parse_args(
        ["--shared-object", str(expected), "--llvm-cxxfilt", "/bin/true"]
    )
    assert first.attention_library == expected
    assert second.attention_library == expected
