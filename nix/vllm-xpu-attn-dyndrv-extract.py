#!/usr/bin/env python3
"""Configure-time extractor for the dyn-drv attn_kernels_xe_2 build.

Run from configureDrv.installPhase after `cmake -GNinja` has populated
$out/repo/build/. Produces four artefacts in $out/repo/:

  - compile_commands.json (rewritten in-place: cmake's $build paths -> $out/repo)
  - tu_manifest.json: every .cpp the link target consumes, each with the
        absolute src path under $out/repo and the ninja-relative .o path.
        Excludes the cmake-synthesised cmake_pch.hxx.cxx source — its compile
        command lives in pch_meta.json instead, and a dedicated pchDrv
        realises it once per (configure) and feeds the .pch to every TU.
  - link_meta.json: the resolved link command for the target SO with $in
        replaced by the sentinel __INPUTS__. linkDrv substitutes that token
        with the per-TU .o store paths in the order ninja listed them.
  - pch_meta.json: just the src + .pch paths (relative to $out/repo).
        Plain JSON with no /nix/store/... mentions so Nix's
        readFile+fromJSON at eval time stays within store-path context.
  - pch_command.txt: the cmake-emitted PCH compile command with -o
        rewritten to the placeholder __PCH_OUT__. pchDrv substitutes
        $out/cmake_pch.hxx.pch at build time (same shell-substitute
        pattern linkDrv uses for link_command.txt + __INPUTS__).
        mkTU's extract-cmd.py rewrites the embedded .pch path in each
        per-TU command to point at ${pchDrv}/cmake_pch.hxx.pch.

build.ninja parsing is intentionally minimal: it handles cmake's emitted
syntax (one top-level build.ninja that `include`s CMakeFiles/rules.ninja)
and nothing else. If cmake's ninja generator changes shape, this file is
where to look first.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import sys


def parse_block_vars(text: str, start: int) -> tuple[dict[str, str], int]:
    """Read indented `key = value` lines starting at `start`. Returns vars + end offset."""
    vars_ = {}
    i = start
    while i < len(text):
        # End of block: a non-indented line.
        line_end = text.find("\n", i)
        if line_end == -1:
            line_end = len(text)
        line = text[i:line_end]
        if not line.strip():
            i = line_end + 1
            continue
        if not (line.startswith("  ") or line.startswith("\t")):
            break
        k, sep, v = line.strip().partition("=")
        if sep:
            vars_[k.strip()] = v.strip()
        i = line_end + 1
    return vars_, i


def find_rule(rules_text: str, rule_name: str) -> dict[str, str]:
    """Find `rule <name>` in rules.ninja and return its variable bindings."""
    pat = re.compile(rf"^rule\s+{re.escape(rule_name)}\s*$", re.MULTILINE)
    m = pat.search(rules_text)
    if not m:
        sys.exit(f"FATAL: rule {rule_name} not found in rules.ninja")
    vars_, _ = parse_block_vars(rules_text, m.end() + 1)
    return vars_


def find_link_edge(ninja_text: str, soname: str) -> tuple[str, str, list[str], dict[str, str]]:
    """Find the `build <target>: <rule> <inputs...>` edge for the link target."""
    edge_re = re.compile(
        rf"^build\s+(\S*?{re.escape(soname)})\s*:\s+(\S+)\s+(.*)$",
        re.MULTILINE,
    )
    m = edge_re.search(ninja_text)
    if not m:
        sys.exit(f"FATAL: build.ninja has no edge for {soname}")
    target_path = m.group(1)
    rule_name = m.group(2)
    rest = m.group(3)
    # Inputs may include explicit, implicit (after |), order-only (after ||).
    parts = rest.split()
    inputs = []
    for p in parts:
        if p in ("|", "||"):
            break
        inputs.append(p)
    edge_vars, _ = parse_block_vars(ninja_text, m.end() + 1)
    return target_path, rule_name, inputs, edge_vars


def expand_vars(cmd: str, env: dict[str, str], max_passes: int = 20) -> str:
    """Substitute $var and ${var} iteratively. ninja vars are flat — a few passes suffice."""
    for _ in range(max_passes):
        new = cmd
        # Longest names first so $TARGET_FILE doesn't match $TARGET_.
        for k in sorted(env, key=len, reverse=True):
            v = env[k]
            new = new.replace("${" + k + "}", v).replace("$" + k, v)
        if new == cmd:
            return new
        cmd = new
    return cmd


def strip_link_artefacts(cmd: str) -> str:
    """Drop flags that depend on cmake's build dir layout (cwd of $TMPDIR won't have these subdirs).

    `-Wl,--dependency-file=<path>` writes to a path relative to the original
    build dir; without this strip, the linker errors trying to mkdir-implicitly.
    """
    return re.sub(r"-Wl,--dependency-file=\S+\s*", "", cmd)


# Heavy-TU classification (used downstream to chain icpx RSS-spiking TUs
# through a serial DAG edge) used to live here as a filename heuristic
# keyed on head_dim. It moved to the Nix layer in #20: the new profile
# stage runs `icpx -E` per TU and the dyndrv module reads measured
# preproc byte counts at eval time, so the manifest no longer carries
# an `is_heavy` field.


# Decode TU template parameters from the generated filename.
#
# Two naming schemes in csrc/xpu/attn/xe_2 (post the 0007 dtype-split patch):
#   chunk_prefill_kernel_template_chunk_policy_head{N}[_b16]_<bool-flags>_<dt>.cpp
#   paged_decode_kernel_template_q{Q}_h{N}_p{P}_<bool-flags>_<dt>.cpp
#
# <dt> is a 2-char dtype token from chunk_prefill_configure.cmake /
# paged_decode_configure.cmake: hh, h4, h5, bb, b4, b5 where the first
# char is Q dtype (half / bfloat) and the second is KV dtype (half /
# bfloat / fp8_e4m3 / fp8_e5m2).
#
# Anything else (e.g. fmha_xe2.cpp, paged_decode_xe2.cpp launchers) is
# unclassifiable here and always kept — pruning a launcher would leave the
# dispatcher with no entry point.
_DTYPE_TOKEN_RE = r"[hb][hb45]"
_CHUNK_PREFILL_RE = re.compile(
    rf"head(\d+)(_b16)?_[ft]+_({_DTYPE_TOKEN_RE})\.cpp$"
)
_PAGED_DECODE_RE = re.compile(
    rf"_h(\d+)_p\d+_[ft]+_({_DTYPE_TOKEN_RE})\.cpp$"
)

_QTOKEN_TO_DTYPE = {"h": "fp16", "b": "bf16"}


def parse_tu_params(src_rel: str) -> dict | None:
    base = os.path.basename(src_rel)
    if (m := _CHUNK_PREFILL_RE.search(base)):
        return {
            "family": "chunk_prefill",
            "head_dim": int(m.group(1)),
            "dtype": _QTOKEN_TO_DTYPE[m.group(3)[0]],
        }
    if (m := _PAGED_DECODE_RE.search(base)):
        return {
            "family": "paged_decode",
            "head_dim": int(m.group(1)),
            "dtype": _QTOKEN_TO_DTYPE[m.group(2)[0]],
        }
    return None


def keep_tu(src_rel: str, kernel_set: dict) -> bool:
    params = parse_tu_params(src_rel)
    if params is None:
        return True
    # Both prune dimensions (head_dim, dtype) require a corresponding
    # launcher rewrite to be fully safe. The launchers (fmha_xe2.cpp,
    # paged_decode_xe2.cpp) hardcode dispatch over the full (head_dim,
    # Q dtype, KV dtype) cross-product; pruning a dimension here drops
    # the matching extern template instantiation but leaves the launcher
    # referring to it, which fails at link time (or at dlopen for kernels
    # marked optional).
    #
    # Currently we only apply the head_dim prune to chunk_prefill (the
    # paged_decode launcher is fully hardcoded); see issue #47 for the
    # chunk_prefill case (also still technically unsafe — see
    # chunk-prefill-prune-bug.md). The dtype prune is left in place for
    # exploratory builds but is similarly unsafe after the dtype-split
    # patch (0007-fa2-dtype-split.patch) widened the launcher dispatch
    # to all 6 (Q, KV) combos. A future patch should make the launcher
    # dispatch conditional on cmake-time toggles matching this prune.
    head_dims = kernel_set.get("head_dims")
    if (
        head_dims is not None
        and params["family"] == "chunk_prefill"
        and params["head_dim"] not in head_dims
    ):
        return False
    dtypes = kernel_set.get("dtypes")
    if dtypes is not None and params["dtype"] is not None and params["dtype"] not in dtypes:
        return False
    return True


def _strip_dep_flags(args: list[str]) -> list[str]:
    """Drop -MD/-MMD/-MT/-MF dep-tracking flags from a compile command."""
    cleaned, i = [], 0
    while i < len(args):
        a = args[i]
        if a in ("-MD", "-MMD"):
            i += 1
        elif a in ("-MT", "-MF"):
            i += 2
        else:
            cleaned.append(a)
            i += 1
    return cleaned


def _find_pch_pch_path(args: list[str]) -> str | None:
    """Return the absolute path embedded after a `-Xclang -include-pch` pair, if any."""
    for i in range(len(args) - 3):
        if (args[i] == "-Xclang" and args[i + 1] == "-include-pch"
                and args[i + 2] == "-Xclang"):
            return args[i + 3]
    return None


def safe_drv_name(rel_name: str) -> str:
    name = rel_name
    for suffix in (".cpp", ".cc", ".cxx"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return re.sub(r"[^A-Za-z0-9_-]", "_", name)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="$out/repo path")
    ap.add_argument("--src-root", required=True, help="cmake's original src root (will be rewritten away)")
    ap.add_argument("--target", required=True, help="cmake target name (e.g. attn_kernels_xe_2)")
    ap.add_argument("--soname", required=True, help="output soname (e.g. libattn_kernels_xe_2.so)")
    ap.add_argument("--kernel-set", default=None,
                    help='JSON: {"head_dims": [128], "dtypes": ["bf16"]}. '
                         'Omit or pass "{}" to keep all TUs.')
    args = ap.parse_args()

    kernel_set: dict = {}
    if args.kernel_set:
        kernel_set = json.loads(args.kernel_set)
        if "head_dims" in kernel_set and kernel_set["head_dims"] is not None:
            kernel_set["head_dims"] = set(kernel_set["head_dims"])
        if "dtypes" in kernel_set and kernel_set["dtypes"] is not None:
            kernel_set["dtypes"] = set(kernel_set["dtypes"])

    repo = args.repo
    build = os.path.join(repo, "build")
    cc_path = os.path.join(build, "compile_commands.json")

    # 1. Rewrite compile_commands.json paths from build sandbox -> $out/repo.
    with open(cc_path) as f:
        cc = json.load(f)
    rewritten = []
    for entry in cc:
        e = dict(entry)
        for k in ("directory", "file"):
            if k in e:
                e[k] = e[k].replace(args.src_root, repo)
        if "command" in e:
            e["command"] = e["command"].replace(args.src_root, repo)
        if "arguments" in e:
            e["arguments"] = [a.replace(args.src_root, repo) for a in e["arguments"]]
        rewritten.append(e)
    with open(cc_path, "w") as f:
        json.dump(rewritten, f, indent=2)
    print(f"[extract] rewrote {len(rewritten)} compile_commands.json entries: "
          f"{args.src_root} -> {repo}")

    # 1a. Extract the cmake-synthesised PCH compile entry. cmake emits one
    # source file named `cmake_pch.hxx.cxx` per target with
    # `target_precompile_headers(...)`. Its command produces the .pch the
    # rest of the TUs consume via `-Xclang -include-pch -Xclang <path>`.
    pch_entry = next(
        (e for e in rewritten
         if os.path.basename(e.get("file", "")) == "cmake_pch.hxx.cxx"),
        None,
    )
    pch_meta = None
    pch_command = None
    if pch_entry is not None:
        if "arguments" in pch_entry:
            pch_args = list(pch_entry["arguments"])
        else:
            pch_args = shlex.split(pch_entry["command"])
        pch_args = _strip_dep_flags(pch_args)
        # Capture the output .pch path so mkTU can find-and-replace it.
        pch_out_path = None
        for i, a in enumerate(pch_args):
            if a == "-o" and i + 1 < len(pch_args):
                pch_out_path = pch_args[i + 1]
                break
        if pch_out_path is None:
            sys.exit("FATAL: PCH compile entry has no -o argument")
        # cmake-Ninja emits the PCH -o as a path relative to the build
        # directory (compile_commands.json's `directory` field). Every
        # other path in the entry — `file`, `-c` source, and sibling
        # TUs' `-include-pch` refs — is absolute; only this one isn't.
        # Resolve against `directory` so the absolute form lines up with
        # the embedded PCH refs the downstream substring rewrite expects.
        if not os.path.isabs(pch_out_path):
            pch_out_path = os.path.normpath(
                os.path.join(pch_entry["directory"], pch_out_path))
        if not pch_out_path.startswith(repo + "/"):
            sys.exit(
                f"FATAL: PCH -o path {pch_out_path} is outside $repo {repo}")
        pch_out_rel = pch_out_path[len(repo) + 1:]
        pch_src_path = pch_entry["file"]
        if not pch_src_path.startswith(repo + "/"):
            sys.exit(
                f"FATAL: PCH src path {pch_src_path} is outside $repo {repo}")
        # The PCH compile command pulls in cmake_pch.hxx (the thin
        # wrapper cmake generates next to cmake_pch.hxx.cxx) via
        # `-Xclang -include -Xclang <path>`. Its single #include bakes
        # an absolute path to attn_pch.hpp using CMAKE_CURRENT_SOURCE_DIR
        # — which at configure time is the sandbox dir (/build/source/...).
        # Rewrite in-place so pchDrv's icpx loads the umbrella header
        # from $repo, not the long-vanished sandbox path. compile_commands
        # paths got the same treatment in step 1; this file is the one
        # other generated artifact in the PCH plumbing that contains
        # baked source paths.
        pch_wrapper = pch_src_path[:-len(".cxx")]
        if os.path.exists(pch_wrapper):
            with open(pch_wrapper) as f:
                wrap_text = f.read()
            wrap_new = wrap_text.replace(args.src_root, repo)
            if wrap_new != wrap_text:
                with open(pch_wrapper, "w") as f:
                    f.write(wrap_new)
                print(f"[extract] rewrote {os.path.basename(pch_wrapper)}: "
                      f"{args.src_root} -> {repo}")
        # Rewrite -o to a placeholder. pchDrv substitutes the real $out
        # path at build time. Keeping the command out of pch_meta.json
        # (it lives in pch_command.txt instead) means readFile+fromJSON
        # at eval time doesn't see icpx / torch / sycl /nix/store paths
        # — same trick link_meta.json + link_command.txt already use.
        for k, a in enumerate(pch_args):
            if a == "-o" and k + 1 < len(pch_args):
                pch_args[k + 1] = "__PCH_OUT__"
                break
        pch_meta = {
            "src_rel_path": pch_src_path[len(repo) + 1:],
            "pch_out_rel_path": pch_out_rel,
        }
        pch_command = shlex.join(pch_args)
        print(f"[extract] pch_meta: src={pch_meta['src_rel_path']} "
              f"out={pch_meta['pch_out_rel_path']}")

    # 2. Parse build.ninja + rules.ninja for the link edge of <target>.
    with open(os.path.join(build, "build.ninja")) as f:
        ninja_text = f.read()
    with open(os.path.join(build, "CMakeFiles", "rules.ninja")) as f:
        rules_text = f.read()

    target_path, rule_name, obj_inputs, edge_vars = find_link_edge(ninja_text, args.soname)
    rule_vars = find_rule(rules_text, rule_name)
    print(f"[extract] link edge: {target_path} (rule {rule_name}, {len(obj_inputs)} obj inputs)")

    # Substitute $in with a sentinel; resolve everything else.
    env = dict(rule_vars)
    env.update(edge_vars)
    env["in"] = "__INPUTS__"
    env["out"] = target_path
    if "TARGET_FILE" not in env:
        env["TARGET_FILE"] = target_path
    if "ARCH_FLAGS" not in env:
        env["ARCH_FLAGS"] = ""

    raw_command = rule_vars.get("command")
    if not raw_command:
        sys.exit(f"FATAL: rule {rule_name} has no command")
    resolved = expand_vars(raw_command, env)
    resolved = strip_link_artefacts(resolved)
    # The expanded command embeds the configure-time sandbox path (e.g.
    # /build/source/csrc/sycl_first.h via -include). That dir vanishes
    # once configureDrv finishes; rewrite to $out/repo where we keep a
    # stable copy of the source tree.
    resolved = resolved.replace(args.src_root, repo)

    # Sanity: no ninja vars should remain. `$$` (ninja-escaped literal $) is fine.
    leftover = re.findall(r"(?<!\$)\$[A-Za-z_][A-Za-z0-9_]*", resolved.replace("$$", ""))
    if leftover:
        sys.exit(f"FATAL: unresolved ninja vars in link command: {set(leftover)}")
    if "\n" in resolved:
        sys.exit("FATAL: resolved link command contains a newline; cannot be used in single-line heredoc")

    # 3. Map .o input paths -> .cpp src paths via build.ninja's compile edges.
    # Each compile edge: `build <obj_path>: <COMPILE_RULE> <src_path>`.
    obj_to_src: dict[str, str] = {}
    for obj in obj_inputs:
        # Match build edges that produce this obj. Allow optional `|` for implicit outputs.
        edge_re = re.compile(
            rf"^build\s+{re.escape(obj)}\s*(?:\|\s*\S+\s*)?:\s+\S+\s+(\S+)",
            re.MULTILINE,
        )
        m = edge_re.search(ninja_text)
        if not m:
            sys.exit(f"FATAL: no compile edge in build.ninja for object {obj}")
        src = m.group(1)
        # Source paths in build.ninja may be absolute (cmake source files
        # under $out/repo via cp -a) or relative to build dir (generated
        # files under build/csrc/...). Normalise to absolute under $repo.
        if not os.path.isabs(src):
            src = os.path.normpath(os.path.join(build, src))
        else:
            src = src.replace(args.src_root, repo)
        obj_to_src[obj] = src

    # Apply the kernel-set filter (if any) before the manifest + link_meta
    # are built, so both the per-TU drv enumeration and the link inputs
    # see the same pruned set. Unclassifiable TUs (launchers etc) are
    # always kept; mis-pruning a launcher would leave the dispatcher with
    # no entry point.
    if kernel_set:
        kept_obj_inputs = []
        dropped_by_dim: dict[str, int] = {}
        for obj_rel in obj_inputs:
            src = obj_to_src[obj_rel]
            src_rel = src[len(repo) + 1:] if src.startswith(repo + "/") else src
            if keep_tu(src_rel, kernel_set):
                kept_obj_inputs.append(obj_rel)
            else:
                params = parse_tu_params(src_rel) or {}
                key = ",".join(f"{k}={v}" for k, v in sorted(params.items()) if v is not None)
                dropped_by_dim[key] = dropped_by_dim.get(key, 0) + 1
        if not kept_obj_inputs:
            sys.exit(f"FATAL: kernel-set filter {kernel_set} dropped every TU; "
                     "either widen the filter or omit it")
        print(f"[extract] kernel-set filter kept {len(kept_obj_inputs)}/{len(obj_inputs)} TUs")
        for k, n in sorted(dropped_by_dim.items()):
            print(f"[extract]   dropped {n} ({k or 'unclassified'})")
        obj_inputs = kept_obj_inputs

    # 4. Build the per-TU manifest. Paths are stored RELATIVE to $repo,
    # not as absolute /nix/store paths — Nix's builtins.fromJSON returns
    # context-less strings, and downstream code that puts those strings
    # into derivation attributes would trip "is not allowed to refer to
    # a store path" if the JSON content embedded /nix/store/... literals.
    # The Nix code prepends ${configureDrv}/repo/ via interpolation, which
    # carries proper context.
    manifest = []
    seen_safe_names: dict[str, int] = {}
    for obj_rel in obj_inputs:
        src = obj_to_src[obj_rel]
        if not src.startswith(repo + "/"):
            sys.exit(f"FATAL: source {src} is outside $repo {repo}")
        src_rel = src[len(repo) + 1:]
        rel_basename = os.path.basename(src)
        base_safe = safe_drv_name(rel_basename)
        # Disambiguate if two TUs share a basename.
        n = seen_safe_names.get(base_safe, 0)
        seen_safe_names[base_safe] = n + 1
        safe = base_safe if n == 0 else f"{base_safe}-{n}"
        manifest.append({
            "safe_name": safe,
            "src_rel_path": src_rel,
            "obj_rel_path": obj_rel,
        })
    launcher_count = sum(
        1 for m in manifest
        if m["src_rel_path"].endswith(("fmha_xe2.cpp", "paged_decode_xe2.cpp"))
    )
    print(f"[extract] tu_manifest: {len(manifest)} TUs ({launcher_count} launchers)")

    with open(os.path.join(repo, "tu_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    link_meta = {
        "target_path": target_path,
        "rule_name": rule_name,
        "inputs": obj_inputs,
    }
    with open(os.path.join(repo, "link_meta.json"), "w") as f:
        json.dump(link_meta, f, indent=2)
    # The resolved link command goes to a separate text file. Nix's
    # builtins.fromJSON returns strings without a "store path context",
    # which trips a safety check when the result is fed to lib.replaceStrings
    # and the string text contains literal /nix/store/... paths. The link
    # command DOES contain store paths (icpx + torch + sycl); keeping it
    # in plain text and substituting __INPUTS__ at build time (via shell)
    # sidesteps that whole class of error — the buildPhase's implicit
    # `${configureDrv}` interpolation already carries the right context.
    with open(os.path.join(repo, "link_command.txt"), "w") as f:
        f.write(resolved + "\n")

    if pch_meta is not None:
        with open(os.path.join(repo, "pch_meta.json"), "w") as f:
            json.dump(pch_meta, f, indent=2)
        with open(os.path.join(repo, "pch_command.txt"), "w") as f:
            f.write(pch_command + "\n")


if __name__ == "__main__":
    main()
