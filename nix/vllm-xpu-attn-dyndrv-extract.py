#!/usr/bin/env python3
"""Configure-time extractor for the dyn-drv attn_kernels_xe_2 build.

Run from configureDrv.installPhase after `cmake -GNinja` has populated
$out/repo/build/. Produces three artefacts in $out/repo/:

  - compile_commands.json (rewritten in-place: cmake's $build paths -> $out/repo)
  - tu_manifest.json: every .cpp the link target consumes, each with the
        absolute src path under $out/repo and the ninja-relative .o path.
  - link_meta.json: the resolved link command for the target SO with $in
        replaced by the sentinel __INPUTS__. linkDrv substitutes that token
        with the per-TU .o store paths in the order ninja listed them.

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


NO_RDC_FLAG = "-fno-sycl-rdc"


def inject_no_sycl_rdc_compile(cc: list[dict]) -> int:
    """Append -fno-sycl-rdc to every compile entry. FA2 TUs are self-contained
    SYCL device images (no cross-TU SYCL_EXTERNAL); RDC is dead weight here
    and icpx's docs cite 10-20% compile-time/RSS savings from disabling it."""
    touched = 0
    for entry in cc:
        if "arguments" in entry:
            if NO_RDC_FLAG not in entry["arguments"]:
                entry["arguments"].append(NO_RDC_FLAG)
                touched += 1
        if "command" in entry:
            if NO_RDC_FLAG not in entry["command"].split():
                entry["command"] = entry["command"] + " " + NO_RDC_FLAG
    return touched


# SYCL-TLA TUs have wildly skewed peak RSS in icpx (~5 GiB median, ~40 GiB
# heavy tail). On a single-host consumer the heavy tail caps usable max-jobs
# even when most TUs would happily run at higher concurrency. Patterns listed
# here are matched against the TU's repo-relative source path; matches get
# `is_heavy: true` in the manifest, and the dyn-drv nix wires those into a
# serial DAG chain so they never overlap. Empty list = behaviour-preserving;
# populate after profiling per-TU max RSS on a fat-RAM builder.
HEAVY_PATTERNS: list[str] = []


def classify_heavy(src_rel: str) -> bool:
    return any(re.search(p, src_rel) for p in HEAVY_PATTERNS)


# Decode TU template parameters from the generated filename.
#
# Two naming schemes in csrc/xpu/attn/xe_2:
#   chunk_prefill_kernel_template_chunk_policy_head{N}[_b16]_<flags>.cpp
#   paged_decode_kernel_template_q{Q}_h{N}_p{P}_<flags>.cpp
#
# Anything else (e.g. fmha_xe2.cpp, paged_decode_xe2.cpp launchers) is
# unclassifiable here and always kept — pruning a launcher would leave the
# dispatcher with no entry point.
_CHUNK_PREFILL_RE = re.compile(r"head(\d+)(_b16)?_[ft]+\.cpp$")
_PAGED_DECODE_RE = re.compile(r"_h(\d+)_p\d+_[ft]+\.cpp$")


def parse_tu_params(src_rel: str) -> dict | None:
    base = os.path.basename(src_rel)
    if (m := _CHUNK_PREFILL_RE.search(base)):
        return {"head_dim": int(m.group(1)), "dtype": "bf16" if m.group(2) else "fp16"}
    if (m := _PAGED_DECODE_RE.search(base)):
        # paged_decode dtype is encoded in the boolean flag suffix; decoding
        # it reliably needs the upstream generator. Leave dtype=None so the
        # dtype filter never excludes a paged_decode TU.
        return {"head_dim": int(m.group(1)), "dtype": None}
    return None


def keep_tu(src_rel: str, kernel_set: dict) -> bool:
    params = parse_tu_params(src_rel)
    if params is None:
        return True
    head_dims = kernel_set.get("head_dims")
    if head_dims is not None and params["head_dim"] not in head_dims:
        return False
    dtypes = kernel_set.get("dtypes")
    if dtypes is not None and params["dtype"] is not None and params["dtype"] not in dtypes:
        return False
    return True


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
    touched = inject_no_sycl_rdc_compile(rewritten)
    with open(cc_path, "w") as f:
        json.dump(rewritten, f, indent=2)
    print(f"[extract] rewrote {len(rewritten)} compile_commands.json entries: "
          f"{args.src_root} -> {repo}")
    print(f"[extract] appended {NO_RDC_FLAG} to {touched} compile entries")

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
    if NO_RDC_FLAG not in resolved.split():
        resolved = resolved + " " + NO_RDC_FLAG

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
            "is_heavy": classify_heavy(src_rel),
        })
    launcher_count = sum(
        1 for m in manifest
        if m["src_rel_path"].endswith(("fmha_xe2.cpp", "paged_decode_xe2.cpp"))
    )
    heavy_count = sum(1 for m in manifest if m["is_heavy"])
    print(f"[extract] tu_manifest: {len(manifest)} TUs "
          f"({launcher_count} launchers, {heavy_count} heavy)")

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


if __name__ == "__main__":
    main()
