#!/usr/bin/env python3
"""Configure-time extractor for the dyn-drv attn_kernels_xe_2 build.

Run from configureDrv.installPhase after `cmake -GNinja` has populated
$out/repo/build/. Produces these artefacts in $out/repo/:

  - compile_commands.json (rewritten in-place: cmake's $build paths -> $out/repo)
  - tu_manifest.json: every .cpp the link target consumes. Each entry
        carries the .cpp src path (relative to $out/repo), the
        ninja-relative .o path, a list of transitive header paths the
        TU pulls from $out/repo (extracted from icpx -M depfiles), and
        a path to a per-TU compile-command text file. Excludes the
        cmake-synthesised cmake_pch.hxx.cxx source.
  - per-tu-cmds/<safe_name>.txt: the cmake-emitted compile command
        for each TU, with -MD/-MT/-MF stripped, -o replaced by
        __OUT_OBJ__, every $out/repo prefix replaced by __SRC_SUBSET__,
        and (if PCH is active) the -include-pch path replaced by
        __PCH_PATH__. mkTU builds a per-TU lib.fileset.toSource over
        the source + headers + this cmd file, then substitutes the
        placeholders in shell. Living inside $repo (and therefore
        inside each TU's srcSubset) is what lets mkTU drop its
        ${configureDrv} reference entirely.
  - link_meta.json / link_command.txt: target link replay (unchanged).
  - pch_meta.json / pch_command.txt: PCH compile artifacts (unchanged).

The depfile pass uses icpx -M (preprocess-only + emit make-style deps)
on every kept TU in parallel. -M is much cheaper than full compile —
no codegen, no SYCL device IR — typically a few hundred ms per TU even
on the heavy chunk_prefill / paged_decode templates.

build.ninja parsing is intentionally minimal: it handles cmake's emitted
syntax (one top-level build.ninja that `include`s CMakeFiles/rules.ninja)
and nothing else. If cmake's ninja generator changes shape, this file is
where to look first.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import shlex
import subprocess
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


def _gen_depfile(args_tuple: tuple) -> tuple:
    """Worker for ProcessPoolExecutor. Runs icpx -M for one TU.

    Returns (src, depfile_path, error_or_None).
    """
    entry, depfile, repo = args_tuple
    if "arguments" in entry:
        cmd = list(entry["arguments"])
    else:
        cmd = shlex.split(entry["command"])
    cleaned = _strip_dep_flags(cmd)
    # Drop -o <obj> (we don't write an .o) and -c (icpx errors with
    # `-c [-Werror,-Wunused-command-line-argument]` under -M because
    # -M stops after the preprocessor — the -c becomes a no-op).
    # The source file follows -c as the next positional arg; without
    # the -c marker, icpx treats it as input the same way.
    #
    # Also drop -include-pch and its path argument. At depfile-pass
    # time we run inside configureDrv.installPhase — ninja has not
    # been invoked, so the cmake_pch.hxx.pch the TU command points at
    # does not exist on disk, and icpx -M would fail trying to open
    # it. -M without PCH walks the include graph fresh, which is
    # exactly what we want for header enumeration anyway. Handle the
    # three forms cmake / icpx can emit:
    #   * `-Xclang -include-pch -Xclang <path>` (cmake-Ninja today)
    #   * `-include-pch <path>` (plain clang)
    #   * `-include-pch=<path>` (combined)
    final, i = [], 0
    while i < len(cleaned):
        a = cleaned[i]
        if a == "-o":
            i += 2
        elif a == "-c":
            i += 1
        elif (a == "-Xclang" and i + 3 < len(cleaned)
                and cleaned[i + 1] == "-include-pch"
                and cleaned[i + 2] == "-Xclang"):
            i += 4
        elif a == "-include-pch":
            i += 2
        elif a.startswith("-include-pch="):
            i += 1
        else:
            final.append(a)
            i += 1
    final += ["-M", "-MF", depfile]
    cwd = entry.get("directory") or repo
    proc = subprocess.run(final, cwd=cwd, capture_output=True, text=True)
    if proc.returncode != 0:
        # Trim the stderr — icpx warnings on heavy SYCL templates can
        # be thousands of lines and only the tail is usually useful.
        tail = proc.stderr.strip().splitlines()[-20:]
        return (entry["file"], depfile,
                f"icpx -M exit {proc.returncode}: " + "\n".join(tail))
    return (entry["file"], depfile, None)


def _parse_depfile(path: str, repo: str, src_rel: str) -> list[str]:
    r"""Parse a make-style .d file into the list of `$repo`-relative header paths.

    icpx -M emits the canonical `target: prereq prereq \` form. We:
      1. Drop the `target:` prefix.
      2. Collapse `\<newline>` line continuations.
      3. shlex-split to honour escaped spaces.
      4. Filter to paths under `$repo` (everything else is a stable
         third-party header from icpx/oneapi/torch/etc and is already
         captured as a runtime dep of compileInputs — including those
         in the per-TU srcSubset would just bloat closures with no
         caching benefit).
      5. Drop the TU's own `src_rel` (the caller already includes it
         in srcSubset under src_rel_path). Anything else under `$repo`
         is kept — a future `#include "foo.cc"` (uncommon but legal,
         e.g. for template instantiation files) must end up in
         srcSubset or the TU compile will fail with "file not found".
    """
    with open(path) as f:
        text = f.read()
    if ":" in text:
        text = text.split(":", 1)[1]
    text = text.replace("\\\n", " ").replace("\\\r\n", " ")
    try:
        toks = shlex.split(text)
    except ValueError as e:
        sys.exit(f"FATAL: failed to parse depfile {path}: {e}")
    prefix = repo.rstrip("/") + "/"
    headers = set()
    for t in toks:
        if t.startswith(prefix):
            rel = t[len(prefix):]
            if rel == src_rel:
                continue
            headers.add(rel)
    return sorted(headers)


def _build_tu_cmd(entry: dict, repo: str, pch_out_path: str | None) -> str:
    """Build the per-TU compile-command text with placeholders.

    Substitutions performed at extract time:
      - strip -MD/-MMD/-MT/-MF (dep-tracking is a no-op for re-run)
      - replace `-o <obj>` with `-o __OUT_OBJ__`
      - if -include-pch points at `pch_out_path`, replace that arg with
        `__PCH_PATH__`
      - replace every $repo prefix with `__SRC_SUBSET__`

    mkTU does the inverse at build time: __SRC_SUBSET__ → ${srcSubset},
    __OUT_OBJ__ → $out/tu.o, __PCH_PATH__ → ${pchDrv}/cmake_pch.hxx.pch.
    Anything *outside* $repo (icpx / oneapi / torch / glibc store paths)
    stays as a literal /nix/store/... ref, which Nix's content-scan picks
    up as an srcSubset closure dep — same SYCL frontend the linkDrv
    closure already pulls in.
    """
    if "arguments" in entry:
        args_in = list(entry["arguments"])
    else:
        args_in = shlex.split(entry["command"])
    out_args: list[str] = []
    i = 0
    while i < len(args_in):
        a = args_in[i]
        if a in ("-MD", "-MMD"):
            i += 1
        elif a in ("-MT", "-MF"):
            i += 2
        elif a == "-o" and i + 1 < len(args_in):
            out_args.extend(["-o", "__OUT_OBJ__"])
            i += 2
        elif pch_out_path is not None and a == pch_out_path:
            out_args.append("__PCH_PATH__")
            i += 1
        else:
            out_args.append(a)
            i += 1
    # If PCH is active, the .pch path must have been replaced with
    # __PCH_PATH__ above. cmake-Ninja emits the path as its own
    # token (`-Xclang -include-pch -Xclang <abs>`), which the
    # exact-arg match catches. A future combined form
    # (`-include-pch=<abs>`) would slip past the loop; the next
    # `repo → __SRC_SUBSET__` pass would then rewrite the embedded
    # path to point inside the (PCH-less) srcSubset, producing a
    # silent runtime "no such file". Fail loud at configure time so
    # the regression is obvious.
    if pch_out_path is not None and any(pch_out_path in a for a in out_args):
        sys.exit(
            f"FATAL: pch_out_path {pch_out_path} still embedded in per-TU "
            "cmd after placeholder rewrite — likely a combined -include-pch="
            "<path> arg the exact-match loop missed"
        )
    # Now swap $repo → __SRC_SUBSET__ across every remaining arg.
    repo_clean = repo.rstrip("/")
    out_args = [a.replace(repo_clean, "__SRC_SUBSET__") for a in out_args]
    return shlex.join(out_args)


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
    pch_out_path: str | None = None
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

    # 4. Per-TU dep extraction + per-TU command emission.
    #
    # `cmake -GNinja` only configures, so no .d files exist yet in
    # $repo/build/CMakeFiles/<target>.dir/. We synthesize them here:
    # for every kept TU, run icpx -M (preprocess-only + emit make-style
    # deps) and parse the resulting depfile into the list of headers
    # under $repo. mkTU then materializes a lib.fileset.toSource
    # containing JUST that TU's src + headers + per-TU cmd file, and
    # drops the `${configureDrv}` ref entirely. Net effect: a header
    # edit invalidates only the TUs that transitively #include it
    # (the issue #53 ceiling).
    #
    # The depfile pass is the dominant cost in extract.py — each icpx
    # -M pass on a SYCL TU still runs the full host + spir64 device
    # parser (icpx doesn't short-circuit -fsycl under -M), so per-TU
    # RSS peaks at ~1.5-2 GB on the heavier chunk_prefill heads.
    # Worker count comes from NIX_BUILD_CORES (canonical Nix sandbox
    # env var, set from `--cores`) so this is portable across hosts —
    # the consumer controls parallelism the same way they would for
    # any other nix derivation. Per-TU drvs are single-threaded and
    # ignore -cores, so the consumer can invoke with a high --cores
    # (e.g. `--max-jobs 6 --cores 18`) to get a fast configureDrv
    # without harming the per-TU phase. DYNDRV_DEPFILE_WORKERS is
    # kept as an explicit override for memory-constrained hosts where
    # even the icpx -M closure is too heavy.
    src_by_obj = {obj: obj_to_src[obj] for obj in obj_inputs}
    cc_by_file = {e["file"]: e for e in rewritten}
    safe_by_obj: dict[str, str] = {}
    seen_safe_names: dict[str, int] = {}
    for obj_rel in obj_inputs:
        rel_basename = os.path.basename(src_by_obj[obj_rel])
        base_safe = safe_drv_name(rel_basename)
        n = seen_safe_names.get(base_safe, 0)
        seen_safe_names[base_safe] = n + 1
        safe_by_obj[obj_rel] = base_safe if n == 0 else f"{base_safe}-{n}"

    depfiles_dir = os.path.join(build, "depfiles")
    os.makedirs(depfiles_dir, exist_ok=True)
    per_tu_cmd_dir = os.path.join(repo, "per-tu-cmds")
    os.makedirs(per_tu_cmd_dir, exist_ok=True)

    depgen_tasks: list[tuple] = []
    for obj_rel in obj_inputs:
        src = src_by_obj[obj_rel]
        entry = cc_by_file.get(src)
        if entry is None:
            sys.exit(f"FATAL: no compile_commands entry for src {src}")
        safe = safe_by_obj[obj_rel]
        depfile = os.path.join(depfiles_dir, f"{safe}.d")
        depgen_tasks.append((entry, depfile, repo))

    workers = (
        int(os.environ.get("DYNDRV_DEPFILE_WORKERS") or 0)
        or int(os.environ.get("NIX_BUILD_CORES") or 0)
        or 1
    )
    workers = max(1, min(workers, len(depgen_tasks)))
    print(f"[extract] generating {len(depgen_tasks)} depfiles via icpx -M "
          f"(workers={workers})")
    failures: list[tuple[str, str, str]] = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as ex:
        for src, depfile, err in ex.map(_gen_depfile, depgen_tasks):
            if err is not None:
                failures.append((src, depfile, err))
    if failures:
        for src, depfile, err in failures[:5]:
            print(f"[extract] depfile FAIL for {src}\n  {err}", file=sys.stderr)
        sys.exit(f"FATAL: {len(failures)} depfile generations failed; "
                 "see first few above")

    # 5. Build the per-TU manifest. Paths are stored RELATIVE to $repo,
    # not as absolute /nix/store paths — Nix's builtins.fromJSON returns
    # context-less strings, and downstream code that puts those strings
    # into derivation attributes would trip "is not allowed to refer to
    # a store path" if the JSON content embedded /nix/store/... literals.
    # The Nix code prepends ${configureDrv}/repo/ via interpolation, which
    # carries proper context.
    manifest = []
    for obj_rel in obj_inputs:
        src = src_by_obj[obj_rel]
        if not src.startswith(repo + "/"):
            sys.exit(f"FATAL: source {src} is outside $repo {repo}")
        src_rel = src[len(repo) + 1:]
        safe = safe_by_obj[obj_rel]
        entry = cc_by_file[src]
        depfile = os.path.join(depfiles_dir, f"{safe}.d")
        headers = _parse_depfile(depfile, repo, src_rel)
        cmd_text = _build_tu_cmd(entry, repo, pch_out_path)
        cmd_rel = f"per-tu-cmds/{safe}.txt"
        with open(os.path.join(repo, cmd_rel), "w") as f:
            f.write(cmd_text + "\n")
        manifest.append({
            "safe_name": safe,
            "src_rel_path": src_rel,
            "obj_rel_path": obj_rel,
            "cmd_rel_path": cmd_rel,
            "headers": headers,
        })
    launcher_count = sum(
        1 for m in manifest
        if m["src_rel_path"].endswith(("fmha_xe2.cpp", "paged_decode_xe2.cpp"))
    )
    headers_lens = [len(m["headers"]) for m in manifest]
    print(f"[extract] tu_manifest: {len(manifest)} TUs ({launcher_count} launchers); "
          f"headers per TU min={min(headers_lens)} median={sorted(headers_lens)[len(headers_lens)//2]} "
          f"max={max(headers_lens)}")

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
