#!/usr/bin/env python3
"""Configure-time extractor for the dyn-drv attn_kernels_xe_2 build.

Run from configureDrv.installPhase after `cmake -GNinja` has populated
$out/repo/build/. Produces these artefacts in $out/repo/:

  - compile_commands.json (rewritten in-place: cmake's $build paths -> $out/repo)
  - tu_manifest.json: every .cpp the link target consumes. Each entry
        carries the .cpp src path (relative to $out/repo), the
        ninja-relative .o path, a list of transitive header paths the
        TU pulls from $out/repo (extracted from a clang-scan-deps run
        over the kept subset of compile_commands.json), and a path to
        a per-TU compile-command text file. Excludes the cmake-
        synthesised cmake_pch.hxx.cxx source.
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

The dep pass writes a filtered compile_commands.scan.json (SYCL flags,
PCH includes, and dep-tracking flags stripped — clang-21 doesn't grok
the icpx-specific `-fno-sycl-instrument-device-code` family, and the
PCH path isn't on disk yet at configure time) and runs
clang-scan-deps once over it with -mode=preprocess-dependency-directives.
That mode lexes only directive-relevant tokens, sharing the parsed
header graph across TUs in memory — ~30x faster than running icpx -M
per TU, which doesn't short-circuit -fsycl's device-side parse and
forks two `clang -cc1` invocations per call.

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


_SCAN_DROP_EXACT = {"-fsycl"}
_SCAN_DROP_PREFIX = ("-fsycl-", "-fno-sycl-", "-Xsycl-target-")


def _strip_scan_unfriendly(args: list[str]) -> list[str]:
    """Strip flags that prevent stock clang-scan-deps from parsing an icpx command.

    - dep-tracking flags (-MD/-MMD/-MT/-MF): redundant for scan-deps.
    - `-include-pch <path>` in all three forms: the PCH doesn't exist at
      configure time (cmake-Ninja hasn't run the PCH compile yet) and
      scan-deps would error trying to read it. The host-only header walk
      we get without PCH is a strict superset of the PCH'd walk anyway.
    - `-fsycl` / `-fsycl-*` / `-fno-sycl-*` / `-Xsycl-target-*`: stock
      upstream clang-21 doesn't recognise the icpx-specific subset
      (`-fno-sycl-instrument-device-code`, `-fno-sycl-id-queries-fit-in-int`,
      ...) and errors out. We're already only collecting `$repo`-relative
      headers, which are the same set across host and device passes for
      this codebase — losing the device-side parse doesn't change the
      srcSubset content.
    """
    cleaned, i = [], 0
    while i < len(args):
        a = args[i]
        if a in ("-MD", "-MMD"):
            i += 1
        elif a in ("-MT", "-MF"):
            i += 2
        elif (a == "-Xclang" and i + 3 < len(args)
                and args[i + 1] == "-include-pch"
                and args[i + 2] == "-Xclang"):
            i += 4
        elif a == "-include-pch":
            i += 2
        elif a.startswith("-include-pch="):
            i += 1
        elif a in _SCAN_DROP_EXACT:
            i += 1
        elif any(a.startswith(p) for p in _SCAN_DROP_PREFIX):
            i += 1
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


def _run_scan_deps(
    scan_cdb_path: str, scan_out_path: str, repo: str, workers: int
) -> dict[str, list[str]]:
    """Invoke clang-scan-deps once over the filtered CDB.

    Returns a map from each entry's `-o` argument (the rule target in
    scan-deps' make output) to the list of `$repo`-relative header
    paths the TU pulls in. Non-$repo deps (stdlib, sycl, torch, ...)
    are filtered out — they're already in compileInputs' runtime
    closure and including them in per-TU srcSubsets would only inflate
    closure size without any caching benefit.

    scan-deps' stdout is streamed to `scan_out_path` rather than
    captured into Python memory: on the full ~3.8k-TU unpruned build
    the output exceeds 1.5 GB, which would peg `subprocess.run(...,
    capture_output=True)` at multi-GB RSS plus extra copies during
    the parse. Streaming + line-by-line read keeps RSS flat at ~100 MB.
    """
    with open(scan_out_path, "w") as out_f:
        proc = subprocess.run(
            [
                "clang-scan-deps",
                "-compilation-database", scan_cdb_path,
                "-format", "make",
                # Lexes only preprocessor-directive-relevant tokens
                # per header and shares the parsed include graph
                # across TUs. This mode is the actual source of the
                # speedup vs running `icpx -M` per TU — without it
                # scan-deps still walks each TU's headers via a full
                # lex and the cross-TU caching wins shrink to ~3-5x.
                "-mode", "preprocess-dependency-directives",
                "-j", str(workers),
            ],
            stdout=out_f,
            stderr=subprocess.PIPE,
            text=True,
            cwd=repo,
        )
    if proc.returncode != 0:
        # Truncate stderr — scan-deps can spew per-TU errors.
        err_tail = "\n".join(proc.stderr.strip().splitlines()[-30:])
        sys.exit(
            f"FATAL: clang-scan-deps exit {proc.returncode}\n"
            f"--- stderr tail ---\n{err_tail}"
        )

    # Make output: one rule per TU, of the form
    #   target: prereq prereq \
    #     prereq prereq \
    #     prereq
    # Targets and prereqs are space-separated; spaces in paths would be
    # escaped as `\ ` in make's grammar, but every path we emit lives
    # under either $repo (a /nix/store/... path) or a SYCL / torch
    # store path, and the Nix store rejects component names with
    # spaces — so a fast `str.split()` is safe and saves the
    # multi-minute `shlex.split` we'd otherwise do across 1.5 GB of
    # output. Rules from concurrent worker threads can interleave but
    # each individual rule is emitted atomically.
    prefix = repo.rstrip("/") + "/"
    by_target: dict[str, list[str]] = {}
    current_target: str | None = None
    current_headers: set[str] | None = None

    def _flush():
        if current_target is not None:
            # Path normalisation: icpx -M emitted `./` in some paths
            # from relative `#include "./collective/foo.hpp"`
            # directives; scan-deps collapses these. Normalising here
            # also defends against future relative-include patterns
            # producing duplicate entries in srcSubset's cp loop.
            by_target[current_target] = sorted(
                {os.path.normpath(p) for p in current_headers}
            )

    with open(scan_out_path) as f:
        for line in f:
            stripped = line.rstrip("\n")
            continuation = stripped.endswith("\\")
            if continuation:
                stripped = stripped[:-1]
            tokens = stripped.split()
            if not tokens:
                if not continuation:
                    _flush()
                    current_target = None
                    current_headers = None
                continue
            # New rule starts when the first token ends with ':'.
            if current_target is None:
                head = tokens[0]
                if not head.endswith(":"):
                    sys.exit(
                        f"FATAL: scan-deps emitted a continuation line "
                        f"with no active target: {line!r}"
                    )
                _flush()
                current_target = head[:-1]
                current_headers = set()
                deps = tokens[1:]
            else:
                deps = tokens
            for t in deps:
                if t.startswith(prefix):
                    current_headers.add(t[len(prefix):])
            if not continuation:
                _flush()
                current_target = None
                current_headers = None
    _flush()
    return by_target


def _build_tu_cmd(
    entry: dict,
    repo: str,
    pch_out_path: str | None,
    torch_prefix: str,
    oneapi_prefix: str,
) -> str:
    """Build the per-TU compile-command text with placeholders.

    Substitutions performed at extract time:
      - strip -MD/-MMD/-MT/-MF (dep-tracking is a no-op for re-run)
      - replace `-o <obj>` with `-o __OUT_OBJ__`
      - if -include-pch points at `pch_out_path`, replace that arg with
        `__PCH_PATH__`
      - replace every $repo prefix with `__SRC_SUBSET__`
      - rewrite the absolute compiler-binary arg under
        `<oneapi>/compiler/latest/bin/<tool>` to its bare basename
        (envSetup puts the SYCL bin on PATH, so `icpx` resolves)
      - sweep every remaining `<torch>` prefix to `__TORCH_PREFIX__` and
        every `<oneapi>` prefix to `__ONEAPI_PREFIX__`

    mkTU does the inverse at build time: __SRC_SUBSET__ → ${srcSubset},
    __OUT_OBJ__ → $out/tu.o, __PCH_PATH__ → ${pchDrv}/cmake_pch.hxx.pch,
    __TORCH_PREFIX__ → ${torch-xpu}, __ONEAPI_PREFIX__ →
    ${intel-oneapi-base}. Per-TU cmd file bytes become invariant to
    icpx-toolchain and torch-xpu store-path bumps; the Nix-side
    interpolation re-attaches store-path context via buildInputs, so
    the drv closure tracks the right runtime deps. Anything outside
    those prefixes (gcc / glibc) stays as a literal /nix/store ref and
    flows into the input hash via envSetup's stdenv.cc.* interpolations
    — left for a follow-up wrapCCWith refactor.
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
    # cmake bakes /nix/store/<oneapi>/compiler/latest/bin/icpx into args[0]
    # of every entry. envSetup already puts that bin dir on PATH, so a
    # bare basename (`icpx`, `icx`, ...) resolves at build time and
    # untangles the per-TU cmd file from oneapi store-path bumps.
    oneapi_clean = oneapi_prefix.rstrip("/")
    sycl_bin_prefix = f"{oneapi_clean}/compiler/latest/bin/"
    out_args = [
        os.path.basename(a) if a.startswith(sycl_bin_prefix) else a
        for a in out_args
    ]
    # Prefix sweep over the remaining args. Any -I/-isystem/-L/-include
    # arg (combined `-Iflag/path` or separate `-I path`) that points
    # under torch-xpu or intel-oneapi-base picks up the placeholder.
    torch_clean = torch_prefix.rstrip("/")
    out_args = [
        a.replace(torch_clean, "__TORCH_PREFIX__")
         .replace(oneapi_clean, "__ONEAPI_PREFIX__")
        for a in out_args
    ]
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
    ap.add_argument("--torch-prefix", required=True,
                    help="torch-xpu store path; rewritten to __TORCH_PREFIX__ "
                         "in every per-TU cmd file")
    ap.add_argument("--oneapi-prefix", required=True,
                    help="intel-oneapi-base store path; rewritten to "
                         "__ONEAPI_PREFIX__ in every per-TU cmd file. The "
                         "compiler binary under <prefix>/compiler/latest/bin "
                         "is collapsed to a bare basename instead.")
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
    # for every kept TU, scan its include graph and record the headers
    # under $repo. mkTU then materializes a lib.fileset.toSource
    # containing JUST that TU's src + headers + per-TU cmd file, and
    # drops the `${configureDrv}` ref entirely. Net effect: a header
    # edit invalidates only the TUs that transitively #include it
    # (the issue #53 ceiling).
    #
    # We use clang-scan-deps in `preprocess-dependency-directives`
    # mode over a filtered copy of compile_commands.json. The earlier
    # approach (`icpx -M` per TU, fanned out through a
    # ProcessPoolExecutor) was the dominant cost in extract.py — icpx
    # doesn't short-circuit `-fsycl` under `-M` and forks a host +
    # spir64 device `clang -cc1` pass for every TU, taking ~24 s per
    # TU at peak (~30 min wall on the kept ~1.4k TU subset, ~80 min on
    # the unpruned 3.8k TU set). scan-deps shares its parsed-include
    # graph across worker threads in a single process, so each
    # repeated `#include` is lexed once and the per-TU work is
    # bottlenecked on `read()` rather than the SYCL frontend.
    # Measured: ~10 s wall on the same 1.4k TU set.
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

    per_tu_cmd_dir = os.path.join(repo, "per-tu-cmds")
    os.makedirs(per_tu_cmd_dir, exist_ok=True)

    # Build the filtered scan-deps compilation database. We carry only
    # the entries we'll actually scan (matches what `icpx -M` did) so
    # scan-deps doesn't blow up on icpx-specific flags in unrelated
    # entries (e.g. oneDNN's `-fiopenmp`). Each entry's `-o` is kept
    # intact so the make-format rule target lines up with `obj_rel`.
    scan_cdb = []
    for obj_rel in obj_inputs:
        src = src_by_obj[obj_rel]
        entry = cc_by_file.get(src)
        if entry is None:
            sys.exit(f"FATAL: no compile_commands entry for src {src}")
        if "arguments" in entry:
            args_in = list(entry["arguments"])
        else:
            args_in = shlex.split(entry["command"])
        scan_cdb.append({
            "directory": entry.get("directory", repo),
            "file": entry["file"],
            "arguments": _strip_scan_unfriendly(args_in),
        })
    scan_cdb_path = os.path.join(build, "compile_commands.scan.json")
    with open(scan_cdb_path, "w") as f:
        json.dump(scan_cdb, f)
    scan_out_path = os.path.join(build, "scan-deps.mk")

    # Worker count: NIX_BUILD_CORES (canonical Nix sandbox env var,
    # set from --cores). DYNDRV_DEPFILE_WORKERS overrides for memory-
    # constrained hosts. scan-deps is far cheaper per TU than icpx -M
    # ever was, so the override exists mainly for symmetry with the
    # icpx-era knob and is unlikely to need use in practice.
    workers = (
        int(os.environ.get("DYNDRV_DEPFILE_WORKERS") or 0)
        or int(os.environ.get("NIX_BUILD_CORES") or 0)
        or 1
    )
    workers = max(1, min(workers, len(scan_cdb)))
    print(f"[extract] scanning {len(scan_cdb)} TUs via clang-scan-deps "
          f"(j={workers})")
    deps_by_target = _run_scan_deps(scan_cdb_path, scan_out_path, repo, workers)
    if len(deps_by_target) != len(scan_cdb):
        print(
            f"[extract] WARN: scan-deps emitted {len(deps_by_target)} rules "
            f"but cdb has {len(scan_cdb)} entries",
            file=sys.stderr,
        )

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
        # scan-deps emits one rule per CDB entry with the rule target
        # equal to the `-o` argument (cmake puts the ninja-relative
        # `obj_rel` there). Drop the TU's own src to avoid listing
        # it twice in srcSubset.
        deps = deps_by_target.get(obj_rel)
        if deps is None:
            sys.exit(
                f"FATAL: clang-scan-deps emitted no rule for {obj_rel} "
                f"(TU {src_rel})"
            )
        headers = sorted(d for d in deps if d != src_rel)
        cmd_text = _build_tu_cmd(
            entry, repo, pch_out_path, args.torch_prefix, args.oneapi_prefix,
        )
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
