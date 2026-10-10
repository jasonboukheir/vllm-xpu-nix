# P04 — skip installation of externally supplied kernel libraries

**Disposition: KEEP unchanged.** The split Nix build still needs this guard;
upstream supplies no equivalent behavior. The smallest residual change is the
existing environment-variable check and `continue` in the additional-library
installation loop.

## Scope and immutable inputs

- Packaging patch: `nix/patches/0004-skip-prebuilt-additional-libs.patch`.
- Stable patch ID: `43178f031490c2ef797d99cda1dfabc885b65252`.
- Target file: kernel `setup.py`, `cmake_build_ext.build_extensions()`.
- Old kernel base: `1c7cbeee1cd0c1481d48f5031f679a47f3f0ef45`.
- Old kernel fork tip: `193cee080d654ff67c4f8bfaf924201bc88b9cf5`.
- New kernel base: `ddf336d86e3c8602888572a3502f951abd51df12`, frozen
  official `release/0.1.15.4` source for the paired vLLM 0.31 update.
- Companion vLLM: `db9527a46873454610df6dbedf79a36d6bf1a7f6`.

Read the source repository's `AGENTS.md`, update workflow section 5,
commit-audit contract, and `docs/fork-delta.md`. Inspected immutable git objects;
no source worktree, branch, pin, or service was changed.

## Requirement and dependencies

`0001-split-kernel-libs.patch` imports separately built Nix shared libraries as
CMake `SHARED IMPORTED` targets. For those libraries, CMake never adds the
library source subdirectory and never creates its corresponding build directory.
The unchanged upstream setuptools driver nevertheless lists enabled libraries
in `additional_libraries` and tries `cmake --install` with that missing directory
as its process working directory. That raises `FileNotFoundError`, which the
upstream handler for `subprocess.CalledProcessError` does not catch.

P04 skips precisely the additional-library install attempt whose
`VLLM_XPU_PREBUILT_<UPPERCASE_LIBRARY_NAME>_LIB` variable is present. It does not
change normal extension compilation or extension installation. More precisely
than the existing fork-delta summary, this patch prevents invalid installation;
P01's imported CMake targets are what prevent recompiling the supplied libraries.

`nix/vllm-xpu-kernels.nix` applies P01 then P04, includes each selected split
library in `buildInputs`, and exports the matching nonempty Nix store path in
`preBuild` (currently lines 125–128 and 163–168). This preserves separately cached
closures for attention, GDN, MQA, MHC and the two grouped-GEMM variants. The
requirement concerns packaging, not a hardware-specific runtime optimization.
No vLLM operator ABI or kernel schema changes are involved.

Removing P04 alone breaks that split packaging contract. Removing both P01 and
P04 would require deliberately abandoning or redesigning the split build. That
requirement has not been retired by this release request.

## Upstream delta and alternatives

Between the exact old and new bases, `setup.py` and `tools/envs.py` have no diff.
Both `setup.py` objects have blob ID
`e841c754d353b39fb566e7c63b31ab70a845dab0`. The only change in root
`CMakeLists.txt` is the Torch requirement from 2.13 to 2.14. None of the four
upstream commits replaces the split-library installation behavior:

| Upstream commit | Change |
| --- | --- |
| `f4b729ae7fb3eb22c564606ec8b4d184c7c8bc4a` | Ragged speculative GDN traversal, #600 |
| `d09702a637cfea2a0c02909af82bd789d26ea319` | Torch 2.14 dependency, #525 |
| `885b120ffa082312b7d93c2be7a1e3b3dd79672f` | Paged-decode kernel configuration, #602 |
| `ddf336d86e3c8602888572a3502f951abd51df12` | `topk_softplus_sqrt` interface, #603 |

At the frozen target, `setup.py:327–344` still installs every dictionary entry
and catches only `CalledProcessError`. The dictionary at lines 557–582 filters
feature/architecture enablement, but never prebuilt-library ownership. Disabling
the feature to remove that entry would also remove desired bindings or library
links; it is not equivalent to supplying that enabled feature externally.

Upstream's `VLLM_USE_PRECOMPILED` alternative, at lines 501–549 and 597–600,
extracts a complete existing wheel and disables all extension building. It
cannot combine independently built Nix kernel libraries with freshly built
local binding extensions and therefore does not replace this patch.

The existing setup dictionary also includes XE3 attention when XE3 is enabled.
P04 is generic enough to skip a future explicitly supplied XE3 library, but P01
and the current Nix library set provide no such target. The parent/P01 audit
must verify architecture selection; P04 does not solve or alter that independent
build-composition concern. XE3 enumeration already existed at the old base.

## Checks run and remaining qualification

1. `git diff` confirmed no old-to-new changes to `setup.py` or `tools/envs.py`;
   immutable blob IDs independently confirmed the `setup.py` identity.
2. `patch --dry-run --fuzz=0 -p1` against the frozen upstream `setup.py` in a
   disposable directory passed. Its single hunk applied at line 325 with offset
   -6 and zero fuzz. No patch refresh is necessary for this target.
3. An isolated behavioral check used the source repository's Python 3.12.13
   virtualenv, extracted the actual `build_extensions` method with `ast`, and
   mocked subprocess execution while preserving the missing-working-directory
   failure. The three observed cases were:

   | Case | Result | Additional-library installation attempts |
   | --- | --- | --- |
   | Frozen upstream, prebuilt variable set, absent library build directory | `FileNotFoundError` | 1 |
   | P04 applied, same prebuilt environment | Pass | 0 |
   | P04 applied, no prebuilt variable, normal build directory present | Pass | 1 |

This demonstrates both the unmet upstream requirement and preservation of the
ordinary installation path. It did not compile kernels or run an actual wheel
build. The parent's paired native build, library-resolution/import checks, and
host-profile serving qualification remain required. Nix exports nonempty
prebuilt paths; hypothetical empty variables are outside the current contract
and do not justify broadening this release's patch.
