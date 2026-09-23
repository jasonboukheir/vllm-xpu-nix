# Patch 0004: skip installation of externally supplied kernel libraries

**Disposition: KEEP unchanged**, after `0001-split-kernel-libs.patch` and before
`0006-forward-mhc-feature-flag.patch`. Upstream v0.1.15 still attempts to install
every enabled additional library from its CMake build subdirectory. Patch 0001
deliberately omits those subdirectories for imported libraries. Patch 0004 is the
remaining necessary install-side half of that packaging interface.

This is a read-only source/package audit for `release-v030-20260922`. Only this
report was written. No source replay, native build, branch/commit, service,
network write, or runtime qualification was performed. The ordinary v0.30 release
preserves eager V1 K4V2 target/draft MTP and vision; paused issue17/DFlash work is
outside this audit.

## Frozen inputs and file identities

All kernel implementation reads used immutable `git show <SHA>:<path>`,
`git ls-tree <SHA>`, or comparisons between frozen SHAs. Repository:
`build-dev/release-v030-20260922/vllm-xpu-kernels`.

| Input | Exact value |
| --- | --- |
| Packaging baseline | `4ac4375aa3d0f649fb4b6b43d9447f413a8b1cc7` |
| Kernel target | v0.1.15, `1c7cbeee1cd0c1481d48f5031f679a47f3f0ef45` |
| Previous kernel upstream base | `e7f20cbc18d741419a84aa57178e1f1cc0b453da` |
| Previous released kernel tip | `2c39287deec5e59cc2656e6d4a6dce51a567c411` |
| Kernel rolling tip recorded in manifest | `9bb1e2d428b8a25b262c0cd6704614bdf6ee3ad8` |
| Companion vLLM target, contextual only | v0.30.0, `ced6857afa0ea7b2e3f0846a62e1394e90f15607` |
| Patch 0004 Git blob | `23ea21f689b931b9d0576f708afef31cad594e51` |
| Patch 0004 SHA-256 | `45c1e47d947d2f576054096278825f0d8fe0ad4787b9536afcc3af73061ad873` |
| Patch 0001 SHA-256 | `0d7b31cef2a1855a67910023d6c35e0a91345376dfd2e82a4f663d55d7809ead` |
| Patch 0006 SHA-256 | `8f2cba38bf161f7d557f3f1fce0b584c6beda21cb315a4b17ee39a4c5411de98` |

Patch 0004 changes only `setup.py`. Record the adjacent CMake input because its
imported-target behavior determines whether the skipped install is correct:

| Kernel input | Target Git blob | SHA-256 of target file |
| --- | --- | --- |
| `setup.py` | `e841c754d353b39fb566e7c63b31ab70a845dab0` | `153b2696834b8161a362870080c59077810b977b36127ba57e472ed51048b4ca` |
| `CMakeLists.txt` | `f7041ef56fc006ed7adeb4432f0a75d3ada7f191` | `5793bac065c0da82323bf53108adcd096f7cd3d40f6cd379552e4b61f3943e79` |

The previous base and released tip both have the same `setup.py` blob listed
above; both have CMake blob `9cda28a59fe06935eb813d3a7ffdabbb63eda8bf`.
The target `tools/envs.py` blob is
`898cc3a7299b7284a6df4103fc81e69ea3d5c76f`.

Packaging inputs inspected at the baseline:

| Path | Git blob |
| --- | --- |
| `nix/vllm-xpu-kernels.nix` | `b28f431f5209ce59278929cab0ae32d6dd77b70a` |
| `nix/mk-kernels.nix` | `9a5940416c63396434f03b3463d08e49ccf86e80` |
| `nix/vllm-xpu-kernels-compose.nix` | `768162d0d0a7c6bf475da13eee1b32207a185af3` |
| `nix/patches/0001-split-kernel-libs.patch` | `fb477faa298e5cd5599f0f883792fcd79af8bbc0` |
| `nix/patches/0006-forward-mhc-feature-flag.patch` | `c8f62b6f8252d80c86c3fada4d9be417438263f8` |

## Requirement and original failure

The Nix factory builds six SYCL-TLA libraries independently, then links Python
extension components against their immutable store paths. This permits reuse of
expensive kernel DSOs and separate base-glue/FA2-binding rebuilding. It applies
to the current Intel XPU package, including Brutus's BMG configuration; the
failure happens during host-side wheel installation and is independent of GPU
execution or model choice.

`nix/vllm-xpu-kernels.nix:125` applies the package sequence 0001, 0004, 0006.
Its lines 163–168 export nonempty paths for each selected
`VLLM_XPU_PREBUILT_<LIB_NAME>_LIB` and lines 112–117 retain the corresponding
libraries in `buildInputs` for linking/fixup. Patch 0001 forwards these values
to CMake and substitutes `SHARED IMPORTED GLOBAL` targets for
`add_subdirectory(...)`. The skipped subdirectory has no generated build tree.

At the frozen target, `setup.py:317–344` nevertheless iterates
`additional_libraries`, issuing `cmake --install . --component <lib_name>` with
`cwd=self.build_temp + file_path`. A missing working directory raises
`FileNotFoundError`; the surrounding handler catches only
`subprocess.CalledProcessError`. Thus the existing warning-and-continue handler
does not cover the missing-directory failure. Patch 0004 checks the matching
override's presence before starting this subprocess. It also keeps external
DSOs out of the wheel's additional-library installation path; runtime resolution
remains the job of the Nix library inputs and ELF fixup.

Packaging history confirms that this is intentional separate ownership:

- `df27cb539c4fdb6fe4d4aefd8ec83751e29774d4` introduced 0004 with this six-line
  install guard.
- `07ea0a5b30d1a2b0b10596c4dd85000a9288491c` removed an accidental duplicate
  install-skip hunk from 0001 and refreshed 0004's context. Its recorded failure
  was a reversed/already-applied patch when both patches supplied the guard.
  Preserve exactly one copy in the final package patch sequence.

## Upstream coverage and interactions

There is no upstream coverage of the prebuilt-library install requirement.
The frozen target's `setup.py` is byte-identical to the previous base and released
tip. The only old-base-to-target change in `setup.py`/`CMakeLists.txt` is
`4111ba268535372314a86f1a2dc930524ba7a92c`, adding
`csrc/xpu/sampler/compact_sampling_mask.cpp` to the `_xpu_C` source list. It does
not alter configuration, target import, library selection, or installation.
The old-base-to-released-tip diff for these two files is empty.

Target `CMakeLists.txt:436–465` still adds the Xe2/default library subdirectories
directly. Target `setup.py:557–582` selects additional libraries using feature
and architecture flags, without consulting prebuilt overrides. Searching target
`setup.py`, `CMakeLists.txt`, and `tools/envs.py` finds no `PREBUILT` interface.

Upstream does have `VLLM_USE_PRECOMPILED`, selecting `precompiled_build_ext` at
`setup.py:597–600` and bypassing extension building altogether. That separate
wheel-extraction path cannot replace the split build: Nix still needs to compile
its current extension sources and link its independently built libraries.

The six names are consistent between 0001, the Nix exports, and the upstream
`additional_libraries` keys:
`attn_kernels_xe_2`, `gdn_attn_kernels_xe_2`, `mqa_logits_kernels_xe_2`,
`mhc_kernels_xe_2`, `grouped_gemm_xe_2`, `grouped_gemm_xe_default`.

Patch 0006 adds `MHC_KERNELS_ENABLED` to setup.py's forwarded feature options;
it does not supersede 0004. Python already uses that option to select additional
libraries, while CMake must receive the same value. In the base component
(`nix/mk-kernels.nix:350–365`), the five non-attention imported libraries,
including MHC, are enabled and FA2 is off. In the FA2 component (lines 376–395), only FA2's
imported library is selected and MHC is disabled. 0006 prevents CMake from
configuring unwanted MHC work in that narrowed component; 0004 handles the
remaining enabled imported libraries in both components.

Xe3 attention is an upstream source-build path, with no corresponding 0001
import hook or Nix export. Both supported split components explicitly set
`VLLM_XPU_ENABLE_XE3P=OFF`; no Xe3 adaptation to 0004 is required. If that feature
scope changes, its split-library interface needs a separate review.

The minimum residual change is the existing six-line guard. It alters no
operator registration, ABI, cache schema, attention arithmetic, MTP behavior, or
vision interface. No vLLM-side operator removal or companion feature change is
justified by this audit.

The guard intentionally checks environment-variable presence, whereas CMake's
`if(...)` checks truth. Empty/false-valued prebuilt overrides could therefore
skip installation despite selecting a source build. Current Nix exports are
nonempty store paths, so this is outside the supported package contract and
does not require adaptation for this release. Do not treat the guard as library
path validation.

## Evidence commands and observed results

Commands were run from the packaging repository unless `git -C "$kernel_repo"`
is shown. Here `kernel_repo=build-dev/release-v030-20260922/vllm-xpu-kernels`,
`target=1c7cbeee1cd0c1481d48f5031f679a47f3f0ef45`,
`old=e7f20cbc18d741419a84aa57178e1f1cc0b453da`, and
`released=2c39287deec5e59cc2656e6d4a6dce51a567c411`.

```sh
git -C "$kernel_repo" status --short
git -C "$kernel_repo" ls-tree "$target" setup.py CMakeLists.txt tools/envs.py
git -C "$kernel_repo" ls-tree "$old" setup.py CMakeLists.txt
git -C "$kernel_repo" ls-tree "$released" setup.py CMakeLists.txt
git -C "$kernel_repo" diff "$old" "$target" -- setup.py CMakeLists.txt
git -C "$kernel_repo" diff --exit-code "$old" "$released" -- setup.py CMakeLists.txt
git -C "$kernel_repo" log --format='%H %s' "$old..$target" -- setup.py CMakeLists.txt
git -C "$kernel_repo" show "$target:setup.py"
git -C "$kernel_repo" show "$target:CMakeLists.txt"
git -C "$kernel_repo" grep -n -E 'PREBUILT|IMPORTED|additional_libraries|VLLM_USE_PRECOMPILED' "$target" -- setup.py CMakeLists.txt tools/envs.py
git log --format='%H %s' --all -- nix/patches/0004-skip-prebuilt-additional-libs.patch
git show 07ea0a5b30d1a2b0b10596c4dd85000a9288491c -- nix/patches/0001-split-kernel-libs.patch nix/patches/0004-skip-prebuilt-additional-libs.patch
sha256sum nix/patches/0001-split-kernel-libs.patch nix/patches/0004-skip-prebuilt-additional-libs.patch nix/patches/0006-forward-mhc-feature-flag.patch benchmark-results/release-v030-20260922/upstream-patch-preflight.json
```

Observed: source status was clean at the target; file identities and single
unrelated CMake addition are recorded above; released comparison exited zero.

The parent's existing exact-tag archive preflight is recorded in
`benchmark-results/release-v030-20260922/upstream-patch-preflight.json`, SHA-256
`322d634a5e36f8fc2494ca7d902c0d90c6ece94e96dd58f323e95fa6b9674d09`.
Its package rows all returned zero for 0001, 0004, and 0006. The parent reports
using `--fuzz=0`; the JSON records 0004 succeeding at line 343 with a 12-line
offset. This proves applicability to the upstream target, not to a future
replayed source or a successful native build.

A read-only behavioral harness ran through
`build-dev/release-v030-20260922/vllm-xpu-kernels/.venv/bin/python -B - <<'PY'`.
It parsed frozen target `setup.py` and the parent's package preflight `setup.py`
with `ast`, extracted the actual `_is_enabled`, `additional_libraries` selection,
and `cmake_build_ext.build_extensions` method, and executed those nodes with
stub build/configure methods and a recording `subprocess`. The stub raised
`FileNotFoundError` only when an additional install attempted to enter an
imported library's deliberately absent directory. No CMake or native build ran.
The harness also inserted exactly the plus-lines from 0004 into the target
method in memory and asserted AST equality with the package preflight method.

| Scenario | Result |
| --- | --- |
| Upstream method, base component's five imports | Unhandled `FileNotFoundError` on `gdn_attn_kernels_xe_2`, as expected |
| Patched method, same base imports | Zero additional installs; extension install preserved |
| Upstream method, FA2 import | Unhandled `FileNotFoundError` on `attn_kernels_xe_2`, as expected |
| Patched method, same FA2 import | Zero additional installs; extension install preserved |
| Patched method, no imports, Xe3 off | All six source-library installs preserved |
| Patched method, only Xe2 attention imported, Xe3 off | Exactly the other five source-library installs preserved |
| Patched method, six imports, Xe3 enabled | Exactly the unsupplied `attn_kernels_xe_3` source install preserved |

All seven assertions passed; every case also preserved extension installation.
This distinguishes the install-dispatch behavior with and without 0004, under
a controlled model of missing directories. It is not evidence that real
compilation, CMake generation, wheel assembly, or runtime loading succeeded.

Parent-preflight package files inspected after the three patches:

| File | Git-style blob hash (`git hash-object`, without `-w`) |
| --- | --- |
| `setup.py` | `4b8eb549118db648c71cf8e728a0b72304f17f59` |
| `CMakeLists.txt` | `6b0274cdbc6b04c85f56d43f83199498c60893be` |

The preflight `setup.py` SHA-256 is
`685a84d7af9046d75cd8ce08b68bd78ea802464613fe9b3838fc14c190ff21d6`.

## Remaining final-source and build gates

1. After parent-owned source replay, record the final kernel SHA and compare its
   `setup.py` and `CMakeLists.txt` blobs to the target identities above. If the
   inputs changed, review configuration, additional-library selection, and
   installation again; applicability alone does not establish equivalence.
2. Apply 0001, 0004, 0006 with `--fuzz=0` to a disposable archive of that exact
   final source. Verify the prebuilt install guard occurs exactly once and that
   0006 still forwards MHC flags. Revalidate the component feature/export matrix
   and any source-projection changes. Record final patched file identities.
3. Build the actual paired Nix candidate with Brutus's current overrides,
   including its independently built DSOs, base glue, FA2 binding, and composed
   package. Inspect logs for absence of attempted installs from imported-library
   subdirectories, and confirm required extension modules are present.
4. Inspect ELF dependencies/RPATH and package contents. The final extensions
   must resolve the intended split libraries; copied duplicate DSOs or references
   to intermediate donor packages must not silently substitute for the selected
   libraries. The existing composer checks the FA2 library's `DT_NEEDED` basename,
   attention RPATH, component contents, and disallowed donor references.
5. Complete the ordinary release's paired import/HTTP/GPU qualification of the
   newly built artifacts for eager V1 K4V2 target/draft MTP and vision. This patch
   has no runtime performance algorithm requiring its own benchmark comparison;
   its benefit is successful, reusable split packaging. Old binaries or this
   source-only audit do not qualify the new release.

No unresolved source question warrants dropping or adapting 0004 for the frozen
target and current Nix contract. Final-source review and actual build/runtime
qualification remain open parent-owned release gates.
