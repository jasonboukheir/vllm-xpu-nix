# Patch 0001: KEEP unchanged, pending final replay verification

Audited `nix/patches/0001-split-kernel-libs.patch` against frozen kernel
v0.1.15, `1c7cbeee1cd0c1481d48f5031f679a47f3f0ef45`, following stage 5 of
`.agents/skills/vllm-xpu-update/references/workflow.md`. Host was Brutus.
The source checkout remained at that target during evaluation. This is a
packaging/source audit, not build or runtime qualification.

## Purpose and upstream alternatives

The patch changes `CMakeLists.txt` and `setup.py`. For six existing Xe2/default
kernel targets, it permits a `SHARED IMPORTED GLOBAL` library with an explicit
path and public source include directories instead of `add_subdirectory`.
Without a prebuilt override the original native build remains intact. It also
forwards the six prebuilt paths and both oneDNN/offline FetchContent options
from the Python build environment to CMake.

The frozen upstream release still builds these targets through
`add_subdirectory`; it has no equivalent prebuilt-target interface or forwarding
of these eight settings. Its `VLLM_USE_PRECOMPILED` path extracts an entire wheel
and disables extension compilation, which cannot replace this composition of
source-built bindings with individually built libraries. Upstream already
provides shared-library targets and feature gates, but these alone do not
provide the imported-target interface. **KEEP**, with no patch modification
required for the frozen release.

`git log` and `git diff` for
`e7f20cbc18d741419a84aa57178e1f1cc0b453da..2c39287deec5e59cc2656e6d4a6dce51a567c411`
restricted to both patch targets produced no changes. Thus none of the complete
released fork stack touches either file. Between that old base and v0.1.15,
`setup.py` is unchanged and CMake only adds
`csrc/xpu/sampler/compact_sampling_mask.cpp` to `_xpu_C`; the patch's target
creation and forwarding regions remain semantically unchanged.

## Consumers and composition

| Consumer | Patch interaction |
| --- | --- |
| `nix/vllm-xpu-lib.nix` | Applies 0001 then 0005; directly configures CMake and builds one named Ninja target. No prebuilt override is supplied, so the original `add_subdirectory` fallback builds that library. Offline options are supplied directly as CMake flags. |
| `nix/vllm-xpu-kernels.nix` | Applies 0001, 0004, then 0006; exports the enabled libraries and offline options for `setup.py` to forward. 0001 is required to avoid rebuilding those libraries and to support the narrow binding projection. |
| Base glue | Imports GDN, MQA logits, MHC, grouped GEMM Xe2 and grouped GEMM default. CMake links them into `_xpu_C`; FA2 is disabled. The component also builds `_C`, `_moe_C` and the allocator. |
| FA2 binding | Imports only `attn_kernels_xe_2`, linked into `_vllm_fa2_C`; other feature families are disabled. |

The imported targets preserve the source include paths their consumers need.
The helpers' native compile definitions are private; architecture definitions
for extension compilation still come from top-level
`SYCL_TLA_COMPILE_OPTIONS`. The omitted build-directory include is not needed
by the inspected binding declarations: FA2 consumes retained `.h` interfaces,
while template generation remains inside the separately built attention DSO.

0004 remains a companion dependency: imported targets have no local additional
library installation tree, so setup must skip installing those intermediates.
0006 remains necessary to forward the disabled MHC setting for FA2. This audit
does not replace those patches' independent audits. The final composition
checks extension counts, the FA2 DSO dependency/RPATH and absence of donor-package
references at build time; those checks have not run for this candidate.

## Architecture, AOT and projected sources

Offline Nix evaluation of the actual `mkVllmXpuKernels` factory succeeded for
both generic settings and overrides read directly from
`/home/jasonbk/.config/nix/hosts/brutus/services/vllm-xpu/package.nix`.
It inspected generated `cmakeFlags`, `ninjaFlags`, feature environment values,
prebuilt exports and AOT exports, without realizing native derivations.

- All six split libraries and both native glue components explicitly disable
  `VLLM_XPU_ENABLE_XE3P`. The default grouped-GEMM library disables Xe2; the
  remaining five retain Xe2 and disable the default architecture. Base glue
  enables Xe2/default; FA2 enables only Xe2.
- Both AOT variables are exported as empty strings for generic packages and
  exactly `bmg` for Brutus. Upstream's `DEFINED ENV` handling preserves the
  empty override. With Xe3p disabled, upstream cannot append `cri,cri-a0` or
  select its `intel_gpu_cri` SYCL-TLA default. Xe2/default link helpers retain
  the existing 256-GRF tuning for explicit AOT.
- Brutus's attention derivation alone receives the default prefill/decode
  preset selectors and appended variants. Sibling libraries receive neither
  selectors nor config-file writes.
- Inspected the actual Nix source projections. All retain `CMakeLists.txt`,
  `setup.py`, `cmake/utils.cmake` and `tools/envs.py`. The six selected library
  subtrees retain all 24/10/3/5/7/6 existing files respectively for attention,
  GDN, MQA, MHC, GEMM Xe2 and GEMM default. Shared grouped-GEMM files remain
  included by their common-root projection.
- FA2 retains its seven binding/interface/declaration files and the required
  common header roots, while omitting the Xe2 implementation/CMake directory
  contents. The imported attention target is therefore essential. Base glue
  omits the attention roots and retains all 41 explicitly named non-attention
  source/header paths inspected from upstream CMake, including the new compact
  sampling-mask implementation. Dynamic common-source globs remain within its
  retained source tree. This is a source-presence check, not compiler validation.

## Checks and evidence identity

The parent's `benchmark-results/release-v030-20260922/upstream-patch-preflight.json`
records successful application with `--fuzz=0` in both patch sequences:
CMake offsets +22 and setup offset +7 for 0001. Its SHA256 is
`322d634a5e36f8fc2494ca7d902c0d90c6ece94e96dd58f323e95fa6b9674d09`.

An isolated AST execution of the option-forwarding portion of the patched
`setup.py` passed for base and FA2 environments: respectively five and one
prebuilt paths, both offline overrides, Xe3p OFF, and MHC ON/OFF reached CMake.
This used the repository's Python 3.12 virtualenv and did not import setup,
configure a native project, compile, or initialize an XPU.

The initial sandboxed Nix attempts could not write the user cache/connect to
the daemon; the authorized offline evaluation succeeded with local-daemon
access. No source, ref, lock, service or issue changes were made.

| Input | Identity |
| --- | --- |
| Patch SHA256 | `0d7b31cef2a1855a67910023d6c35e0a91345376dfd2e82a4f663d55d7809ead` |
| Target CMake Git blob | `f7041ef56fc006ed7adeb4432f0a75d3ada7f191` |
| Target CMake SHA256 | `5793bac065c0da82323bf53108adcd096f7cd3d40f6cd379552e4b61f3943e79` |
| Target setup Git blob | `e841c754d353b39fb566e7c63b31ab70a845dab0` |
| Target setup SHA256 | `153b2696834b8161a362870080c59077810b977b36127ba57e472ed51048b4ca` |
| `nix/mk-kernels.nix` SHA256 | `bf6b026722e66168b9ad7dd66ae12b86b736f9da9b1ae9067c86d962759c286d` |
| `nix/vllm-xpu-lib.nix` SHA256 | `4e0975eaa17604e94e0cc61cf8aa4bbb5f2ad648f88268e886f946af2b9d9063` |
| `nix/vllm-xpu-kernels.nix` SHA256 | `a04700b23e39b9531c97b2ea9c6d25b856801ee8299e861330bed2451b303647` |
| `nix/lib/kernel-lib-src.nix` SHA256 | `cf9597b2cfa02304aea1124864ae561828f9cf5a841f7628235b0117eb98ec8e` |
| `nix/lib/kernel-glue-src.nix` SHA256 | `1d745679df67d1e4de877dae6be6eda73ac6797606cd09c07e27f1c1d85cd40b` |
| Brutus override SHA256 | `7ac223d83a815c55976851aa1181187ae81aa8d9ae17f0c5a1fabed9e5779b95` |

## Final-source gate and limits

Final replay and source locking were pending when this audit ran. Before using
this disposition for stage 5 acceptance, compare the final pair's two target
blobs with the identities above, rerun both patch sequences with `--fuzz=0`,
and verify generated options/source projections against the final source and
actual Brutus overrides. Unchanged target blobs support reusing the patch
mechanics finding; they do not alone attest any changed interfaces, included
headers, or kernel implementation. Any behavior-affecting replay adaptation
requires this report's relevant findings to be revalidated. Native linking,
ELF resolution, imports, model functionality, performance and GPU qualification
remain deferred to the release workflow.
