# P01 — split kernel libraries: KEEP unchanged

Audited `nix/patches/0001-split-kernel-libs.patch` independently against frozen
kernel target `ddf336d86e3c8602888572a3502f951abd51df12` (wheel 0.1.15.4),
old upstream `1c7cbeee1cd0c1481d48f5031f679a47f3f0ef45`, and old fork tip
`193cee080d654ff67c4f8bfaf924201bc88b9cf5`. The paired vLLM target is
`db9527a46873454610df6dbedf79a36d6bf1a7f6`.
Read the kernel `AGENTS.md`, workflow stage 5, and `docs/fork-delta.md`.
Stable patch ID: `c1f5da06a4b36c909e5e88bdc0a9ebdfe2425791`.

## Purpose and upstream evidence

The patch lets CMake import six separately built shared libraries: Xe2
attention, GDN, MQA logits, MHC, and grouped GEMM for Xe2/default. It preserves
the normal source-build fallback when a prebuilt path is absent. Its setup.py
addition forwards those six paths and the two offline oneDNN FetchContent
options. Nix uses these paths to link the base Python extensions and the
separate FA2 binding to reusable kernel DSOs.

The four-commit upstream delta adds no replacement for this interface:

- `d09702a637cfea2a0c02909af82bd789d26ea319` bumps the root CMake Torch
  expectation from 2.13 to 2.14. Its `cmake/utils.cmake` change sets C++20 for
  the Xe3 library; this does not affect the selected Xe2/default composition.
- `885b120ffa082312b7d93c2be7a1e3b3dd79672f` adds a paged-decode preset row,
  and `f4b729ae7fb3eb22c564606ec8b4d184c7c8bc4a` fixes ragged GDN traversal.
  Neither changes the split-library API or source locations.
- `ddf336d86e3c8602888572a3502f951abd51df12` changes top-k ops, outside this
  import machinery.

At the frozen target, `CMakeLists.txt:435–505` still directly adds the six
source subdirectories. `setup.py:187–233` still does not forward the prebuilt
paths or offline overrides. setup.py is byte-identical between old fork tip
and target (SHA256 `153b2696834b8161a362870080c59077810b977b36127ba57e472ed51048b4ca`).
The old fork has no root CMake/setup.py edits to replay. Dropping P01 therefore
breaks the existing split/prebuilt design; no upstream refactor reduces its
necessary logic in this release.

## Source projection, architecture, and AOT contracts

`nix/lib/kernel-lib-src.nix` retains each selected feature subtree, including
its generators/configs, and the common CMake/setup/header sources. Both grouped
GEMM libraries retain their shared grouped_gemm tree. Feature flags in
`nix/mk-kernels.nix` prevent CMake entering filtered-out siblings. The new
paged-decode preset remains inside the attention projection. Shared root CMake
and cmake/utils changes intentionally invalidate all split projections.

The base-glue projection excludes attention and disables FA2; the FA2 projection
keeps the binding plus public `.h` headers and disables all other features.
The old fork's KVarN binding includes five public Xe2 `.h` files, all covered by
the existing projection. Their dependencies are Torch headers, not filtered
private `.hpp` implementations. No new upstream binding include or source path
requires projection expansion.

Xe3p was already enabled by upstream defaults before this delta. At target,
`CMakeLists.txt:274–275` would append CRI devices to an explicit general AOT
selection, and `:378–385` would switch the common SYCL target to CRI.
The split factory explicitly passes `-DVLLM_XPU_ENABLE_XE3P=OFF`; both glue
components set the same option false. Keep those guards. The existing setup.py
forwards that option correctly. All split/glue derivations export both general
and Xe2 AOT lists from the same `aotDevices`; `[]` remains JIT and `[ "bmg" ]`
remains BMG only. No new Xe3 split target or import override is needed.

Dependencies: P04 prevents setuptools' secondary build/copy of imported DSOs;
P06 forwards the disabled MHC flag for the FA2 projection. Those independent
audits remain necessary. P01 itself does not change native operator schemas.

## Checks and required follow-through

- `patch --dry-run --fuzz=0 -p1` against an archive of the exact target passed:
  root CMake hunks had +22-line offsets; setup.py had +7. Application in the
  temporary archive also passed without fuzz.
- An isolated CMake 4.3.4 parser/configure harness used the actual patched
  library-selection block and target's AOT block. All six combinations passed:
  all libraries/base glue/FA2-only, each with empty or `bmg` AOT. It checked
  expected imported targets and locations, absence of disabled targets, absence
  of a Xe3 target, and no AOT list expansion. Harness location during the audit:
  `/tmp/v031-p01.vbI8Mn/check.py`. An initial harness assertion treated unset
  empty CMake variables as literal names; quoting those variables corrected the
  harness before the six passing runs.
- Source/include review covered the frozen upstream target and the old fork's
  public KVarN headers. No shared source worktree, pin, branch, or service was
  mutated; the temporary parser harness did not compile native code.

**Required patch changes: none.** Preserve Xe3p guards and existing projection
rules. Operator should verify the final replay did not introduce new binding
includes, evaluate representative split/glue derivations, and run the paired
native build/import and server qualification. This bounded parser check is not
evidence that Torch 2.14/native linking or runtime dispatch has passed.
