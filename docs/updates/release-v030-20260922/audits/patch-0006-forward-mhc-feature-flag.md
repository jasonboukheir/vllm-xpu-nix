# Patch 0006: forward the MHC feature flag

**Disposition: KEEP unchanged.** Upstream v0.1.15 still omits
`MHC_KERNELS_ENABLED` from `setup.py`'s CMake option forwarding. The existing
one-line patch is the minimum fix needed by the split FA2 binding build.
Final-source revalidation and native qualification remain pending.

## Frozen inputs and scope

| Input | Identity |
| --- | --- |
| Run | `release-v030-20260922` |
| Packaging baseline | `4ac4375aa3d0f649fb4b6b43d9447f413a8b1cc7` |
| Assigned patch | `nix/patches/0006-forward-mhc-feature-flag.patch` |
| Patch introduction | `ec8c035b8deda3606e2c4960a416f64a8584dd32`, `packaging: split reusable kernel glue from FA2 binding` |
| Kernel old upstream base | `e7f20cbc18d741419a84aa57178e1f1cc0b453da` |
| Kernel released tip | `2c39287deec5e59cc2656e6d4a6dce51a567c411` |
| Kernel selected release | `v0.1.15`, `1c7cbeee1cd0c1481d48f5031f679a47f3f0ef45` |
| Paired vLLM selected release | `v0.30.0`, `ced6857afa0ea7b2e3f0846a62e1394e90f15607` |
| Kernel object repository | `build-dev/release-v030-20260922/vllm-xpu-kernels` |

Read the project update skill, commit-audit contract, execution-host guidance,
kernel `AGENTS.md`, run manifest, relevant packaging sources, and patch 0001's
independent report. Host check returned `brutus`. All kernel implementation
reads used immutable `git show SHA:path` or other object-based Git queries;
the parent's changing checkout was not used as source evidence. This report
is the only written artifact. No source, ref, lock, service, issue, build, or
network operation was performed.

The release retains eager V1, K4V2 target/draft MTP and vision. This packaging
fix does not depend on the paused issue17/DFlash work.

## Requirement and failure path

The packaging commit introducing 0006 also introduced separate reusable base
glue and FA2 binding derivations. Both still configure through `setup.py`:
`nix/vllm-xpu-kernels.nix` sets `dontUseCmakeConfigure = true` and converts
`featureOptions` booleans into `ON`/`OFF` environment values.

In `nix/mk-kernels.nix`, base glue enables MHC and supplies the prebuilt MHC
library. FA2 binding sets `MHC_KERNELS_ENABLED = false` and
`withMhcLibrary = false`, while enabling SYCL-TLA and Xe2 for attention.
`nix/lib/kernel-glue-src.nix` retains the common build metadata and attention
bindings for FA2 but excludes the entire `csrc/xpu/mhc` subtree.

Without 0006, that FA2 environment disables Python's additional-library
installation entry for MHC, but never sends `-DMHC_KERNELS_ENABLED=OFF` to
CMake. CMake's MHC option defaults to ON. With SYCL-TLA and Xe2 enabled,
CMake enters the MHC library branch; patch 0001 sees no prebuilt MHC override
and falls back to `add_subdirectory(csrc/xpu/mhc/xe_2)`. That directory is
absent from the FA2 source projection. This is a configure-time failure path
inferred from the inspected control flow, not a native configure failure
reproduced during this bounded audit.

The requirement is therefore correct feature isolation and a buildable narrow
FA2 projection, supporting reuse of the expensive base glue when attention
changes. It applies to the generic package and Brutus's `aotDevices = [ "bmg" ]`
override. No GPU-specific implementation, operator schema, runtime ABI, or
model feature is changed by 0006. No new throughput benefit is claimed.

## Upstream coverage and alternatives

At frozen v0.1.15:

- `setup.py:185–202` explicitly forwards `_kernel_options`, but its list omits
  MHC. `_is_enabled` at lines 78–81 uses `os.environ` directly, defaults to ON,
  and recognizes `0`, `OFF`, `FALSE`, and `NO` after stripping/uppercasing.
- `CMakeLists.txt:76–77` defines MHC with default ON; lines 462–465 build its
  Xe2 library, lines 518–520 set `VLLM_MHC_ENABLED`, and the `_xpu_C` link list
  includes `MHC_LIB_NAME`. Those CMake branches do not read the MHC environment
  variable directly.
- `setup.py:568–570` already tests the MHC environment flag while selecting
  `additional_libraries`. This controls installation bookkeeping only; it does
  not configure CMake and does not cover the missing forwarding requirement.
- `tools/envs.py` supplies neither an MHC forwarding mechanism nor a generic
  CMake-argument escape hatch. Object-based search of `setup.py`, `tools`,
  `cmake`, and the relevant CMake file found no alternative MHC environment
  forwarding or `CMAKE_ARGS` mechanism.
- Direct CMake invocation can accept `-DMHC_KERNELS_ENABLED=OFF`, and the six
  standalone library derivations already supply direct CMake flags. These
  bypass `setup.py` and do not solve the Python package's configure path.
  Upstream's precompiled-wheel path skips native builds altogether; it is not
  the Nix split-library composition used here.

`setup.py` is byte-identical at old upstream base, released fork tip, and
selected target. `git log` and `git diff` over the complete old-base-to-released
range show no fork changes to `setup.py`, `CMakeLists.txt`, or `tools/envs.py`.
The old-base-to-target diff leaves setup/envs unchanged and only adds
`csrc/xpu/sampler/compact_sampling_mask.cpp` to CMake's source list, in
`4111ba268535372314a86f1a2dc930524ba7a92c`. The relevant feature option and
dispatch paths remain unchanged. Neither upstream nor a later local commit
supersedes 0006.

## Package sequence and dependencies

The package applies **0001 → 0004 → 0006**. Patch 0001 forwards prebuilt
library paths and provides CMake imported targets, including MHC. Patch 0004
skips the installation of intermediates supplied as prebuilt libraries.
Neither changes `_kernel_options`. Patch 0006 adds MHC to that existing list,
aligning configure behavior with the already-present Python install predicate.

For base glue, the patched setup sends MHC ON and forwards the prebuilt MHC
path; patch 0004 skips installing its nonexistent local intermediate build.
For FA2, it sends MHC OFF and does not supply a prebuilt MHC path; neither the
CMake branch nor Python's MHC install entry is selected. The source projections
retain `setup.py`, so the patch target exists in both components.

The standalone library sequence is 0001 → 0005 and needs no 0006 because it
configures CMake directly. Its flag matrices enable MHC only for the MHC
library and explicitly disable it for the other five libraries.

No paired vLLM change or operator deletion is required. Keep the one-line
residual; removing MHC from the package or broadening the FA2 source projection
would change the existing packaging contract without addressing the forwarding
bug. Changes to 0001/0004 remain owned by their separate audits.

## Observed validation

The parent's exact-tag disposable preflight records successful `--fuzz=0`
application of the complete package sequence. The 0006 hunk applies without
an offset. Evidence file:
`benchmark-results/release-v030-20260922/upstream-patch-preflight.json`.

A separate read-only check used
`/home/jasonbk/Projects/vllm-xpu-kernels/.venv/bin/python -B` (Python 3.12).
It loaded the immutable target setup with `git show`, applied each package
patch's setup hunks entirely in memory, and required each complete old hunk
context to match exactly once. It parsed the actual base/FA2 `featureOptions`
blocks from `nix/mk-kernels.nix`, then executed only the setup AST nodes for
`_is_enabled`, option forwarding, and additional-library selection. It did not
import setup, torch, or setuptools, configure CMake, or compile anything.

| Case | 0001 + 0004 | 0001 + 0004 + 0006 | Result |
| --- | --- | --- | --- |
| Base glue MHC ON | No MHC CMake argument | Exactly `-DMHC_KERNELS_ENABLED=ON` | PASS; MHC remains in Python's library list and its prebuilt path is forwarded |
| FA2 MHC OFF | No MHC CMake argument | Exactly `-DMHC_KERNELS_ENABLED=OFF` | PASS; Python's library list contains only attention, with no prebuilt MHC path |
| Unset MHC | — | ON | PASS; preserves default behavior |
| `OFF`, `0`, `FALSE`, `NO`, ` off ` | — | OFF | PASS, all five cases |
| `ON`, `1`, `TRUE`, `YES` | — | ON | PASS, all four cases |

For each component, the new MHC argument was the only difference in forwarded
arguments and the additional-library mapping was unchanged. Thus the check
distinguishes the missing behavior in upstream plus 0001/0004 from the fix.
Source projection conclusions above come from reading the maintained filters;
this agent did not independently perform Nix evaluation. The independent 0001
report records successful offline Nix evaluation of the actual factory and
source projections, including the Brutus overrides.

## Evidence identities for final replay

| Input | SHA256 |
| --- | --- |
| Patch 0006 | `8f2cba38bf161f7d557f3f1fce0b584c6beda21cb315a4b17ee39a4c5411de98` |
| Patch 0001 | `0d7b31cef2a1855a67910023d6c35e0a91345376dfd2e82a4f663d55d7809ead` |
| Patch 0004 | `45c1e47d947d2f576054096278825f0d8fe0ad4787b9536afcc3af73061ad873` |
| `nix/mk-kernels.nix` | `bf6b026722e66168b9ad7dd66ae12b86b736f9da9b1ae9067c86d962759c286d` |
| `nix/vllm-xpu-kernels.nix` | `a04700b23e39b9531c97b2ea9c6d25b856801ee8299e861330bed2451b303647` |
| `nix/lib/kernel-glue-src.nix` | `1d745679df67d1e4de877dae6be6eda73ac6797606cd09c07e27f1c1d85cd40b` |
| `nix/lib/kernels-src.nix` | `b0e9192497b375fe93fef2e67b56ab54723518831bcc427528706aaa12371457` |
| Parent preflight JSON | `322d634a5e36f8fc2494ca7d902c0d90c6ece94e96dd58f323e95fa6b9674d09` |
| Setup after 0001 + 0004 | `9e868521d6ae8fab0cd1c68ba70773e9a90f58f2f2b8e8d4d0677af88c5081b2` |
| Setup after 0001 + 0004 + 0006 | `685a84d7af9046d75cd8ce08b68bd78ea802464613fe9b3838fc14c190ff21d6` |

| Frozen upstream file | Git blob | SHA256 |
| --- | --- | --- |
| `setup.py` (patch target) | `e841c754d353b39fb566e7c63b31ab70a845dab0` | `153b2696834b8161a362870080c59077810b977b36127ba57e472ed51048b4ca` |
| `CMakeLists.txt` (feature consumer) | `f7041ef56fc006ed7adeb4432f0a75d3ada7f191` | `5793bac065c0da82323bf53108adcd096f7cd3d40f6cd379552e4b61f3943e79` |
| `tools/envs.py` (alternate-path check) | `898cc3a7299b7284a6df4103fc81e69ea3d5c76f` | `fcc73cd0693783360259b2bd7b2f21773cfa4e97354598ce22ca447483bc75a6` |

## Final-source and build gates

1. Freeze the final replayed kernel SHA and compare the three upstream file
   blobs above, patch hashes, and packaging inputs. Rerun the complete package
   sequence with `--fuzz=0` against that final source; verify its patched setup
   hash or investigate any difference. This report does not attest an unknown
   final replayed commit.
2. Revalidate the generated Nix feature environments and projections for the
   final source with actual Brutus overrides. Base must send MHC ON with its
   prebuilt library, while FA2 must send MHC OFF without its library or source
   subtree. Revalidate changed call paths even if 0006 still applies cleanly.
3. During authorized candidate builds, inspect the FA2 CMake cache/configure
   output for `MHC_KERNELS_ENABLED=OFF` and absence of a local MHC target. Check
   base glue has MHC ON with its imported library and correct installed ELF
   dependencies. Complete the existing composed-package extension-count,
   attention dependency/RPATH, donor-reference, and Python import checks.
4. Complete the release's paired native build and GPU qualification for eager
   V1, K4V2 target/draft MTP and vision. No production restart or extra MHC GPU
   benchmark is justified merely by this forwarding fix; this bounded audit
   does not claim candidate build, runtime, or release qualification.
