# Effective patches outside the source forks

Packaging baseline: `4ac4375aa3d0f649fb4b6b43d9447f413a8b1cc7`.
Exact file hashes and dry-run command results are in `packaging-delta.json`.

| Packaging delta | Requirement and disposition |
| --- | --- |
| `0001-split-kernel-libs.patch` | Retain independent native DSOs and prebuilt-library linkage for Nix incremental builds. Upstream still builds the monolithic graph; both patch-set dry runs pass. |
| `0004-skip-prebuilt-additional-libs.patch` | Retain glue-wheel handling of externally supplied libraries; avoid duplicating the split DSOs in the wheel. |
| `0005-reduce-kernel-build-memory.patch` | Retain split-library compilation memory controls pending exact-package build validation. |
| `0006-forward-mhc-feature-flag.patch` | Retain propagation of the MHC build flag through setup.py. |
| `nix/vllm-xpu.nix` postPatch | Relax setuptools build bound, remove unused audio/video/AutoRound requirements, replace kernel wheel dependency with the paired Nix kernel package, and disable the unused Rust frontend. Recheck each replacement on final source; do not relax Torch/Triton merely to hide incompatibility. |
| Kernel source projections | Retain shared-source invalidation and split native/glue identities. New sampler and LayerNorm files are covered by the existing whole-csrc projection; final identity tests remain required. |
| Runtime packaging | Retain pinned Torch 2.13.0+xpu, Triton 3.7.2+xpu, PTI 0.17.0 and oneAPI 2026.0 unless selected-source build evidence requires a change. Preserve unrelated flake locks. |

No packaging patch has been dropped based on application success. Application
success establishes only textual compatibility. The release must still run
library/glue cache-identity, build-parallelism, formatting and unit-collection
checks, then qualify the exact realized source pair and loaded DSOs.
