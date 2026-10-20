# Native build concurrency controls

Disposition: KEEP `nix/patches/0005-reduce-kernel-build-memory.patch`.

Selected kernel fork `dbb684f4c9b38a8b3bbf52bda0c490ee47049221` retains
the original fixed build controls. `nix/vllm-xpu-lib.nix` applies this patch
to expose device-link concurrency and template diagnostic-depth knobs for
the separate native libraries. It is not a patch applied by the base-glue
derivation. No source or toolchain change in the selected pair makes those
controls redundant.

All three CMake hunks applied in an isolated archive (30/36-line offsets).
This proves continued applicability and control plumbing, not a measured
memory or performance benefit. If native inputs change during V2 work, verify
actual compiler concurrency honors the requested lower core limit.
