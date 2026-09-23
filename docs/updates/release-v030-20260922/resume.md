# Completed release checkpoint and separate host handoff

The ordinary v0.30 release is complete. Do not repeat audits, source replay,
GPU qualification or optional experiments. Read `CURRENT.md`,
`release-acceptance.json` and `publication-verification.json`.

Published immutable xpu-v1.11.0 branches and annotated tags:
- Packaging: `93c021f9433e132e4f594d6c64353777f391dcf2`.
- vLLM: `466b9d5abe84f13c29b38c686e4d77fc09e09aa4`.
- Kernels: `193cee080d654ff67c4f8bfaf924201bc88b9cf5`.

The remote packaging manifest evaluates to the exact qualified Brutus derivation:
`/nix/store/cjr2ccv93h5cqhvh49nkxqrkp8y4dp90-python3.12-vllm-xpu-0.30.0+unstable.2026.09.23.g466b9d5.drv`.
The source clones in `build-dev/release-v030-20260922/` are clean. Recovery branches
and original audit reports are preserved. Never retarget an immutable release.

The user subsequently requested `~/.config/nix` follow updated packaging main.
Inspect that checkout's actual state before resuming any handoff; update only
`vllm-xpu-release` and intended transitive nodes in the server partition. Preserve
its documented anonymous HTTPS source overrides, selecting xpu-v1.11.0 source
tags so they match the packaging lock. Verify the package derivation and Brutus
system evaluation. Commit the repin; do not activate or restart services without
the separate operator handoff. Runtime baseline/rollback is xpu-v1.10.0, packaging
`4ac4375aa3d0f649fb4b6b43d9447f413a8b1cc7`, until activation.

Issue #17 remains PAUSED. Its checkpoint branches and draft PR #18 are unrelated;
do not merge them, change its goal status or start V2/DFlash work. No DFlash
speedup or performance ceiling is established. Prepared unrun controls and all
raw measurements remain under `benchmark-results/release-v030-20260922/`.
No qualification helpers remain active. Preserve user-stopped services and the
two untracked issue17 report directories. Original host dirty-lock observations
are historical; reread current status instead of assuming they still apply.
