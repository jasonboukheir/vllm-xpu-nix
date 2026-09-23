# Separate v0.30 release: packaging prepared, native qualification pending

The user authorized the ordinary release workflow on latest main, preserving
existing eager V1 K4V2 target/draft bundled MTP2 and vision. Issue #17 remains
paused and unrelated. Its preservation-only [draft PR #18](https://git.sunnycareboo.com/jasonbk/vllm-xpu-nix/pulls/18)
will not be merged as part of this release. No V2/DFlash feature or performance
investigation belongs to this run.

## Completed and preserved

- All 27 independent source commit audits (16 vLLM, 11 kernels), parent
  reconciliation and four independent packaging patch audits are complete.
- Minimal source replay, CPU regression tests, applicable hooks, paired schema,
  ancestry, effective-diff and range-diff review passed. Five vLLM and eight
  kernel commits remain. See `source-review.md` and `commit-map.json`.
- Both rolling source mains are published and verified, with both remote
  recovery branches intact. vLLM: `466b9d5abe84f13c29b38c686e4d77fc09e09aa4` on upstream
  v0.30.0 `ced6857afa0ea7b2e3f0846a62e1394e90f15607`; kernels:
  `193cee080d654ff67c4f8bfaf924201bc88b9cf5` on v0.1.15
  `1c7cbeee1cd0c1481d48f5031f679a47f3f0ef45`.
- Local packaging branch `release-v030-20260922` pins that exact pair. Hub
  advances to 1.31.0; all unrelated lock nodes, Torch/Triton/oneAPI and XGrammar
  remain unchanged. Generic and actual Brutus evaluation, final patch/projection/
  AOT checks, flake evaluation and four lightweight packaging checks pass.

## Next handoff

The candidate has not been built or qualified natively. No new stable release
has been published. Operational baseline remains **xpu-v1.10.0**, packaging
`4ac4375aa3d0f649fb4b6b43d9447f413a8b1cc7`. No service changes or host
activation were performed. Chat is enabled but now inactive after a clean stop
at 2026-09-22 21:01:56 PDT; embedding is absent/inactive. This run did not stop
chat. Maintenance authorization has not been recorded; confirm operator intent
at the runbook handoff and preserve the actual prior service state.

Read `resume.md` for exact build/qualification instructions and
`qualification-status.json` for open gates. Provisional kernel performance
changes still require controlled evidence. No DFlash speedup or ceiling is
established by this work. Source-only tests are not native qualification.

Source clones: `build-dev/release-v030-20260922/{vllm,vllm-xpu-kernels}`, both
clean on `release-v030-20260922-replay`. Raw artifacts are private and ignored
under `benchmark-results/release-v030-20260922/`; reviewed evidence is here.
Host `~/.config/nix` has pre-existing changes to root and server `flake.lock`;
preserve them. Host repinning/activation was not requested.
