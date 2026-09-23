# Separate v0.30 release: native build passed, GPU qualification pending

The user authorized the ordinary release workflow on latest main, preserving
existing eager V1 K4V2 target/draft bundled MTP2 and vision. Issue #17 remains
paused and unrelated. Its preservation-only [draft PR #18](https://git.sunnycareboo.com/jasonbk/vllm-xpu-nix/pulls/18)
will not be merged into this release. No V2/DFlash investigation belongs here.

## Verified checkpoint

- All 27 independent source commit audits (16 vLLM, 11 kernels), parent
  reconciliation and four independent packaging patch audits are complete.
- Minimal replay, applicable source/CPU checks, hooks, paired interfaces,
  ancestry, effective-diff and range-diff review passed. Five vLLM and eight
  kernel commits remain; see `source-review.md` and `commit-map.json`.
- Rolling source mains are published and verified, with recovery branches:
  vLLM `466b9d5abe84f13c29b38c686e4d77fc09e09aa4` on v0.30.0
  `ced6857afa0ea7b2e3f0846a62e1394e90f15607`; kernels
  `193cee080d654ff67c4f8bfaf924201bc88b9cf5` on v0.1.15
  `1c7cbeee1cd0c1481d48f5031f679a47f3f0ef45`.
- Packaging branch `release-v030-20260922` pins that pair. Required Hub change
  is 1.31.0; unrelated locks, Torch/Triton/oneAPI and XGrammar are unchanged.
- Exact Brutus native build passed in 1857.1 seconds with max-jobs=1/cores=4.
  All four extension modules and six split libraries load. Eleven KVarN/GDN
  schemas match the source and have XPU dispatch registered.
- All **608 packaging unit tests across 45 files passed**, with no skips or
  failures. The forced-decode tests used the new Brutus Python closure.
  Generic/Brutus evaluation and four lightweight packaging checks also passed.

`native-build-review.json` records the tested packaging commit, exact outputs,
resource observations and evidence hashes. The package is
`/nix/store/h3phd3ijav9zp34kzbvhyicxqsbanr94-python3.12-vllm-xpu-0.30.0+unstable.2026.09.23.g466b9d5`;
its test runtime is
`/nix/store/zhv84h5afvyx99fkw0l56rpj8rldhivl-v030-brutus-qualification-env`.
Source clones under `build-dev/release-v030-20260922/` remain clean.

## Maintenance and remaining work

The user initially stopped chat and confirmed use of the downtime. The build
started with no service process. Chat then started again at **2026-09-22
22:15:57 PDT**, MainPID 2761986, NRestarts=0. It remains active. This run neither
started nor stopped it, and requested clarification before stopping the new
instance. Embedding is absent/inactive. No host repin or activation occurred.

No release-owned background build or GPU test remains running. GPU correctness,
serving/MTP/vision, lifecycle, capacity and performance checks have **not run**.
The controlled k02/k04/k10/k11 and patch0005 comparisons are prepared but
unmeasured. Native import and CPU test passes do not satisfy those gates.
See `qualification-status.json` and `resume.md` for the remaining sequence.

No new stable release is published. **xpu-v1.10.0 remains the operational
baseline**, packaging `4ac4375aa3d0f649fb4b6b43d9447f413a8b1cc7`.
No DFlash speedup or performance ceiling is established. The remaining handoff
is maintenance clarification, not a measured performance limit.

Private artifacts are preserved under
`benchmark-results/release-v030-20260922/`. The packaging checkout also retains
two untracked issue17 report directories; do not add them to this release.
Host `~/.config/nix` has pre-existing root/server `flake.lock` changes; preserve
them. Host repinning and activation were not requested.
