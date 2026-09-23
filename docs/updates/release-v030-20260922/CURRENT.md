# Separate v0.30 release: Brutus foreground qualification passed; controls pending

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

The user explicitly authorized stopping the restarted chat instance for
qualification. The stop command required a sudo password, but the subsequent
check already found chat inactive/dead, MainPID/ControlPID=0, empty cgroup and no
tasks. Embedding is also inactive with no process. This run did not successfully
stop either service. Preserve the user-stopped state; no host repin/activation.

All 705 installed native/backend tests passed with no skips. Fourteen additional
GDN fresh-state independent-reference cases passed. The actual Brutus foreground
app passed text, reasoning, two-image vision, tools and 12 concurrent requests
with four running at once and active MTP draft/acceptance counters. Mixed-prefill,
cancellation/reuse and 131071/262015-token retrieval passed. Near-capacity checks
also passed: two 172415-token prompts, 512 generated tokens each, correct labels
and arithmetic, active MTP, and no preemptions. Both foreground instances exited.
The actual budget supplies 346880 usable attention tokens; processed prompts
covered 99.409% of that capacity. Owned DRM sampling had no errors or gaps over
0.55 seconds during these checks. Read `foreground-qualification-review.json`,
`foreground-memory-review.json` and `native-correctness-review.json`.

The retained NaN diagnostic reproduces six historical unused-BF16-NaN failures;
both sparse-writer controls pass. Read `nan-diagnostic-review.json`. The scoped
AEON check is running in `aeon-api-02/`; an earlier attempt stopped after readiness
because the harness's offline setting rewrote the canonical model ID, making the
existing narrow selector ineligible. Source was unchanged; the repeat restores
the preserved canonical-ID launch. Do not launch concurrent GPU work.

Controlled k02/k04/k10/k11 and patch0005 comparisons, metadata synchronization
measurement, and matched released/candidate performance and memory checks remain
pending. Prepared controls are not results. See
`qualification-status.json` and `resume.md`.

No new stable release is published. **xpu-v1.10.0 remains the operational
baseline**, packaging `4ac4375aa3d0f649fb4b6b43d9447f413a8b1cc7`.
No DFlash speedup or performance ceiling is established. GPU qualification is authorized and proceeding; no performance limit is claimed.

Private artifacts are preserved under
`benchmark-results/release-v030-20260922/`. The packaging checkout also retains
two untracked issue17 report directories; do not add them to this release.
Host `~/.config/nix` has pre-existing root/server `flake.lock` changes; preserve
them. Host repinning and activation were not requested.
