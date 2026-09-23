# Resume the separate v0.30 release workflow

Read `CURRENT.md`, `manifest.json`, `status.json`, `commit-map.json`,
`source-review.md`, `packaging-checks.json` and `qualification-status.json`.
The ordinary v0.30 release is authorized. Issue #17 remains PAUSED; do not
resume its V2/DFlash work, merge checkpoint PR #18, or change its goal status.

## Exact preserved state

- Source mains are published and verified: vLLM `466b9d5abe84f13c29b38c686e4d77fc09e09aa4`;
  kernels `193cee080d654ff67c4f8bfaf924201bc88b9cf5`. Clean isolated clones are under
  `build-dev/release-v030-20260922/`, branch `release-v030-20260922-replay`.
- Packaging candidate branch: `release-v030-20260922`. It starts from the
  published xpu-v1.10.0 baseline `4ac4375aa3d0f649fb4b6b43d9447f413a8b1cc7`.
  Use `git rev-parse release-v030-20260922` for its preserved local commit.
  Packaging main and the stable release remain at the baseline.
- All three issue17 checkpoint branches are
  `checkpoint/issue17-paused-20260920`. Packaging tip
  `78a30c20abacc2455d8f36991a9783e251d60ace` preserves former local
  `880e74ed488a795963e631b46be5295a9389c323`; [draft PR #18](https://git.sunnycareboo.com/jasonbk/vllm-xpu-nix/pulls/18)
  tracks that paused work only. Do not merge or replay it into this release.
- Both source remotes retain `backup/main-pre-v030-20260922` and
  `backup/origin-main-pre-v030-20260922`; original tips are in the manifest.
  Do not repeat the already completed audit/rebase or move old stable refs.

## In-flight work (read before starting anything)

The native build is running. Inspect this run's private `native-build-status.json`,
`native-build.log` and `native-build-resources.jsonl`; verify PID start ticks
against `owned-build-processes.json`. Do not launch a duplicate. Read current
user messages: chat unexpectedly restarted at 22:15:57 PDT, and clarification
was requested before stopping the new instance or starting GPU qualification.
Initial maintenance authorization persists, but do not override an intentional
operator restart. The initial build itself continues with max-jobs=1/cores=4.

`qualification-packages.nix` prepares the actual Brutus test runtime and isolated
no-k02/no-k10/no-k11 controls; evaluations pass. Only the attention source projection
changes in each arm, with FA2 relinked; other native sources/derivations match.
`controlled-source-variants.json` and `controlled-component-equivalence.json`
record the exact differences. They have not been built or measured.
`qualification-protocol.json` freezes native ABBA and matched release comparisons.
The copied API/native harnesses and old serving workload inputs are retained here;
check their provenance before execution. Use the current final source tests and
installed candidate modules; never count an old installed binary as a new-pair pass.

## Remaining sequence

1. Recheck actual refs, worktree/lock identities and `hostname -s` = `brutus`.
   The 27 source audits, four patch audits, parent reconciliation/replay, source
   publication, package evaluation and four lightweight checks are complete.
   Native build and GPU qualification are not complete.
2. Maintenance is confirmed: the user said "yes I stopped it" when asked to
   use the downtime for the native build and GPU qualification. Chat has no
   running main process (failed state, MainPID=0); embedding is absent/inactive.
   This run stopped neither. Recheck current state, preserve any operator
   changes, and leave user-stopped workers stopped. Do not ask again for the
   same maintenance authorization or change host activation.
3. With maintenance confirmed, build the exact Brutus
   override below with bounded jobs. Record actual compile concurrency/peak
   memory and perform the audit-required controlled build checks. Preserve
   artifacts when fixing a failure. Full packaging unit-tests require this
   native closure and are still pending.
4. Run the built package in the foreground using Brutus's generated profile
   (`nix run ~/.config/nix#vllm-xpu-brutus -- "$candidate_package" --port 18000`).
   Retain eager V1 K4V2 target/draft MTP2, vision, 262144 context, four slots,
   2048 batched prefill and two 448px images. Model:
   `RedHatAI/Qwen3.8-27B-INT4`, revision
   `bf08f3dbd9a324e53956920aad378a1f1b6dd24a`. Record actual launch/environment,
   imports, native schemas and runtime source identities before interpreting tests.
5. Complete `qualification-status.json` and each applicable audit's gates:
   independent numerical references/native-only dispatch, GDN controls,
   finite-storage/padding diagnostics, target/draft lifecycle/cancel/reuse,
   concurrent MTP, vision/tools, long-context retrieval, capacity and matched
   xpu-v1.10.0 throughput/latency/memory comparisons. Measure exact-metadata
   synchronization cost. Keep historical semantic/NaN/concurrent divergence
   limitations explicit. Source-only passes do not replace installed tests.
6. Resolve the provisional k02 cold compile/RSS and k10/k11 controlled ABBA
   gates (identical companion optimization, native proof, numerical reference,
   repeatability, K4V4 anomalies), plus patch0005 resource observations.
   If a repair is needed, fold it into its source owner, preserve published
   recovery tips, use fresh exact leases, update maps/pins and rerun affected
   checks. Do not append temporary repairs to the minimized final stacks.
7. Only after qualification, recheck an unused coordinated minor version
   (`xpu-v1.11.0` is proposed, not reserved), create new immutable source refs,
   lock packaging to them, prove exact package equivalence, then publish
   packaging main and coordinated refs with reviewed release notes. Verify all
   remote peeled tags and release branches. Restore only workers this run
   stopped unless the operator directs otherwise. No host repin or activation
   has been requested.

## Exact package build (after maintenance handoff)

```bash
cd /home/jasonbk/Projects/vllm-xpu-nix
candidate_package=$(nix build --impure --no-link --print-out-paths \
  --max-jobs 1 --cores 4 --expr '
    let
      candidate = builtins.getFlake "git+file:///home/jasonbk/Projects/vllm-xpu-nix";
    in import /home/jasonbk/.config/nix/hosts/brutus/services/vllm-xpu/package.nix {
      vllm-xpu-unstable = candidate.packages.x86_64-linux.vllm-xpu-unstable;
    }
  ')
```

Expected Brutus derivation:
`/nix/store/cjr2ccv93h5cqhvh49nkxqrkp8y4dp90-python3.12-vllm-xpu-0.30.0+unstable.2026.09.23.g466b9d5.drv`.
Re-evaluate after the packaging commit and verify equality before building.
Use the Git flake URL; do not copy ignored private artifacts using a raw path
flake. Raw evaluation command/data are in this run's
`evaluate-candidate.nix`, `packaging-evaluation.json` and
`packaging-final-checks.json`. Final patch preflight and remote-ref records are
in the same private artifact directory.

## Local tooling and artifacts

Both isolated source clones have fresh `.venv` environments, created through
uv with Nix Python 3.12.13, and installed pre-commit 4.6.2. Repository hooks are
installed. For NixOS, `.git/release-pre-commit-config.yaml` preserves hook pins
while using Nix Node (`language_version: system`) and Bash for script hooks.
Use Node `/nix/store/glcp73hgagq2b24i80jlgbvj28vdb6kk-nodejs-24.19.0/bin`
on PATH and `PRE_COMMIT_HOME` pointing to this run's own
`build-dev/release-v030-20260922/pre-commit-cache`.

Both hook environments installed successfully. Generic wheel ELF executables
initially hit NixOS's missing-loader stub; the new cache's executables were
adapted to Nix glibc/GCC. Exact paths and successful version probes are retained
in `benchmark-results/release-v030-20260922/lint-elf-adaptations.json`.
Applicable hooks subsequently passed; exact results are in `source-review.md`.
Kernel upstream mypy is a no-op, not a type-check result. Do not modify the paused
goal's tool cache or raw artifacts.

Raw logs, commands, dependency evidence and status mirrors belong under
`benchmark-results/release-v030-20260922/` (private, ignored). Reviewed reports
belong under this documentation directory. Stage intended paths explicitly;
the packaging checkout also contains preserved untracked issue17 records that
must not be accidentally added to the unrelated release.

`~/.config/nix` has pre-existing modifications to `flake.lock` and
`modules/flake/nixos/server/flake.lock`. Preserve them. The actual consumer is
the server flake, and Brutus overrides are in
`~/.config/nix/hosts/brutus/services/vllm-xpu/package.nix`.
