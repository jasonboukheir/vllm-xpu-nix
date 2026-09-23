# Issue #17 — PAUSED AT USER REQUEST

**Resume only on explicit user request.** The supported goal control is paused.
Do not continue implementation, rebasing, optimization, qualification or
publication. No DFlash2 speedup or performance ceiling has been established.
The published/deployed operational baseline remains **xpu-v1.10.0**.

Read [PAUSED.md](/home/jasonbk/Projects/vllm-xpu-nix/docs/updates/issue17-20260920-snapshot-audit/PAUSED.md) for the complete checkpoint and waiting criteria,
[RESUME.md](/home/jasonbk/Projects/vllm-xpu-nix/docs/updates/issue17-20260920-snapshot-audit/RESUME.md) for the future resume procedure, and the synchronized
[issue body](/home/jasonbk/Projects/vllm-xpu-nix/docs/updates/issue17-20260920-snapshot-audit/issue17.md). The live issue is
https://git.sunnycareboo.com/jasonbk/vllm-xpu-nix/issues/17.
The original body is preserved as `issue17-before-pause.md`/`issue17.json`.

## Actual completed work

- 27 distinct individual commit audits (16 vLLM, 11 kernels), including release-only
  fixes; all report hashes and parent reconciliation verified. KEEP/DROP/ADAPT
  decisions remain candidate decisions for the historical September pair.
- Recovery/snapshot refs preserved locally and remote recovery branches verified.
- One kernel replay only: k01 -> `c66a48486dcd6b76336ec8a8b984a4700f89e1d9`;
  hooks passed, native build/GPU qualification not run. No vLLM replay.
- Packaging dependency commit `880e74ed488a795963e631b46be5295a9389c323`
  (local/unpublished): Hub 1.31.0 and XGrammar 0.2.7 built; offline API/JSON
  grammar smoke and formatting checks passed. No final candidate stack build.
- Node24 realization completed. Kernel hooks worked; vLLM hook setup remains
  incomplete. Preserve failure logs and ignored caches; no repair while paused.

## Exact checkout checkpoint

| Checkout | Branch / HEAD | State |
| --- | --- | --- |
| Packaging | main / `880e74ed488a795963e631b46be5295a9389c323` | Only untracked `docs/updates/`; no tracked/staged changes |
| `build-dev/issue17/vllm` | issue17-20260920-replay / `e378275a8f20eac9e92212e4a1fee84ec5a31dc7` | Clean; zero replay commits |
| `build-dev/issue17/vllm-xpu-kernels` | issue17-20260920-replay / `c66a48486dcd6b76336ec8a8b984a4700f89e1d9` | Clean; one replay above `da16a5595c105605bcd44b558c7933b934a05f85` |
| `../vllm` | main / `3b6c822e8a4f56dc3904c7fce388a76dff4c7bdd` | Clean; unchanged |
| `../vllm-xpu-kernels` | main / `9bb1e2d428b8a25b262c0cd6704614bdf6ee3ad8` | Clean; unchanged |

No unfinished Git operations. All audit reports, commits, recovery refs and
artifacts remain. `pause-checkpoint.json` contains exact local/remote refs,
report hashes, service identity and the inspection timestamp. The final source
map and minimized effective diff are unfinished. Integration, profiling,
performance work, quant comparisons and final qualification/publication have
not started.

## Future reassessment and scope

A published upstream **v0.32-series** release is the planned reassessment and
candidate-rebase point, with no reserved patch version or readiness guarantee.
v0.32 targets V1 removal, not stabilization of this configuration. Recheck the
five upstream conditions in PAUSED.md and every audit's applicability at explicit
resume; do not blindly replay September snapshots. Waiting does not solve exact
committed history, target/draft ownership/MTP prefill, D128 noncausal sliding
K4V2, concurrent draft-window budgeting or lifecycle/reset correctness.

The proposed split (V2 existing K4V2 MTP+vision first, quantized DFlash2/performance
second) awaits user approval. Neither phase is started or authorized. Original
end-to-end obligations remain preserved in the issue, including profiling first,
MTP speed/capacity comparisons, two adversarial stopping reviews and publication.

## Host and artifacts

At 2026-09-20 07:38 UTC, chat was **active/running, PID 1638887**, using the
released `v8jzs75...g2fdd9d7` runtime, MTP2 and K4V2. The earlier inactive
observation is superseded. No goal-owned background work remained to stop; no
service or activation was changed. Preserve the separate GPU-maintenance and
host-activation handoff; recheck actual ownership before any future GPU work.

Records: `docs/updates/issue17-20260920-snapshot-audit/` (this directory).
Ignored evidence: `benchmark-results/issue17-20260920/`, including synchronized
CURRENT/RESUME/status, pause inspection, artifact hashes, pre-edit archive,
dependency and tooling logs, and issue-body API verification. Source clones,
venvs and hook adapters: `build-dev/issue17/`. Do not clean or delete them.

Issue body updated inline and verified by fresh API read at 2026-09-20T07:44:59.201285+00:00.
Exact body SHA256: `4e80324a899a4e9b8483108ba23c59b40eb618383f41fa2e48b66d2b9fd31f3e`.
Supported goal control reverified paused; source/worktree/host checks unchanged.
