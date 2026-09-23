# Resume ordinary v0.30 release publication

Read `CURRENT.md`, `release-acceptance.json`, `manifest.json`, `status.json` and
`qualification-status.json`. Actual Brutus settings and reasonable-output checks
are the user-selected acceptance scope and have passed. Do not resume optional
experiments, source replay or GPU qualification just because historical audit
reports recommended them. Preserve unmeasured claims and observed limitations.

Issue #17 remains PAUSED. Its checkpoint branches and draft PR #18 are unrelated;
do not merge them, change its goal status or start V2/DFlash work.

1. Inspect actual refs against `benchmark-results/release-v030-20260922/publication-preflight.json`.
   Selected release: xpu-v1.11.0. Use the exact source heads in the manifest and
   clean clones at `build-dev/release-v030-20260922/{vllm,vllm-xpu-kernels}`.
   Preserve all `refs/recovery/release-v030-20260922/*` and remote backup branches.
2. Publish missing annotated source tags and release branches only after verifying
   existing refs match the recorded tested heads. Never retarget an existing ref.
3. Change both unstable source input refs to `refs/tags/xpu-v1.11.0` and update only
   those two locks. Rev/narHash/lastModified and unrelated locks must be unchanged.
   Verify exact Brutus package derivation using the preserved host override:
   `/nix/store/cjr2ccv93h5cqhvh49nkxqrkp8y4dp90-python3.12-vllm-xpu-0.30.0+unstable.2026.09.23.g466b9d5.drv`.
   Do not rebuild or requalify an identical derivation.
4. Commit intended packaging/docs files, excluding both untracked issue17 report
   directories. Publish packaging main normally plus new immutable release branch
   and annotated tag at that commit. Verify all remote branch and peeled tag SHAs.
   Update durable publication status and preserve verification evidence.
5. Leave user-stopped host services stopped. No host repin or activation is requested.
   xpu-v1.10.0 packaging `4ac4375aa3d0f649fb4b6b43d9447f413a8b1cc7` is rollback
   and the host operational baseline until separately authorized deployment.

All optional qualification processes have exited; diagnostic02 finished naturally.
No background build/timing work is active. Chat is failed with no PID/cgroup tasks.
Private evidence and prepared but unrun controls remain in
`benchmark-results/release-v030-20260922/`. No new performance benefit or ceiling
has been established. Host root/server lock changes predate this work; preserve them.
