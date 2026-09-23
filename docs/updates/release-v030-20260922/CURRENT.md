# xpu-v1.11.0 checkpoint

The ordinary v0.30 release is complete. Brutus's eager V1, K4V2 target/draft
MTP2 and vision profile passed the actual foreground checks. See the
[release notes](../../releases/xpu-v1.11.0.md),
[manifest](../../releases/xpu-v1.11.0.json),
[publication verification](publication-verification.json), and
[current fork delta](../../fork-delta.md).

| Repository | Released commit |
| --- | --- |
| Packaging | `93c021f9433e132e4f594d6c64353777f391dcf2` |
| vLLM | `466b9d5abe84f13c29b38c686e4d77fc09e09aa4` |
| XPU kernels | `193cee080d654ff67c4f8bfaf924201bc88b9cf5` |

All three annotated tags and release branches remain immutable. Full reports,
27 individual commit audits, four packaging audits, original commit maps,
qualification results and historical resume records are in the
[published archive at 8c72895](https://git.sunnycareboo.com/jasonbk/vllm-xpu-nix/src/commit/8c72895340deafef77998a579335d078800626ab/docs/updates/release-v030-20260922).
The release manifest's hashed evidence uses its original xpu-v1.11.0 snapshot.
To read an archived file locally, use:

```sh
git show 8c72895340deafef77998a579335d078800626ab:docs/updates/release-v030-20260922/commit-map.json
```

The user requested the host follow packaging main. Host-config commit
`0f0d46f83078404a3285325dc279b3b5fed277e1` locks packaging `8c72895` and the
same source pair over HTTPS. Its Brutus package exactly matches the qualified
release and full system evaluation passed. The user activated Brutus on
September 23. The system service started at 00:25:28 PDT and its HTTP API became
ready at 00:29:06. Post-activation chat, reasoning, two-image vision and tool-call
checks passed on the exact qualified package, with active MTP, 346880 usable
attention tokens and zero restarts. Evidence is retained in
`benchmark-results/post-activation-20260923/`. Rollback is xpu-v1.10.0, packaging
`4ac4375aa3d0f649fb4b6b43d9447f413a8b1cc7`.

Issue #17 remains paused. Its complete reports and exact resume instructions
are preserved on [checkpoint PR #18](https://git.sunnycareboo.com/jasonbk/vllm-xpu-nix/pulls/18),
commit `78a30c20abacc2455d8f36991a9783e251d60ace`. Resume only on explicit user
request, then reassess upstream and audit applicability. No DFlash2 speedup or
performance ceiling has been established. Identical untracked report copies
were removed from this worktree; the checkpoint and private artifacts remain.

Raw release evidence stays in `benchmark-results/release-v030-20260922/`.
Repository cleanup inventory and a complete stash bundle are in
`benchmark-results/repository-cleanup-20260923/`. The old stash is also retained
as `refs/archive/stashes/20260923-pre-v1.10.0` (including index/untracked parents).
Recover it with `git stash store -m recovered-pre-v1.10.0 refs/archive/stashes/20260923-pre-v1.10.0`.
No stashed source changes were applied or silently declared redundant.
