# Source replay complete and published

This is the historical source-stage checkpoint. Final native qualification and
user-selected release acceptance are in `release-acceptance.json` and `CURRENT.md`;
its pending-work descriptions below are superseded by those records.

Both clean source stacks passed parent final review and were published with
exact leases after remote backups were verified. vLLM has five commits ending
at `466b9d5abe84f13c29b38c686e4d77fc09e09aa4`; kernels have eight ending at
`193cee080d654ff67c4f8bfaf924201bc88b9cf5`. `commit-map.json` maps all 27 original
commits to their final owner(s) or justified drops. `source-review.md` is the
authoritative record of semantic adaptations and source-only validation.

The KVarN port is committed. It preserves exact V1 metadata, target/draft
physical zeroing, independent pool consumers, HiSparse fields and cache/process
reset distinctions. CPU tests cover these adaptations and shared-pool behavior.
All applicable hooks pass; the upstream kernel mypy hook is a no-op and does
not establish a type-check result. No GPU/native qualification is claimed.

Remote backup branches `backup/main-pre-v030-20260922` and
`backup/origin-main-pre-v030-20260922` retain the original mains. Local kernel
`refs/recovery/release-v030-20260922/pre-final-format` retains the pre-EOF-cleanup
stack at `dfeb336f5d4947af7ca7636c0145e42e3e4ff29b`. All original audits
and recovery refs remain. xpu-v1.10.0 tags and issue17 checkpoint branches were
verified unchanged after publication.

Packaging now pins the final pair. Evaluation, final patch/projection/AOT checks
and lightweight checks pass; native build, correctness, lifecycle, serving,
capacity and controlled performance gates remain pending. No new stable refs
exist. Issue17 PR #18 is separate preservation work and is not a release input.
