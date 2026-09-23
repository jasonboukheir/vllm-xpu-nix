# Issue #17 preserved checkpoint

Issue #17 remains paused at user request; resume only on explicit request.
This branch preserves the local dependency commit and all 27 source audit
reports, reconciliation, partial replay record, pause/resume records and the
September 22 read-only upstream assessment. It is not a qualified release.

The matching source checkpoint branches are:

- [vLLM checkpoint](https://git.sunnycareboo.com/jasonbk/vllm/src/branch/checkpoint/issue17-paused-20260920): `e378275a8f20eac9e92212e4a1fee84ec5a31dc7`, zero replay commits.
- [Kernel checkpoint](https://git.sunnycareboo.com/jasonbk/vllm-xpu-kernels/src/branch/checkpoint/issue17-paused-20260920): `c66a48486dcd6b76336ec8a8b984a4700f89e1d9`, one k01 replay, unqualified.

Full state is in [the pause checkpoint](docs/updates/issue17-20260920-snapshot-audit/CURRENT.md).
Large/ignored artifacts remain on Brutus at
`/home/jasonbk/Projects/vllm-xpu-nix/benchmark-results/issue17-20260920/`;
source clones/tool caches remain under `build-dev/issue17/`. They were not deleted
or uploaded by this checkpoint PR. The original artifact hashes and pre-edit
archive are preserved. No DFlash2 speedup or ceiling has been established.

The separately authorized v0.30 release workflow operates on current main and
must not be confused with this paused snapshot/V2/DFlash work. This checkpoint's
Hub/XGrammar dependency changes do not automatically belong in that release.
No host services or activation are changed by creating this branch/PR.
