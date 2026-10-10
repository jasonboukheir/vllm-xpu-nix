# Current fork delta and removal tradeoffs

For **xpu-v1.12.0**, vLLM has **five commits** on upstream v0.31.0
`db9527a46873454610df6dbedf79a36d6bf1a7f6`, and kernels have **seven commits**
on the official source for wheel 0.1.15.4,
`ddf336d86e3c8602888572a3502f951abd51df12`. The kernel wheel has no matching
Git tag; its release-branch source mapping is recorded in the audit. Both fork
mains are published; installed native and actual Brutus foreground
qualification passed, including full-context and near-full-capacity requests.

This refresh removes the local early dispatcher-pruning optimization and
replaces our ragged-MTP production fix with upstream #600, retaining independent
regressions. Together they reduce the native production diff by 173 changed
lines across four files. The previous release passed the real Brutus profile;
that does not prove every retained patch is indispensable. Distinguish required custom functionality, correctness guards,
optional supported configurations, regression tests and unmeasured optimizations.
The earlier 27 commits were audited and consolidated; the registry workaround
and its revert disappeared together. Squashing more commits would not reduce
the actual maintenance delta.

GDN is Qwen's recurrent-attention component. Several small fixes address its
metadata, scheduling and state handling independently of KVarN quantization.

## vLLM: five commits

| Commit | Purpose | Removal consequence / retention strength |
| --- | --- | --- |
| [3ecc9a1d4d](https://git.sunnycareboo.com/jasonbk/vllm/commit/0da901567df9c27924eab9fe75370b6c2af3d49b) | Trim padded GDN metadata | Pass only live request indices to the native kernel, keeping speculative rollback columns intact. Upstream still forwards padded indices to exact-size native checks. This is a metadata compatibility safeguard; it is not proof that every eager request needs it. |
| [490bf7e422](https://git.sunnycareboo.com/jasonbk/vllm/commit/5b30fbe7cb821b5b4c64f57eae58a405a9dcb01f) | Qwen GDN scheduling and mixed batches | Keep non-final prefills on the 64-token arithmetic grid and select the paired native mixed-batch path. Removing it restores scheduler-dependent arithmetic choices. The mixed-batch portion and kernel 7bc80d1 must be changed together. |
| [41ab2f1882](https://git.sunnycareboo.com/jasonbk/vllm/commit/e3deabcbe900c03dba4f0d278fc82a6b8ad279c5) | Functional, row-stable XPU residual RMSNorm | Preserve a non-mutating, fixed-reduction-tree helper and the optional scoped request-stability caller. The current RedHat MTP/vision profile does not activate that policy. Removable if we explicitly retire that behavior; passing helper tests does not make whole-model AEON invariance pass. |
| [1e1d1471b7](https://git.sunnycareboo.com/jasonbk/vllm/commit/3506c0947e58b9c5fc29b31471196921574c85bb) | Explicit FP16/BF16 KV-cache names | The native writer rejects these literal names. Translate only its argument to auto after checking all four tensor dtypes; retain the explicit engine setting so checkpoint FP8 metadata cannot override it. Optional for the current all-K4V2 setup, needed for explicit unquantized cache/draft configurations. |
| [ef92dc5600](https://git.sunnycareboo.com/jasonbk/vllm/commit/6587f68a48f7459b3d87148e084b518872aa1f4b) | KVarN integration, K4V2, MTP and cache ownership | Supplies the custom backend, compact layouts, separate attention/recurrent pools, target/draft ownership, capacity accounting, exact committed-history metadata, flush/materialization and reset handling. Upstream v0.31 has no equivalent KVarN feature. The replay preserves its new checkpoint, allocator, DCP/TP and MTP annotation behavior; it adapts the spec-aware block-size API and drops an obsolete failure-loop assertion. Removing this loses the requested cache configuration; it must match kernel dbb684f. |

## XPU kernels: seven commits

| Commit | Purpose | Removal consequence / retention strength |
| --- | --- | --- |
| [dcbbeac](https://git.sunnycareboo.com/jasonbk/vllm-xpu-kernels/commit/e50d437ee5560e5b5f12581ee61b1bddab157733) | Deterministic oneDNN W4A16 matmul | Requests deterministic accumulation for the model’s INT4 projections. Upstream uses the same oneDNN path without that setting. Can be dropped only by accepting a different reproducibility policy or an equivalent replacement; no current on/off speed benefit was measured. |
| [7a1af66](https://git.sunnycareboo.com/jasonbk/vllm-xpu-kernels/commit/63a31d01e8fcf44a7b08f8e76e85ec40a208b6f1) | Use untiled GDN convolution prefill | Avoids the tiled route implicated in historical fresh-state corruption/repeated punctuation. The selected upstream source retains that route unchanged. Remove when a corrected tiled path is shown to work on the supported model/toolchain, rather than reinstating it incidentally. |
| [2322b83](https://git.sunnycareboo.com/jasonbk/vllm-xpu-kernels/commit/ad5d9e16721f5c219de65c8d408c02edab104919) | Order fresh-state GDN output reads | Adds a workgroup barrier before fresh-state output computation reads values written by other subgroups. Other upstream barriers do not cover that producer/consumer boundary. Keep the ordering fix or replace it with equivalent synchronization. |
| [7bc80d1](https://git.sunnycareboo.com/jasonbk/vllm-xpu-kernels/commit/d8e1f69699496beeda61da9a544f49d9b6378d73) | Keep cached-decode arithmetic in mixed batches | Separates eligible ordinary decode requests from prefill requests inside GDN, preserving their standalone arithmetic path. It is paired with vLLM 490bf7e422 and adds the trailing native flag that caller uses. Removing only this side breaks that interface; removing both abandons the retained arithmetic policy. |
| [c75a699](https://git.sunnycareboo.com/jasonbk/vllm-xpu-kernels/commit/7d72670ba22c67480a4805a2d418beb19f83d004) | Shortened-query rollback/storage regressions | Test-only residual: upstream #600 now supplies the native ragged-query implementation. Keep local coverage of acceptance beyond current query length, interleaved storage guards and bitwise state continuation. |
| [2e7f76c](https://git.sunnycareboo.com/jasonbk/vllm-xpu-kernels/commit/348a8b91e4c301ff3c94b0fce68a4cda7a591bfc) | Native FlashAttention replay/reference tests | Test-only: five native cases check independent numerical references and replay while rejecting fallback. Removing it changes no serving behavior. It reduces regression coverage, so upstreaming or retaining the small test delta is preferable to deleting it solely to reduce commit count. |
| [dbb684f](https://git.sunnycareboo.com/jasonbk/vllm-xpu-kernels/commit/05d4536b2440db171741b5de5c148cdc834bd692) | Native KVarN/K4V2 attention kernels | Supplies the paired quantized record writers/readers, packing, scatter, decode and bounded materialization kernels. Removing it makes the vLLM KVarN feature unavailable. The folded reducer and resident-load optimizations are separately reviewable, not inherently inseparable from K4V2. |

## Optional pieces inside the large KVarN commit

The sixteen-split reducer and aligned resident-FP16 block loads were folded into
the native feature commit. They can be removed independently of the cache format
if their fallback paths remain correct. The historical reducer gain was small
(about 3% in one short cell), with an unresolved 3.67% K4V4 regression. The load
change had stronger historical K4V2 kernel evidence, but also a short K4V4 anomaly.
Neither received a fresh controlled with/without performance comparison for
v1.11.0; this refresh does not claim to re-establish those controlled gains.

Existing variants, reference paths and optional scoped stability code also make
the feature commits large. They are not all required by one RedHat profile.
Removing supported alternatives is a deliberate scope reduction; eliminating
unused definitions and folding historical repairs was already done in the replay.
The dedicated reports distinguish source equivalence, installed CPU checks and
installed native qualification; controlled optimization measurements are separate.

## Four additional Nix build patches

These are applied by packaging and are additional source delta, even though
they are not commits in either fork's five/seven-commit stack:

| Patch | Why retained | Removal opportunity |
| --- | --- | --- |
| `0001-split-kernel-libs` | Lets CMake import separately built Nix kernel libraries, so their closures can be cached independently. | Requires upstream equivalent support or redesigning the split package build. |
| `0004-skip-prebuilt-additional-libs` | Prevents setuptools from rebuilding/copying libraries supplied by Nix. | Tied to that same split/prebuilt packaging design. |
| `0005-reduce-kernel-build-memory` | Makes device-link concurrency and template diagnostic backtrace limits configurable. | No controlled comparative resource gain was measured. Split libraries use NIX_BUILD_CORES link jobs (four in qualification); base glue/FA2 retain sixteen. The backtrace knob is especially weak as a successful-build memory claim. |
| `0006-forward-mhc-feature-flag` | Forwards the MHC option from setup.py to CMake so the split FA2 binding respects packaging's disabled feature selection. | A small upstreamable plumbing fix; removable when upstream forwards it. |

## Practical next reductions

The early dispatcher-pruning optimization is retired; both normal
and narrow build gates passed. The ragged-MTP production fix is now upstream-owned.
Optional request-stability and explicit unquantized-cache support remain
supported; removing them would be a deliberate scope reduction. The test-only
commits can be upstreamed without changing serving. The folded reducer merits
more skepticism than the essential KVarN record ABI. V2/DFlash2 still needs the
KVarN cache ownership/history lifecycle port; it is not a runtime flag flip.

Current per-change evidence and disposition mapping are in
[the v0.31 audit](updates/release-v031-20261010/reconciliation.md), with a
[performance review](updates/release-v031-20261010/performance-review.md).

Full per-commit evidence and the original KEEP/DROP/ADAPT map are preserved in
[the published audit archive](https://git.sunnycareboo.com/jasonbk/vllm-xpu-nix/src/commit/8c72895340deafef77998a579335d078800626ab/docs/updates/release-v030-20260922).
The [prior release notes](releases/xpu-v1.11.0.md) distinguish actual Brutus passes from
optional AEON/NaN limitations and unmeasured performance comparisons.
Issue #17 remains paused; no V2/DFlash support or speedup is implied.
