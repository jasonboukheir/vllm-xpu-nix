# Current fork delta and removal tradeoffs

For **xpu-v1.13.0**, vLLM has **five commits** on upstream v0.32.0
`5df63a7203cdb9328d824a27b682e77c7b9928be`. Kernels retain **seven commits**
on the frozen official source for wheel 0.1.15.4,
`ddf336d86e3c8602888572a3502f951abd51df12`, matching vLLM's exact dependency.
The wheel has no matching Git tag; its released-wheel/source mapping is recorded
in the audit. The four independent correctness fixes remain separate; KVarN,
RAM prefix retention and their V2/DFlash lifecycle port form one feature commit.

All eight previous vLLM commits and all seven unchanged kernel commits were
reviewed. The [reconciliation map](updates/release-v032-v2-20261019/reconciliation.json)
records ownership and retained behavior. Installed source equivalence, native
references, synthetic serving checks and matched performance have separate
evidence; whole-profile gains do not establish an isolated gain for every patch.

GDN is Qwen's recurrent-attention component. Several small fixes address its
metadata, scheduling and state handling independently of KVarN quantization.

## vLLM: five commits

| Commit | Purpose | Removal consequence / retention strength |
| --- | --- | --- |
| [d8d41a655d](https://git.sunnycareboo.com/jasonbk/vllm/commit/d8d41a655d47faba309a5fe5c98f08c0333ef51c) | Trim padded GDN metadata | Pass only live request indices to the native kernel, keeping speculative rollback columns intact. Upstream still forwards padded indices to exact-size native checks. This is a metadata compatibility safeguard; it is not proof that every eager request needs it. |
| [1214f2058a](https://git.sunnycareboo.com/jasonbk/vllm/commit/1214f2058ad9c91790f87c92f6fec6b3ad312430) | Qwen GDN scheduling and mixed batches | Keep eligible text-only, non-speculative Mamba-none prefills on the 64-token arithmetic grid and select the paired native mixed-batch path. Aligned-cache mode already follows cache-block boundaries. Removing it restores scheduler-dependent arithmetic choices. The mixed-batch portion and kernel 7bc80d1 must be changed together. |
| [549d6509e7](https://git.sunnycareboo.com/jasonbk/vllm/commit/549d6509e7adcad95c54a634148f1aa1e25e3875) | Functional, row-stable XPU residual RMSNorm | Preserve a non-mutating, fixed-reduction-tree helper and the optional scoped request-stability caller. The current RedHat DFlash/vision profile does not activate that policy. Removable if we explicitly retire that behavior; passing helper tests does not make whole-model AEON invariance pass. |
| [49f42db428](https://git.sunnycareboo.com/jasonbk/vllm/commit/49f42db4285d3ebfa25f0b0a3cde413afa7442a5) | Explicit FP16/BF16 KV-cache names | The native writer rejects these literal names. Translate only its argument to auto after checking all four tensor dtypes; retain the explicit engine setting so checkpoint FP8 metadata cannot override it. Required for the explicit BF16 DFlash cache; retained for other unquantized-cache configurations. |
| [c0722c98d0](https://git.sunnycareboo.com/jasonbk/vllm/commit/c0722c98d0fa1588eaf748cc9f7ef69ca3c32245) | KVarN integration, RAM prefixes and V1/V2 cache ownership | Supplies the custom backend, compact layouts, separate attention/recurrent pools, target/draft ownership, capacity accounting, exact committed-history metadata, flush/materialization and reset handling. Upstream v0.32 has no equivalent KVarN feature. The replay preserves upstream checkpoint proofs and aligned strides, adds independent BF16 draft pools and bounded native verification, and retains per-pool async admission and RAM replay-window ownership. Removing this loses the requested cache configuration; it must match kernel dbb684f. |

## XPU kernels: seven commits

| Commit | Purpose | Removal consequence / retention strength |
| --- | --- | --- |
| [dcbbeac](https://git.sunnycareboo.com/jasonbk/vllm-xpu-kernels/commit/dcbbeac2f2453948471569de43a33ded09fae0f8) | Deterministic oneDNN W4A16 matmul | Requests deterministic accumulation for the model’s INT4 projections. Upstream uses the same oneDNN path without that setting. Can be dropped only by accepting a different reproducibility policy or an equivalent replacement; no current on/off speed benefit was measured. |
| [7a1af66](https://git.sunnycareboo.com/jasonbk/vllm-xpu-kernels/commit/7a1af6693963e91b3e63c119501ce5ea8fbfa955) | Use untiled GDN convolution prefill | Avoids the tiled route implicated in historical fresh-state corruption/repeated punctuation. The selected upstream source retains that route unchanged. Remove when a corrected tiled path is shown to work on the supported model/toolchain, rather than reinstating it incidentally. |
| [2322b83](https://git.sunnycareboo.com/jasonbk/vllm-xpu-kernels/commit/2322b830903a4a0d0d8e4c113b7e2acc5c7967cb) | Order fresh-state GDN output reads | Adds a workgroup barrier before fresh-state output computation reads values written by other subgroups. Other upstream barriers do not cover that producer/consumer boundary. Keep the ordering fix or replace it with equivalent synchronization. |
| [7bc80d1](https://git.sunnycareboo.com/jasonbk/vllm-xpu-kernels/commit/7bc80d1569bedeb8f4adceb43524fbc1b4d07516) | Keep cached-decode arithmetic in mixed batches | Separates eligible ordinary decode requests from prefill requests inside GDN, preserving their standalone arithmetic path. It is paired with vLLM 1214f2058a and adds the trailing native flag that caller uses. Removing only this side breaks that interface; removing both abandons the retained arithmetic policy. |
| [c75a699](https://git.sunnycareboo.com/jasonbk/vllm-xpu-kernels/commit/c75a699930db98a3c77e4d6498892734e624b3a5) | Shortened-query rollback/storage regressions | Test-only residual: upstream #600 now supplies the native ragged-query implementation. Keep local coverage of acceptance beyond current query length, interleaved storage guards and bitwise state continuation. |
| [2e7f76c](https://git.sunnycareboo.com/jasonbk/vllm-xpu-kernels/commit/2e7f76c799e3c8b68b4ae93d8c3e8eb65a525454) | Native FlashAttention replay/reference tests | Test-only: five native cases check independent numerical references and replay while rejecting fallback. Removing it changes no serving behavior. It reduces regression coverage, so upstreaming or retaining the small test delta is preferable to deleting it solely to reduce commit count. |
| [dbb684f](https://git.sunnycareboo.com/jasonbk/vllm-xpu-kernels/commit/dbb684f4c9b38a8b3bbf52bda0c490ee47049221) | Native KVarN/K4V2 attention kernels | Supplies the paired quantized record writers/readers, packing, scatter, decode and bounded materialization kernels. Removing it makes the vLLM KVarN feature unavailable. The folded reducer and resident-load optimizations are separately reviewable, not inherently inseparable from K4V2. |

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
more skepticism than the essential KVarN record ABI. The V2/DFlash2 port
now carries target, recurrent and draft ownership through allocation, rejected
proposals, context boundaries and RAM transfers. The existing fused DFlash2
grouped-convolution path remains CUDA-only; enabling it on XPU is a separate
optimization requiring numerical and serving measurements.

Current per-change evidence and disposition mapping are in
[the v0.32 reconciliation](updates/release-v032-v2-20261019/reconciliation.json),
with [serving qualification](updates/release-v032-v2-20261019/finish-serving-validation.json)
and a [matched performance review](updates/release-v032-v2-20261019/finish-performance-review.json).
The [release manifest](releases/xpu-v1.13.0.json) identifies the exact pair and profile.

Earlier evidence remains in the [v0.31 audit](updates/release-v031-20261010/reconciliation.md)
and [published v0.30 audit archive](https://git.sunnycareboo.com/jasonbk/vllm-xpu-nix/src/commit/8c72895340deafef77998a579335d078800626ab/docs/updates/release-v030-20260922).
Optional AEON stability and controlled per-kernel optimization claims remain
separate from the qualified RedHat V2/DFlash2 profile.
