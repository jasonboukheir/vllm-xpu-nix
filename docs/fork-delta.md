# Current fork delta and removal tradeoffs

For published **xpu-v1.11.0**, vLLM has **five commits** on upstream v0.30.0
`ced6857afa0ea7b2e3f0846a62e1394e90f15607`, and kernels have **eight commits**
on upstream v0.1.15 `1c7cbeee1cd0c1481d48f5031f679a47f3f0ef45`.
This explains the frozen release pair, not an assertion about today's upstream main.

We can remove some of this delta. The release retained existing behavior and
passed the real Brutus configuration; that does not prove every retained patch
is indispensable. Distinguish required custom functionality, correctness guards,
optional supported configurations, regression tests and unmeasured optimizations.
The earlier 27 commits were audited and consolidated; the registry workaround
and its revert disappeared together. Squashing more commits would not reduce
the actual maintenance delta.

GDN is Qwen's recurrent-attention component. Several small fixes address its
metadata, scheduling and state handling independently of KVarN quantization.

## vLLM: five commits

| Commit | Purpose | Removal consequence / retention strength |
| --- | --- | --- |
| [cfbd127232](https://git.sunnycareboo.com/jasonbk/vllm/commit/cfbd127232d3121e185eac2d24a9a50ea8141aa0) | Trim padded GDN metadata | Pass only live request indices to the native kernel, keeping speculative rollback columns intact. Upstream still forwards padded indices to exact-size native checks. This is a metadata compatibility safeguard; it is not proof that every eager request needs it. |
| [6efe5e7cc5](https://git.sunnycareboo.com/jasonbk/vllm/commit/6efe5e7cc59de64ae8d3ec67bbe77379be34d2c9) | Qwen GDN scheduling and mixed batches | Keep non-final prefills on the 64-token arithmetic grid and select the paired native mixed-batch path. Removing it restores scheduler-dependent arithmetic choices. The mixed-batch portion and kernel 9a17a2c must be changed together. |
| [d909b25f01](https://git.sunnycareboo.com/jasonbk/vllm/commit/d909b25f0119419481bc378c3c0a3c656f9b76b0) | Functional, row-stable XPU residual RMSNorm | Preserve a non-mutating, fixed-reduction-tree helper and the optional scoped request-stability caller. The current RedHat MTP/vision profile does not activate that policy. Removable if we explicitly retire that behavior; passing helper tests does not make whole-model AEON invariance pass. |
| [984e6c3edd](https://git.sunnycareboo.com/jasonbk/vllm/commit/984e6c3edda05c4f2dca37703534cb0e69cfa0fa) | Explicit FP16/BF16 KV-cache names | The native writer rejects these literal names. Translate only its argument to auto after checking all four tensor dtypes; retain the explicit engine setting so checkpoint FP8 metadata cannot override it. Optional for the current all-K4V2 setup, needed for explicit unquantized cache/draft configurations. |
| [466b9d5abe](https://git.sunnycareboo.com/jasonbk/vllm/commit/466b9d5abe84f13c29b38c686e4d77fc09e09aa4) | KVarN integration, K4V2, MTP and cache ownership | Supplies the custom backend, compact layouts, separate attention/recurrent pools, target/draft ownership, capacity accounting, exact committed-history metadata, flush/materialization and reset handling. Upstream v0.30 has no equivalent KVarN feature. Removing this loses the requested cache configuration; it must match kernel 193cee0. |

## XPU kernels: eight commits

| Commit | Purpose | Removal consequence / retention strength |
| --- | --- | --- |
| [98bdbfd](https://git.sunnycareboo.com/jasonbk/vllm-xpu-kernels/commit/98bdbfde3e75f164bc5040574990db843bf888de) | Deterministic oneDNN W4A16 matmul | Requests deterministic accumulation for the model’s INT4 projections. Upstream uses the same oneDNN path without that setting. Can be dropped only by accepting a different reproducibility policy or an equivalent replacement; no current on/off speed benefit was measured. |
| [e1bec3b](https://git.sunnycareboo.com/jasonbk/vllm-xpu-kernels/commit/e1bec3b3e9d0c02cc0e8ce9354d5d2908d51a87f) | Skip disabled prefill-policy template expansion | A build-only optimization for narrowed policy sets. Brutus enables all relevant policies, so this prunes none in its normal build. No controlled compile/RSS benefit was measured. This is the weakest standalone retention case and a reasonable removal candidate. |
| [956b61f](https://git.sunnycareboo.com/jasonbk/vllm-xpu-kernels/commit/956b61f95a0b38e73eeebe9cbfeff1be782a9283) | Use untiled GDN convolution prefill | Avoids the tiled route implicated in historical fresh-state corruption/repeated punctuation. Upstream v0.1.15 retains that route unchanged. Remove when a corrected tiled path is shown to work on the supported model/toolchain, rather than reinstating it incidentally. |
| [3b5deb0](https://git.sunnycareboo.com/jasonbk/vllm-xpu-kernels/commit/3b5deb0773be875a328a1831c5e4a08233460d32) | Order fresh-state GDN output reads | Adds a workgroup barrier before fresh-state output computation reads values written by other subgroups. Other upstream barriers do not cover that producer/consumer boundary. Keep the ordering fix or replace it with equivalent synchronization. |
| [9a17a2c](https://git.sunnycareboo.com/jasonbk/vllm-xpu-kernels/commit/9a17a2c8617be2b0471db79fb301912601ee945c) | Keep cached-decode arithmetic in mixed batches | Separates eligible ordinary decode requests from prefill requests inside GDN, preserving their standalone arithmetic path. It is paired with vLLM 6efe5e7cc5 and adds the trailing native flag that caller uses. Removing only this side breaks that interface; removing both abandons the retained arithmetic policy. |
| [30fda95](https://git.sunnycareboo.com/jasonbk/vllm-xpu-kernels/commit/30fda95e35666948deb3c9bd7fd72aaff9c2ad66) | Handle shortened speculative GDN queries | Separate actual query length from retained rollback capacity. Without it, short/ragged MTP verification can hit a rectangular-size assertion or traverse/write the wrong token/state positions. Equivalent upstream fix da16a559 (#600) is just after v0.1.15, not inside this release. Reassess removal on a base that includes it. |
| [8b6ec9d](https://git.sunnycareboo.com/jasonbk/vllm-xpu-kernels/commit/8b6ec9d6a31836e3f19447ec790169655ce7da6a) | Native FlashAttention replay/reference tests | Test-only: five native cases check independent numerical references and replay while rejecting fallback. Removing it changes no serving behavior. It reduces regression coverage, so upstreaming or retaining the small test delta is preferable to deleting it solely to reduce commit count. |
| [193cee0](https://git.sunnycareboo.com/jasonbk/vllm-xpu-kernels/commit/193cee080d654ff67c4f8bfaf924201bc88b9cf5) | Native KVarN/K4V2 attention kernels | Supplies the paired quantized record writers/readers, packing, scatter, decode and bounded materialization kernels. Removing it makes the vLLM KVarN feature unavailable. The folded reducer and resident-load optimizations are separately reviewable, not inherently inseparable from K4V2. |

## Optional pieces inside the large KVarN commit

The sixteen-split reducer and aligned resident-FP16 block loads were folded into
the native feature commit. They can be removed independently of the cache format
if their fallback paths remain correct. The historical reducer gain was small
(about 3% in one short cell), with an unresolved 3.67% K4V4 regression. The load
change had stronger historical K4V2 kernel evidence, but also a short K4V4 anomaly.
Neither received a fresh controlled with/without performance comparison for
v1.11.0. We cannot claim those benefits were re-established on v0.30.

Existing variants, reference paths and optional scoped stability code also make
the feature commits large. They are not all required by one RedHat profile.
Removing supported alternatives is a deliberate scope reduction; eliminating
unused definitions and folding historical repairs was already done in the replay.
No additional runtime/source changes were made as part of repository cleanup.

## Four additional Nix build patches

These are applied by packaging and are additional source delta, even though
they are not commits in either fork's five/eight-commit stack:

| Patch | Why retained | Removal opportunity |
| --- | --- | --- |
| `0001-split-kernel-libs` | Lets CMake import separately built Nix kernel libraries, so their closures can be cached independently. | Requires upstream equivalent support or redesigning the split package build. |
| `0004-skip-prebuilt-additional-libs` | Prevents setuptools from rebuilding/copying libraries supplied by Nix. | Tied to that same split/prebuilt packaging design. |
| `0005-reduce-kernel-build-memory` | Makes device-link concurrency and template diagnostic backtrace limits configurable. | No controlled comparative resource gain was measured. Split libraries use four link jobs; base glue/FA2 retain sixteen. The backtrace knob is especially weak as a successful-build memory claim. |
| `0006-forward-mhc-feature-flag` | Forwards the MHC option from setup.py to CMake so the split FA2 binding respects packaging's disabled feature selection. | A small upstreamable plumbing fix; removable when upstream forwards it. |

## Practical next reductions

Start with `e1bec3b` if narrowing the fork for the actual Brutus build. Consider
retiring the optional request-stability policy and explicit unquantized-cache
support only if those configurations are no longer wanted. The test-only commit
can be upstreamed without changing serving. The ragged-MTP production fix can
be revisited when the selected kernel release includes upstream's equivalent.
The folded reducer merits more skepticism than the essential KVarN record ABI.
These are recommendations for future source work, not removals performed here.

Full per-commit evidence and the original KEEP/DROP/ADAPT map are preserved in
[the published audit archive](https://git.sunnycareboo.com/jasonbk/vllm-xpu-nix/src/commit/8c72895340deafef77998a579335d078800626ab/docs/updates/release-v030-20260922).
The [release notes](releases/xpu-v1.11.0.md) distinguish actual Brutus passes from
optional AEON/NaN limitations and unmeasured performance comparisons.
Issue #17 remains paused; no V2/DFlash support or speedup is implied.
