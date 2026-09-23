# Issue #17 upstream progress check — September 22, 2026

This is a read-only upstream assessment requested while the release goal remains
**paused**. No implementation, rebase, build, benchmark, qualification, source
publication, issue-body edit, service change or activation was performed. The
September 20 checkpoint, its artifact index and its resume rules remain intact.

**Assessment: useful upstream progress, but the pinned syvai W4A16 DFlash2 +
Model Runner V2 + K4V2 target/draft configuration is still not established as
supported on Brutus.** No DFlash2 speedup or performance ceiling is established.
The planned published-v0.32-series reassessment point has not been reached.

## Observed identities and release boundary

GitHub API/source observations were made September 22 PDT (September 23 UTC
for the final observations):

| Item | Observed identity |
| --- | --- |
| Latest vLLM release | [v0.30.0](https://github.com/vllm-project/vllm/releases/tag/v0.30.0), published 2026-09-22 05:20 UTC, commit `ced6857afa0ea7b2e3f0846a62e1394e90f15607` |
| vLLM main inspected | `94f4170df37fe29d68bd2f7e5a496501b78d55ae` |
| Latest XPU-kernels release | [v0.1.15](https://github.com/vllm-project/vllm-xpu-kernels/releases/tag/v0.1.15), published 2026-09-22 11:14 UTC |
| XPU-kernels main | `d7c35d281cf50564996fb0b9c41676b64f13f970`, unchanged since the previous observed main |

These are observation identities, not a newly selected/frozen candidate pair.
No v0.32-series release appears in the current releases list.

The new kernel release has Torch 2.13.0+xpu in its requirements and a oneAPI
2026.0.0 Docker build base. However, both vLLM v0.30.0 and inspected main still
pin **kernels 0.1.14.1, Torch 2.13.0, Triton 3.7.2+xpu** in
[`requirements/xpu.txt`](https://github.com/vllm-project/vllm/blob/94f4170df37fe29d68bd2f7e5a496501b78d55ae/requirements/xpu.txt).
New kernel release availability does not establish that substituting 0.1.15 is
qualified. The release branch also differs from the newer kernel main whose
dependency boundary was recorded in the original audit.

The six recent main fixes below are absent from v0.30.0's ancestry. The complete
11-commit release-only branch was checked for backports; none of these six appears
there. Relevant released sampler/profiler/config source was also inspected. Do
not infer that installing 0.30.0 supplies these new main changes.

## Concrete progress since the preserved snapshot

| Merged change | Date UTC | Relevance and evidence limits |
| --- | --- | --- |
| [#57277: XPU sampler in V2](https://github.com/vllm-project/vllm/pull/57277) | Sep 21 | Uses fused XPU top-k/top-p sampling. Author reports lower latency for Qwen3-0.6B eager runs at batches 1/32. This is useful XPU V2 evidence, not our Qwen3.8/MTP/vision or DFlash2 qualification. |
| [#57460: platform-aware profiling](https://github.com/vllm-project/vllm/pull/57460) | Sep 21 | Recreates the one-shot Torch profiler for each collection round, validates activity selection and shares worker lifecycle while retaining XPU defaults. Reported unit tests and manual CUDA collection do not establish Brutus XPU events/counters/overhead. |
| [#58065: DFlash async scheduling](https://github.com/vllm-project/vllm/pull/58065) | Sep 22 | Enables DFlash in explicit/default async scheduling. Validation uses Qwen3.8-27B + DFlash2 on 4xGB200; the reported 41% decode-throughput gain is an async-on/off NVIDIA experiment, not a stock-MTP comparison or a Brutus result. Exact committed-history integration remains our responsibility. |
| [#56448: cap synthetic DFlash/DSpark profiling batches](https://github.com/vllm-project/vllm/pull/56448) | Sep 22 | Avoids startup/dummy-run query-buffer overflow; Qwen3.8/DFlash2 startup/concurrency validation is on NVIDIA H20. This is memory-profiling batch correctness, distinct from validating performance-profiling tools. |
| [#56734: prevent dummy draft writes through stale tables](https://github.com/vllm-project/vllm/pull/56734) | Sep 21 | Protects V2 autoregressive draft dummy/padded rows from writing real KV slots. Motivating case is DP MTP with prefix caching; this does not finish our custom cache reset/ownership work. |
| [#51565: GDN stateless first chunk](https://github.com/vllm-project/vllm/pull/51565) | Sep 22 | Corrects classification of one-token initial chunks that could consume stale recurrent state; shared Qwen-relevant metadata fix with unit regression evidence. |

Also merged: [#55881](https://github.com/vllm-project/vllm/pull/55881), XPU
batch-invariance improvements for MoE backend selection and multi-rank reduction.
Its benefit to our TP1 dense target is limited; it does not prove the fork's
deterministic W4A16/GDN patches obsolete.

v0.30.0 now includes the XPU Qwen DFlash context-key normalization fix #56431,
but this fix was already present in the preserved September source snapshot.
Likewise, stacked-weight RMSNorm in kernels 0.1.15 was already covered by the
September upstream audit. These are newly released coverage, not newly resolved
audit gaps.

## Remaining waiting criteria

1. **Published v0.32 and compatible dependencies: pending.** Latest vLLM is
   0.30.0. v0.32 remains a planned reassessment point, with no patch version
   reserved and no promise that V1 removal establishes readiness.
2. **Our XPU Qwen + bundled MTP + vision on V2: not established by the reviewed
   evidence.** The sampler and correctness fixes help, but the reviewed upstream
   reports do not qualify our full target/profile. The removal of the dedicated
   Intel V2 CI file in [#58050](https://github.com/vllm-project/vllm/pull/58050)
   follows [#54823](https://github.com/vllm-project/vllm/pull/54823), which treats
   separate V2 jobs as redundant now that V2 is the default. It is not evidence
   that Intel has abandoned V2, nor proof of our exact configuration's coverage.
3. **Quantized loading/mapping/context projection: still unresolved.**
   [#53122](https://github.com/vllm-project/vllm/pull/53122) remains open and
   unchanged since September 16. Its proposed context-projection guard rejects
   incompatible weights; merging that alone would not make syvai run.
   [#51684](https://github.com/vllm-project/vllm/pull/51684), updated September 22,
   is a more substantial open proposal: deferred post-load construction and
   quantization-aware fused/dequantized/per-layer projection paths. Its test-result
   section still contains a TODO. Other proposals #51620 and #55262 also remain
   open; the latter is GPTQ-specific rather than general support for syvai's
   compressed-tensors checkpoint.
4. **XPU DFlash compilation: unchanged.**
   [#56787](https://github.com/vllm-project/vllm/issues/56787) remains open, with
   no comments and no update since September 14. The reported source workaround
   keeps the target compiled and the draft eager. This remains useful prior
   evidence, not a newly merged supported switch or validated cost for our
   quantized/K4V2 configuration. Compilation remains optional under our contract.
5. **V2 metadata/cache stability: not enough evidence to clear this criterion.**
   The common metadata and attention helper files inspected are byte-identical to
   the September snapshot, while the runner/speculator continues changing. A
   two-day unchanged-file window is not a stability guarantee. The local cache
   requirements below remain unresolved.

## Source confirmation and local work waiting cannot solve

At inspected main, draft quantization configuration still returns the config
without the mapping setup requested in #53122; its whole helper file has the
same Git blob as the preserved snapshot.
[`get_draft_quant_config`](https://github.com/vllm-project/vllm/blob/94f4170df37fe29d68bd2f7e5a496501b78d55ae/vllm/model_executor/models/utils.py#L959).

The context path still slices `qkv_proj.weight` and feeds a fused buffer to
`F.linear`, so an equivalent quantization-aware implementation has not replaced
the identified gap. The file's changes since the snapshot add attention-value
scaling, not packed-weight support.
[`_build_context_kv_buffers`](https://github.com/vllm-project/vllm/blob/94f4170df37fe29d68bd2f7e5a496501b78d55ae/vllm/model_executor/models/qwen3_dflash.py#L470),
[`_project_context_kv`](https://github.com/vllm-project/vllm/blob/94f4170df37fe29d68bd2f7e5a496501b78d55ae/vllm/model_executor/models/qwen3_dflash.py#L535).

V2 target builder initialization still lacks the custom active-layer ownership
filter needed by our integration, and autoregressive/MTP prefill still reuses
target attention metadata. This source evidence preserves the local ownership
gap even though upstream fixed a different dummy-slot corruption.
[`target initialization`](https://github.com/vllm-project/vllm/blob/94f4170df37fe29d68bd2f7e5a496501b78d55ae/vllm/v1/worker/gpu/model_runner.py#L623),
[`draft prefill`](https://github.com/vllm-project/vllm/blob/94f4170df37fe29d68bd2f7e5a496501b78d55ae/vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py#L318).

Our exact committed history, target/draft mutable ownership and MTP prefill,
D128 noncausal sliding-window K4V2, concurrent per-slot draft-window allocation,
and lifecycle/reset correctness still require local implementation and tests.
No upstream observation establishes correctness, capacity or throughput for
that custom cache implementation.

Recommendation: retain the pause and the planned v0.32 reassessment. The most
useful additional signal to watch is a merged, tested quantized-context solution
such as #51684 plus the draft mapping fix. On an explicit future resume, recheck
all statuses and audit applicability. The proposed MTP/vision-first phase split
still awaits approval; neither phase was started by this check.

Raw public API/source evidence, query timestamp and SHA256 index are preserved
under `benchmark-results/issue17-upstream-check-20260922/`. The original paused
records remain under `docs/updates/issue17-20260920-snapshot-audit/`.
