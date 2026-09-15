# xpu-v1.10.0 — compact K4V2 MTP qualification

This follow-up qualifies explicit K4V2 MTP drafting with a K4V2 target on
Brutus. K4V2 drafting already exists in xpu-v1.9.0; the new vLLM changes fix
CPU draft-metadata ownership and permit an explicit BF16 draft cache for
comparison. The target format, quantization, history policies, native kernels
and both upstream bases remain unchanged.

The compact performance comparison supports K4V2 drafting: the largest measured
decode regression is 1.48%, and throughput, first-token time and latency differ
by less than 1%. Every metric and workload stays within the practical 3%
regression threshold on both fresh-start comparisons. The locked release package
resolves to the exact measured runtime and passes the final packaging checks.

## Configuration

The qualified profile selects both caches explicitly:

```nix
kvCacheDtype = "kvarn_k4v2_g128_compact";
speculativeConfig = {
  method = "mtp";
  num_speculative_tokens = 2;
  kv_cache_dtype = "kvarn_k4v2_g128_compact";
};
```

Keep BF16 compute, eager V1 execution, prefix caching disabled, four scheduler
slots, 2,048 batched prefill tokens and the existing two-image limit. The tested
model is `RedHatAI/Qwen3.8-27B-INT4`, revision
`bf08f3dbd9a324e53956920aad378a1f1b6dd24a`, using W4A16 weights on Intel Arc
Pro B70. The individual request limit remains 262,144 combined input/output
tokens. Service defaults are unchanged.

The MTP layer has its own attention weights and KV contents. Matching target
and draft formats does not make their caches interchangeable or reusable.

## Prediction agreement

The repaired common package was compared at 8,448 unique committed prefixes
across eight retained fixtures. Two fresh starts reproduce each format's
predictions exactly; those repeats are not additional independent accuracy
samples. Target token histories, hidden states and causal metadata match
between formats. The second proposal is scored on the drafter's own rollout.

| Draft cache | First agreement | Both proposals agree | Accepted drafts per pair |
| --- | ---: | ---: | ---: |
| K4V4 | 84.730% | 66.596% | 1.51326 |
| K4V2 | 84.647% | 66.323% | 1.50971 |
| BF16 | 84.813% | 66.607% | 1.51420 |

K4V2 has seven fewer matching first proposals and 23 fewer matching second
proposals after a matching first than K4V4, over those 8,448 prefixes. This is
a small observed difference on these fixtures, not proof of statistical
equivalence or a speed result. Free-running acceptance and output trajectories
are reported separately with the serving measurements.

## Memory and capacity

At equal 262,144-token usable attention capacity, K4V2 drafting saves
64.03125 MiB of observed attention tensor storage versus K4V4 drafting,
including the null page. The logical draft KV allocations are 210.10254 MiB
and 274.13379 MiB respectively. These are tensor-storage measurements, not
whole-process VRAM peaks. Shared scratch is counted once.

With the same 0.96 device-memory budget and four scheduler slots:

| Draft cache | Usable shared attention tokens | Minimum observed occupied tokens |
| --- | ---: | ---: |
| K4V4 | 340,608 | 339,456 |
| K4V2 | 346,880 | 345,728 |

K4V2 adds 6,272 shared tokens, about 1.84%, while retaining the same per-request
limit. Each compact configuration completed two concurrent long prompts with
512 output tokens per request and no preemption. Those capacity checks use
different prompt lengths and are not matched performance measurements.

## Serving performance

The target remains K4V2, with MTP2 and vision enabled. The comparison used
K4V2/K4V4/K4V4/K4V2 order, two fresh starts per draft format, and a warmup plus
three measured repetitions at every workload. Both formats had 262,144 usable
attention tokens and four scheduler slots; every request generated 256 tokens.

The table gives the range of K4V2 percentage changes across the two paired
fresh-start medians. Positive decode/throughput changes are faster; negative
first-token/latency changes are shorter.

| Context / concurrency | Decode | Output throughput | First-token time | Latency |
| --- | ---: | ---: | ---: | ---: |
| 4K / 1 | -1.48% to +1.37% | -0.92% to +0.93% | -0.55% to +0.06% | -0.91% to +0.91% |
| 4K / 4 | +0.67% to +1.07% | +0.10% to +0.32% | -0.50% to -0.39% | -0.75% to -0.52% |
| 16K / 1 | +1.00% to +1.74% | +0.38% to +0.55% | -0.10% to -0.02% | -0.53% to -0.37% |
| 16K / 4 | +0.46% to +0.79% | +0.25% to +0.47% | -0.04% to -0.03% | -0.35% to -0.17% |
| 65K / 1 | +1.81% to +3.12% | +0.22% to +0.38% | +0.01% to +0.04% | -0.38% to -0.21% |
| 65K / 4 | -1.30% to -1.24% | -0.56% to -0.50% | -0.04% to +0.04% | +0.53% to +0.58% |
| 128K / 1 | +0.10% to +0.49% | +0.02% to +0.02% | -0.01% to -0.00% | -0.02% to -0.01% |
| near-262K / 1 | +0.58% to +0.81% | +0.04% to +0.05% | -0.02% to -0.02% | -0.05% to -0.04% |

All 32 metric/workload assessments pass the practical 3% regression criterion
on both starts (64 individual comparisons). This is a practical release
decision, not a statistical-equivalence test. The 65K single-request decode
gain reaches 3.12% in one comparison; it is not a universal speedup. Absolute
timings, all six-wave ranges, per-start medians and actual MTP verification
counters are retained in the accompanying JSON manifest.

All 30 paired single-request outputs match. At concurrency four, 15 of 72
paired outputs match; the remaining 57 diverge. These are free-running
performance measurements with potentially different verification work and
histories. The matched-history agreement study above remains the precision
comparison.

Decode excludes the entire first SSE chunk. At concurrency four, its interval
can include other requests' prefill, so it is not isolated steady-state decode.
SSE chunks are not individual-token latency samples.

Each corrected start has complete timestamped memory coverage, with no collector
errors and maximum sample gaps below 0.71 seconds. At equal capacity, sampled
owned DRM residency peaks are 30,557,466,624 / 30,557,458,432 bytes for K4V2
and 30,626,807,808 / 30,626,836,480 bytes for K4V4. The paired sampled-peak
difference is about 66.1 MiB; this is not an instantaneous allocator peak.
The separate attention-tensor storage saving remains 64.03125 MiB.

A single retained MTP-off start is a descriptive control. Against the first
K4V2 MTP2 start, its five matched workloads show mixed effects: 4K single-request
decode improves by 35.4%, while 65K concurrency-four output throughput falls
by 25.0%. Thirty-two of 33 paired output trajectories differ. This is not a
two-start net-MTP-benefit experiment or a blanket MTP-speedup claim.

BF16 serving performance and deeper profiling are deferred. The completed
compact measurements show no material regression under the tested profile.

## Correctness and qualification limits

The metadata fix prevents shared or overlapping CPU sequence-length shadows
from being incremented twice during one draft step. The former behavior could
advance cache metadata prematurely and flush a partially committed KVarN tile.
Independent regressions reproduce the defect on the predecessor. Earlier
matched-history results collected with that defect remain excluded.

The common candidate passes 133 focused CPU checks and 62 selected GPU checks,
plus 175 affected packaging-harness checks after the collector repair. Both
compact formats pass the retained
functional suite: MTP activity, tools, two images, concurrent requests,
prefill during decode, cancellation and reuse across a cache flush, and
long-context retrieval. Vision retrieval passes at 131K and near-262K.

Quality remains bounded by the retained fixtures. An exact long-context
arithmetic control returns 901 instead of the expected 899 with both compact
drafts, BF16 drafting and MTP off. The original BF16 functional suite therefore
retains its failed sixth semantic gate; its other five checks are independently
reviewed. The exact numerical cause of the shared failure is not isolated.

Earlier tests poisoning unused BF16 cache positions with NaNs also retain six
failures on the unchanged native attention library. The selected causal
reference tests with zero and finite padding pass; they do not establish
robustness to NaN-poisoned unused storage. A startup sampling gap in one K4V2
diagnostic excludes that diagnostic's VRAM peak. A separate incomplete K4V4
timing capture is excluded because its memory collector stopped early. The
corrected comparison requires timestamped memory samples covering every wave
and propagates collector failures. All four corrected service starts are fresh.
Earlier capacity and MTP-off memory histories were checked through shutdown.

These results do not establish lossless generation, broad task accuracy or
that every diagnostic passed.

## Sources and release checks

| Component | Retained upstream base | Release source |
| --- | --- | --- |
| vLLM | `51da0ca66c8065619c79e35dff97aa99aeaf5644` | `2fdd9d71f6895379b0940a8f755b4c9095840558` |
| XPU kernels | `e7f20cbc18d741419a84aa57178e1f1cc0b453da` | `2c39287deec5e59cc2656e6d4a6dce51a567c411` |

The release adds two focused vLLM commits. Kernel source and native libraries
remain at the released revisions. Packaging commit
`24555feacbfcfab2b85240688ddc2ac92a60de07` adds the audited serving-acceptance
and matched-history helpers with their tests. Commit
`282114f7b5b44ef62e75a0c656c94742c4db9c69` repairs collection of memory samples
and audits their coverage. Code within the timed request interval is unchanged.
Only the two paired source locks change; all unrelated dependency pins are unchanged.

The actual tagged source pair realizes the exact benchmarked vLLM package and
Python runtime, with both native library hashes verified. The final packaging
checks pass: kernel library identity, kernel binding identity, build parallelism,
formatting, and 597 CPU tests across all 45 test files, with no failures or skips.
The unit loop retains its original quantization environment and uses the
qualified model runtime for forced-decode tests. These tests overlap the earlier
175 affected harness checks and must not be added together as unique tests.

All three repositories use the annotated tag `xpu-v1.10.0` and permanent
`releases/xpu-v1.10.0` branch. The accompanying [manifest](xpu-v1.10.0.json)
records the source pair, exact realized package, measurements and evidence hashes.
Raw captures and failed candidates remain in the private benchmark archive.
Host deployment is a separate operational validation.
