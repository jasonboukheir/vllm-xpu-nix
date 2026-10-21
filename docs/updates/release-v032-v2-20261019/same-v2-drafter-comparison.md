# MTP and DFlash2 with the same V2 runner

This is an experimental comparison on the `xpu-v1.13.0` package, based on
vLLM 0.32.0 at `c0722c98d0fa1588eaf748cc9f7ef69ca3c32245`. Both profiles use
Model Runner V2. It separates the drafter choice from the V1-to-V2 change in
the earlier whole-stack release comparison.

## Experimental MTP prerequisite

Unmodified `xpu-v1.13.0` rejects V2 MTP with KVarN. Removing just that gate
allows startup but the first request fails because the MTP attention layer
has no metadata. The target's KVarN metadata deliberately excludes draft
layers, while the upstream MTP prefill path reuses target metadata.

The [unapplied compatibility patch](experimental-mtp-v2.patch) builds the
draft's metadata with its own attention groups and the existing target
slot mappings. Recomputing those mappings from draft positions would risk
using stale positions in rejected-token padding. The patch was loaded through
a private Python overlay; both benchmark profiles used the identical overlay.
Installed packages, source commits and release tags were not changed.

This patch is an experiment for review, not a supported MTP release. Its gate
change is broader than the tested compact K4V2, K2, eager, single-device
profile. These probes do not qualify every configuration admitted by it,
general model quality, deterministic replay or long-term service operation.

## Matched speed measurements

Both profiles used the same B70, target model revision, KVarN K4V2 target cache,
93% GPU budget, 131,072-token context limit, four request slots, two speculative
proposals, 2,048-token batch budget and 1,024-token long-prefill threshold.
MTP used compact K4V2 draft cache; Syvai's quantized DFlash2 used its BF16
sliding draft cache. Those drafter-specific costs are part of the comparison.

Each cell below is the median of three requests with 512 output tokens,
temperature zero and seed 17. There were 24 measured requests per profile.
Request and prompt-token hashes matched across profiles. Server counters
confirmed speculation, complete token counts and zero preemptions. No media
transcodes ran during these speed measurements. The 4K and 16K measurements
used warm RAM-restored prefixes, so they do not measure cold prefill.

| Workload | Concurrent requests | MTP tokens/s | DFlash2 tokens/s | DFlash2 gain |
| --- | ---: | ---: | ---: | ---: |
| Short English | 1 | 44.32 | 49.26 | 11.1% |
| Short reasoning task | 1 | 52.28 | 59.97 | 14.7% |
| 4K prefix | 1 | 49.41 | 65.05 | 31.7% |
| 16K prefix | 1 | 55.60 | 60.89 | 9.5% |
| Short English | 2 | 42.95 | 48.31 | 12.5% |
| Short reasoning task | 2 | 51.80 | 58.11 | 12.2% |
| 4K prefix | 2 | 41.64 | 52.19 | 25.3% |
| 16K prefix | 2 | 47.88 | 52.81 | 10.3% |

Rates are per request, excluding the first complete SSE token chunk and
measuring through the last token chunk. The reasoning-task prompt also had
thinking disabled, consistently with the other workloads. Time to first
content, ranges, aggregate rates, cache counters and hashes are retained in
[the machine-readable comparison](same-v2-drafter-comparison.json).
One server session per profile and three repeats per cell establish a local
screen, not a statistical or general throughput guarantee. Neither drafter's
proposal count was tuned independently.

DFlash2's observed gain is about 11–15% for these short prompts and 10–32%
across the full matrix. The earlier cross-version gains also included runner
and configuration changes and must not be attributed to the drafter alone.

## Context and GPU sharing

At the same 93% budget, MTP's shared attention pool holds 226,688 tokens;
DFlash2's holds 149,888. MTP loads about 0.45 GiB less model weight memory and
does not reserve DFlash2's separate 502.5 MiB sliding cache. It instead adds
one compact attention layer to the shared pool. Both retain four recurrent
request slots; those slots share the attention pool rather than each receiving
a full context allocation.

In a separate capacity check, experimental MTP passed a **196,608-token (192K)
combined input/output limit** with both Intel SDR transcodes already running.
A 196,479-token two-image request returned the correct label and image colors;
two RAM restores returned identical tokens with 196,224 cache-hit tokens and
255 recomputed tokens each. Two exact-limit requests and a four-request
decode/RAM-restore check also passed without preemption. Both transcodes
maintained approximately real-time progress with no reported dropped frames.
The looped synthetic media produced non-monotonic DTS warnings at the null
muxer; these were throughput probes, not encoded-file integrity tests.

Thus the tested context choices at this budget are **DFlash2 128K versus
experimental MTP 192K**, a 50% larger configured limit for MTP. The speed table
used 128K for both; increasing MTP's limit was a separate capacity test, not
a change to the timed comparison. 192K is a tested limit, not a search for
the absolute maximum. See [the capacity evidence](mtp-v2-context-probe.json).

The configured host remains on the already-tested DFlash2 128K profile in
local host commit `d93ce45a`. Both alternatives use a 28.18 GiB planning budget,
leaving about 2.12 GiB nominally outside it for media and other GPU allocations.
This is not an enforced GPU memory partition. No host switch was performed.

Raw local evidence is under
`benchmark-results/release-v032-v2-20261019/same-v2-{mtp,dflash}-{short,context}-performance`
and `experimental-v2-mtp-{metadata-02,192k-01}`. Private runtime preparation
and collection scripts are under
`build-dev/release-v032-v2-20261019/review-round-01/v2-mtp-probe`.
