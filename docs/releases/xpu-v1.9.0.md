# xpu-v1.9.0 — compact K4V2 on Brutus

This release adds the optional `kvarn_k4v2_g128_compact` cache format on
Intel Arc Pro B70, including native writing, decode, cached-prefill
reconstruction and Qwen bundled MTP support. Keys remain four-bit; values
use two-bit codes. Compact K4V4 remains available. The paired XPU changes
also support explicit BF16/FP16 reference caches and independently owned
K4V4 draft-cache metadata when the target uses K4V2.

GPU, model and final source-lock package qualification have passed. Performance comparisons and tuning use
**MTP off**. The retained throughput/decode floor is 0.95, the latency/TTFT
limit is 1.10, and the working throughput/decode target is within 3% of K4V4.
MTP functionality, quality and lifecycle checks pass separately.
Historical MTP speed differences do not determine release acceptance.

## Sources and package

The [release manifest](xpu-v1.9.0.json) records the released source pair, package
identities, evidence hashes and completed qualification checks.

Both upstream bases remain unchanged:

| Component | Retained upstream base | Released clean source |
| --- | --- | --- |
| vLLM | `51da0ca66c8065619c79e35dff97aa99aeaf5644` | `32289022a4e362b506a4de1f89e1abe2957660c1` |
| XPU kernels | `e7f20cbc18d741419a84aa57178e1f1cc0b453da` | `2c39287deec5e59cc2656e6d4a6dce51a567c411` |

The vLLM feature stack contains six signed-off commits, and the kernel
stack contains three. Production kernel sources match measured
`6c002b1a972b891d0256fecfd7527557d53b5299`. The final review also corrected a
static test that depended on C++ line wrapping, preserving its assertions and
checking both value widths. Original measured revisions and commit mappings
are retained. No nixpkgs, Torch, Triton, oneAPI or unrelated dependency update
is included.

The clean source pair builds through the pinned Brutus package overrides and
passes all 603 installed tests, with no failures or skips. The final allocation
repair preserves recurrent state for every scheduler slot when an explicit
attention-capacity override is used. Previously, an override of one full-context
request with four scheduler slots could reject startup because its budget
counted recurrent state for only one slot.

The occupied-capacity test then exposed a separate failure: two long requests
could fit the packed cache but exceed the shared FP16 materialization area during
MTP verification. The final repair reuses that area one request at a time only
when the combined history cannot fit. It preserves the buffer sizes, memory budget,
causal masks and request order. The before/after checks reproduce the failure on
the preceding package and pass after the repair, including independent GPU
attention, canaries and repeated buffer reuse in both formats.

Against the preceding package, 3,002 vLLM execution files and all 13 kernel package
files match byte for byte. The only changed vLLM files are version metadata and
the attention backend. The backend's normal execution is unchanged; its new
behavior is confined to an exception that previously always failed a DPAS request.
Native libraries retain their exact paths and bytes. Dependency comparison confirms
450 common closure paths, with three revision-labelled package/wrapper paths
replaced and equivalent wrapper contents.

This reviewed scope supports carrying the completed core performance, teacher
quality, fixed-allocation memory and full API results: their successful DPAS
requests could not have taken that exception. Original measured identities remain
recorded. The failed capacity result is preserved; the repaired final package
now passes the full occupied-capacity test and both MTP-off and MTP-on
long-context matrices. The actual released source locks resolve to the exact
qualified package and runtime, including both native library paths and hashes.
The runtime closure contains 453 paths. The content-addressed derivations resolve
to the already tested outputs; source publication introduces no runtime change.

## Selection and storage

Select the new format through the Nix service option:

```nix
kvCacheDtype = "kvarn_k4v2_g128_compact";
```

For the qualified MTP2 profile, the target uses K4V2 and the draft
explicitly uses `kvarn_k4v4_g128_compact`. Defaults are unchanged.

At G128/D256, including quantization metadata:

| Format | Bytes per block/head | Bytes per layer with four KV heads |
| --- | ---: | ---: |
| Compact K4V4 | 35,072 | 140,288 |
| Compact K4V2 | 26,880 | 107,520 |

K4V2 uses 23.36% less paged attention storage, allowing 30.48% more attention
tokens at a fixed attention-cache budget. Across sixteen target attention
layers, 262,144 logical tokens save 1.0 GiB; using K4V2 in all seventeen
target-plus-draft layers would save 1.0625 GiB. These are layout calculations.
Null pages, weights, recurrent state and FP16 pools are separate allocations.

The preceding package passed the fixed-allocation comparison with 262,144 usable
attention tokens, 2,049 physical pages per attention pool, four scheduler slots,
and MTP2 with a K4V4 draft. Four fresh starts used K4V4/K4V2/K4V2/K4V4 order;
each ran a 4K prompt with 256 output tokens, warmup and three measured repeats.
The allocated capacity was 262K; these device-memory peaks are from the 4K workload.

| Memory observation (GiB) | K4V4 | K4V2 target / K4V4 draft |
| --- | ---: | ---: |
| Actual attention tensor storage | 4.551049 | 3.550561 |
| Actual recurrent tensor storage | 1.887634 | 1.887634 |
| Actual FP16 tail tensor storage | 0.796875 | 0.796875 |
| Allocator live bytes after first execution | 26.913016 | 25.912925 |
| Allocator reserved bytes after first execution | 27.707031 | 26.707031 |
| Mean sampled device VRAM peak | 29.476748 | 28.476746 |

The measured attention-storage saving is **1,074,266,112 bytes**, including null
pages, consistent with 1 GiB across the usable pages. Storage and allocator
snapshots repeat exactly within each format. Device-residency peaks were sampled
every 0.5 seconds across the whole run, including startup; they are not
steady-state, per-request, or near-262K-workload peaks. The metadata observer was
used only for this memory comparison.

The final package passes the separate occupied-capacity test at the same 0.96
memory-utilization budget, with four scheduler slots and MTP2:

| Shared attention capacity | K4V4 | K4V2 target / K4V4 draft |
| --- | ---: | ---: |
| Usable tokens | 265,856 | 340,608 |
| Additional tokens | — | 74,752 (+28.12%) |

The K4V2 run completes two concurrent 151,039-token prompts with 512 generated
tokens each, preserving the required labels and arithmetic answer. Both prefills
finish more than 121 seconds before either request's final token. The scheduler
samples prove at least 302,976 simultaneously occupied attention tokens, exceeding
the entire K4V4 pool, with zero preemptions. Actual drafting and accepted-token
counters confirm MTP execution. This is a capacity and correctness check; its
fixed output cap and exceptional scratch reuse do not establish MTP speed parity.

The earlier all-K4V2 projection was approximately 346,880 shared tokens. The
qualified mixed format retains a K4V4 draft layer: its combined page cost is
1,860,608 bytes across sixteen K4V2 target layers and one K4V4 draft layer,
compared with 1,827,840 bytes if all seventeen used K4V2. The measured mixed pool
has 2,662 physical pages, including one null page, leaving 340,608 usable tokens.
Recurrent storage remains 2,026,831,872 bytes in both formats.

The per-request limit stays at 262,144 combined input/output tokens.
Concurrent requests share the available cache; the extra pool does not raise
that individual request limit.

## Quality and supported profile

The frozen model is `RedHatAI/Qwen3.8-27B-INT4`, revision
`bf08f3dbd9a324e53956920aad378a1f1b6dd24a`, with W4A16 weights and BF16
activations, eager V1 execution, TP1/PP1, four scheduler slots, 2,048 batched
prefill tokens, prefix caching disabled and a two-image limit.

K4V2 keeps eight full recent-history blocks (1,024 tokens) in FP16 during
decode, with a sixteen-block high-water mark. K4V4 retains its four/zero
policy. Both fit the existing sixteen-block prefill reservation, so the K4V2
policy does not allocate a larger FP16 pool.

Reducing value precision is lossy. The clean block-load package passed the
complete frozen BF16/K4V4 quality screen across eight cases and 8,448 unique
teacher-forced continuation positions per format and start. All eight captures
passed the original thresholds, with exact repeats across two independent
starts. The static-test correction preserves the measured execution code. The subsequent
allocation repair affects only an explicit capacity override, which the quality
screen does not use. The subsequent materialization repair changes only a previously
failing DPAS exception path. Both installed scopes are reviewed; the successful
quality captures retain their original identities.

Results below are weighted by continuation positions. The K4V2/K4V4 row uses
matched captures from the same package.

| Comparison | Top-1 agreement | Exact top-5 agreement | Mean top-5 Jaccard | Selected-token logit MAE |
| --- | ---: | ---: | ---: | ---: |
| K4V4 vs BF16 | 98.994% | 89.441% | 0.96412 | 0.08162 |
| K4V2 vs BF16 | 98.769% | 86.636% | 0.95460 | 0.11038 |
| K4V2 vs K4V4 | 98.923% | 86.683% | 0.95474 | 0.10727 |

The quality screen covers contexts through 65K with MTP off. Captured top-k
and selected-token logits do not establish full-distribution KL or broad task
accuracy. There is no same-GPU BF16 control for the required 128K and near-262K
model requests. Long-context retrieval, coding, tools, images and live MTP
behavior have separate release checks.

The preceding package passes the complete retained service suite in both formats
with MTP2 and a K4V4 draft: streaming text, reasoning, tool-call parsing, two
images, four concurrent requests, prefill during decode, cancellation and cache
reuse. Actual draft and accepted-token counters confirm MTP execution. Cancelling
after 3,073 tokens crosses the cache-flush boundary; the matching server abort,
idle scheduler, correct fresh response and repeated baseline token IDs verify
the retained lifecycle checks.

Both formats retrieve the correct label and ordered image colors at 131,071
and 262,015 prompt tokens. Each answer uses 12 completion tokens and finishes
normally. The generated merge-sort functions also pass 1,096 independent input
cases for each of four outputs per format. These are retained synthetic service
checks and one coding task, not broad task-accuracy benchmarks. The separate
MTP-off and MTP-on long-context matrices both pass with sustained 256-token
output.

## Profiling and validation

The profiling milestone is complete. Matched MTP-off model traces, native
replay, exact device code, register/resource inspection and qualified counters
identified resident FP16 page loads as a useful optimization target. The new
kernel uses block loads for aligned resident K/V pages and retains scalar
fallback for unaligned contiguous views. Packed storage, quantization,
arithmetic, history policy and reduction order are unchanged by this step.

The inspected device code comes from the exact attention library loaded by the
final package. Both producer kernels still allocate 256 GRFs and declare no
per-thread spill or scratch buffers. This static allocation does not establish
peak live registers or runtime spill traffic. The original resource and counter
captures retain their measured scope; unsupported register-liveness and stall-PC
claims are not inferred from these results.

The final block-load package passes **603 installed tests**, including
48 MTP verification cases, six alignment/residency cases, twenty compact-layout
and static checks, two explicit BF16/FP16 cache tests, and eighteen capacity
override cases covering matching and mixed formats and recurrent reservations,
plus twelve bounded-scratch CPU cases. The materializer GPU fixtures also cover
attention output with fitting and overflowing scratch, independent reconstructed
K/V, canaries and repeated reuse. The initial measured
package's reference-SDPA environment failure was resolved by restoring the exact
pinned OpenCL vendor path; failed artifacts are retained and no tolerance changed.

In unprofiled native comparisons at 16K/65K, K4V2 attention call spans take
39–51% less time than the previous K4V2 candidate, with bit-identical outputs
on those cells. These synthetic measurements include submission gaps and are
separate from the full-model results below. A variable short K4 control remains
recorded; the full-model K4 controls show no corresponding regression.

The complete MTP-off serving comparison passes every formal floor and the 3%
throughput/decode target at all six core anchors. It covers 96 waves and 240
streams in K4V4/K4V2/K4V2/K4V4 order, with two fresh starts per format and three
measured repetitions plus warmup at each anchor. Ratios below are K4V2/K4V4;
higher decode/throughput and lower TTFT/latency are better.

| Context / concurrent requests | Decode | Throughput | TTFT | Latency |
| --- | ---: | ---: | ---: | ---: |
| 4K / 1 | 1.0028 | 1.0011 | 1.0044 | 0.9990 |
| 4K / 4 | 1.0100 | 1.0077 | 0.9950 | 0.9916 |
| 16K / 1 | 1.0166 | 1.0069 | 1.0006 | 0.9931 |
| 16K / 4 | 1.0042 | 1.0032 | 0.9974 | 0.9967 |
| 65K / 1 | 1.0236 | 1.0044 | 0.9990 | 0.9956 |
| 65K / 4 | 1.0045 | 1.0027 | 0.9985 | 0.9975 |

These small positive medians establish measured parity in this profile; they
do not establish a universal speed advantage or statistical confidence. All
48 within-format wave output-multiset comparisons repeat and match the
previous candidate, including warmup. K4V2 decode improves 1.1–6.8% against
that candidate. The separate released K4V4 control also passes the retained
floors for both formats.

The final package also passes the complete MTP-off long-context comparison.
It covers 131,071 and 261,887 prompt tokens, one active request, 256 generated
tokens, and four fresh starts in the same balanced order. Each arm includes a
warmup and three measured repetitions per context: 32 completed streams in total.

| Context / concurrent requests | Decode | Throughput | TTFT | Latency |
| --- | ---: | ---: | ---: | ---: |
| 128K / 1 | 1.0246 | 1.0032 | 0.9985 | 0.9967 |
| Near 262K / 1 | 1.0365 | 1.0034 | 0.9979 | 0.9967 |

Both contexts pass the original formal limits and the 3% throughput/decode
targets. All streams have no retained quality findings, and matching output
token sequences repeat exactly between fresh starts of each format. Actual
arguments and request metrics confirm MTP is off; passive process captures
verify the qualified native libraries in all four starts. Within each format,
the observed sample range is at most 0.45% of the median across these metrics.
Prefill dominates total request time, so the decode gains translate to about
0.3% higher end-to-end output throughput. These results apply to the measured
profile and do not establish a universal speed advantage or broad task accuracy.

The full MTP2 matrix also passes at the same two context lengths, with the same
256-token outputs, warmups, three measured repetitions and four fresh starts.
All 32 streams complete without retained quality findings. Every wave has valid
two-token drafting and positive accepted-token counts, and both formats repeat
identical output token sequences between starts. All four loaded library
identities match the qualified package. Both formats use an explicit K4V4 draft.

The following K4V2/K4V4 timing ratios are descriptive MTP-on observations:

| Context / concurrent requests | Decode | Throughput | TTFT | Latency |
| --- | ---: | ---: | ---: | ---: |
| 128K / 1 | 1.1054 | 1.0080 | 0.9985 | 0.9920 |
| Near 262K / 1 | 1.0434 | 1.0035 | 0.9980 | 0.9965 |

The largest within-format sample range is 1.60% of its median. Draft acceptance
and verification work differ between formats, so these observations neither
isolate native-kernel performance nor replace the MTP-off release gates above.

Packaging validation passes all 563 tests across the repository's 45 test
files, with no skips or failures. The complete existing per-file test loop uses
the clean vLLM interpreter for forced decode and retains the separate
quantization environment. Changed Nix file formatting/parsing and the existing
kernel-library cache identity, binding cache identity, build parallelism and
repository formatting checks also pass. NixOS module evaluation verifies the
new option and explicit K4V4 draft setting, with defaults unchanged. These
checks are complemented by final verification of the released source locks,
unchanged unrelated dependencies and exact qualified package identity.

## Release references

All three repositories use the annotated `xpu-v1.9.0` tag and permanent
`releases/xpu-v1.9.0` branch. The packaging lock changes only
`vllm-xpu-unstable-src` and `vllm-xpu-kernels-unstable-src`, resolving to the
source pair above. The manifest records the resolved package derivations and
the checksummed final lock verification.

Raw captures, failed candidates, the previous draft and execution checkpoints
are retained privately under `benchmark-results/kvarn-k4v2/20260913T051252Z/`.
Chat and embedding services remain operator-stopped. Host deployment and
service activation are outside this release goal.
