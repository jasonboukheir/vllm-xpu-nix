# Kvarn Xe2 optimization factory

This document is the compact operating record for Kvarn decode optimization on
the Intel Arc Pro B70. It complements the formal service gates in
`kvarn-native-xpu-gates.md`; it does not weaken them.

## Fixed controls and evidence rules

- `auto` KV cache is the performance control.
- Natural-layout Kvarn is the numerical/correctness oracle.
- Only measurements executed by XPU kernels on exactly
  `Intel(R) Arc(TM) Pro B70 Graphics` count as performance evidence. CPU runs
  are smoke tests only.
- Auto and Kvarn use the same model revision, prompts, input/output lengths,
  batch/concurrency, eager/graph mode, scheduler limits, quantization, service
  process, warmups, and build closure except where the KV implementation
  necessarily differs.
- Cache layout is selected once before allocation and is immutable for the
  engine lifetime. A writer and every reader/materializer must agree on the
  layout identifier.
- Each kernel strategy is a separate host-dispatched compile-time
  specialization. Avoid device-side feature tangles.
- Primitive results eliminate candidates. Only finalists pay for complete
  service correctness and statistical ABBA runs.

Every artifact records these axes explicitly:

| Axis | Required examples |
|---|---|
| `variant_id` | `r1-p2-dpas-q6-t64` |
| `cache_layout` | `natural`, `xe2_dpas` |
| `kernel_strategy` | `q8_bf16_dpas`, `q6_bf16_dpas` |
| `split_policy` | `fixed16`, `fixed24`, `b70_context_bucket_v1` |
| `fusion_strategy` | `reduce_h256`, `qkv_store_reduce_h256` |
| `scheduling_variant` | `tile64`, `page128` |
| provenance | source commits/diffs, derivations and closures, exact launcher |
| workload | model revision, B1/B4, context, output length, seed, all serve args |
| hardware | exact device name plus a successful candidate XPU tensor operation |

The expensive build is the round boundary, not the variant boundary. One
extension contains the baseline and every layout-compatible round
specialization. The native dispatch ABI receives explicit layout, variant, and
split values, so the direct factory can change reader and scheduling candidates
between calls without rebuilding or re-packing the cache. The service resolves
named selectors once at startup for reproducibility, but a batch-aware split
policy may choose an effective split count at each decode call from scratch
allocated for the declared maximum. The C++ hot path does not read environment
variables. Cache layout is the exception: changing it always requires a fresh
engine and fresh cache allocation.

The public beta remains simple: selecting a Kvarn KV-cache dtype is sufficient
to enable the conservative natural-layout/reference implementation. Factory
selectors are optional overrides for B70 experiments, not prerequisites for
using Kvarn.

The round-1 library exposes this frozen engine-start selection surface. Kernel
variants and split counts do not require rebuilding the extension:

| Selector | Values in the combined build | Lifetime |
|---|---|---|
| `KVARN_NATIVE_XPU_CACHE_LAYOUT` | `natural`, `xe2_dpas` | engine/cache ABI; restart and allocate a fresh cache to change |
| `KVARN_NATIVE_XPU_KERNEL_VARIANT` | `baseline`, `qk_i8u4`, `q6_scalar`, `q8_vector`, `q6_vector`, `q6_cached_weights`, `q6_exact_rows`, `q6_cached_weights_exact_rows`, `q6_page_pair`, `q6_main_grf128`, `q6_split_reducer_specialized`, `q6_next_page_prefetch` | startup selector; every listed specialization is in the same library |
| `KVARN_NATIVE_XPU_SPLIT_POLICY` | `fixed`, `b70_q6` | startup policy; `b70_q6` selects the effective count per decode batch |
| `KVARN_NATIVE_XPU_SPLITS` | `1`, `2`, `4`, `8`, `16`, `17`, `24`, `32` | scratch-allocation maximum; effective count may be selected per call |

`b70_q6` allocates for 32 and selects B1=32, B2=16, B3--4=8,
B5--8=4, and B9--12=2. It is valid only with a Q6 DPAS reader. The named
policy and `KVARN_NATIVE_XPU_SPLITS` are mutually exclusive so a launcher
cannot present two different scheduling contracts.

The direct B70 factory runner bypasses service startup and passes the same
explicit variant ID, layout bit, and split count to the operator, allowing all
compatible cells to be swept in one process. `--output-dtypes fp16,bf16`
likewise exercises both output paths from that same binary; production BF16 is
the default. Round-1 variant IDs are `0` through `4` in the order listed above;
ID `5` is reserved for the page-128 experiment and fails closed. Round-2 IDs
`6`, `7`, and `8` are cached weights, exact rows, and their combination. The
subsequent independently dispatched experiments are ID `9` page-pair, ID `10`
GRF128 main kernel, ID `11` specialized split reducer, and ID `12` next-page
prefetch.

Build `.#vllm-xpu-kvarn-factory` for the complete current-layout matrix. It
compiles every implemented Round-1 and Round-2 decode specialization plus the
fused-QKV operator into one BMG-AOT attention library. Runtime selection
therefore does not start another Nix build. The package also freezes the
generated upstream FA2 buildout to Brutus's text-only Qwen3.8 profile: two
head-dimension-256 chunk-prefill policies and one qgroup-8, block-64
paged-decode policy. This reduces the attention target from 663 Ninja actions
to about 12 while retaining matched auto and Kvarn paths.

That partial buildout is deliberately fail-narrow. It is valid for the frozen
eager, no-MTP, no-prefix-cache, no-DCP Brutus profile and requires the startup
log to report block size 64. A different model, effective block size, prefix
caching/cascade attention, MTP, DCP, multimodal attention, sliding windows, or
attention sinks requires a broader kernel package. The other split kernel
libraries remain present in this round; pruning unrelated feature libraries is
kept separate from kernel-performance experiments.

The direct runner defaults to `--fixture-mode matched-production` and likewise
requires auto block size 64. It creates one deterministic logical BF16 K/V
corpus, writes that corpus into the auto cache with the production cache op,
and derives both natural and DPAS Kvarn records with the production Hadamard,
Sinkhorn, RTN, and packing paths. Sink pages and each current page remain in
their production-style FP16 tail slots. Corpus construction, validation, and
hashing occur before timing. The old unrelated random payloads are available
only through explicit `--fixture-mode unmatched-diagnostic` and can never
produce matched-ratio evidence.

A matched primitive ratio is still candidate-ranking evidence, not service
parity. Each case must first pass candidate-versus-natural and quantized
natural-versus-auto output checks. Full model execution, scheduler behavior,
page flushing, TTFT, ITL, and service throughput remain finalist gates.

Run the complete matched primitive matrix from the repository root after the
vLLM service has been stopped:

```console
nix run .#kvarn-factory -- \
  --vllm-xpu-nix-repo "$PWD" \
  --vllm-repo /tmp/vllm-kvarn-upstream-sync \
  --kernels-repo /tmp/vllm-xpu-kernels-upstream-sync
```

This realizes one BMG-AOT package, then tests every selected kernel variant and
the configured split sweep in one pinned Python/Torch/XPU process. The default
`--variants all` is literal: it runs every compiled, runnable ID (`0`--`4` and
`6`--`12`) at split 8 and 32 with direct BF16 output. Use an explicit named
comma-separated shortlist when a smaller sweep is intended. Sixteen warmup
rounds precede twenty
measured rounds per arm because the first B70 sweep showed material
short-context clock settling after four warmups. Override `--variants`,
`--splits`, `--output-dtypes`, `--warmup-rounds`, or `--sample-rounds` to change
the runtime matrix without rebuilding the native library. Pytest is included
for the mandatory native kill suite, and inherited Python, loader, service,
and Kvarn-selector variables are scrubbed before launch. The launcher
refuses to run beside a vLLM service, against dirty or source-mismatched
repositories, with ambiguous shared libraries, or without exact Nix
derivation and closure attestations. Evidence is written atomically under
`benchmark-results/kvarn/`; `/tmp` and overwrites are rejected.

## Evidence entering round 1

The direct-BF16 B70 device-stage checkpoint in
`benchmark-results/kvarn/20260904T011700Z-direct-bf16-device-stages` found:

| Context / batch | Natural reader | DPAS reader | DPAS gain |
|---|---:|---:|---:|
| 4K / B1 | 92.9 us | 71.3 us | 1.30x |
| 4K / B4 | 293.5 us | 165.1 us | 1.78x |
| 16K / B1 | 251.7 us | 146.5 us | 1.72x |
| 16K / B4 | 1341.7 us | 677.6 us | 1.98x |
| 65K / B1 | 1266.7 us | 601.9 us | 2.10x |
| 65K / B4 | 5393.0 us | 2569.5 us | 2.10x |

At 65K, the matched dense-BF16 auto reader checkpoint was 526.2 us for B1
and 1896.6 us for B4. DPAS therefore closes most of the gap, but the reader
alone remains about 14% slower for B1 and 35% slower for B4. Kvarn query and
output rotations add about 120 us for B1 and 150 us for B4 at this point.
These are isolated device stages, not service-parity claims.

The byte-rate points away from raw HBM bandwidth as the main remaining
bottleneck. A 65,023-token Kvarn read covers approximately 71.3 MB per request
(508 pages x 4 KV heads x 35,072 bytes), or about 118 GB/s at the measured B1
time and 111 GB/s at B4. Dense BF16 auto moves approximately 266 MB per request
and reaches roughly 506–561 GB/s in the same checkpoint. This makes packed
unpack/conversion instructions and matrix utilization first-round targets;
merely reducing already-compressed memory traffic is unlikely to close parity.

The existing B70 split sweep also showed that fixed split count is leaving
performance on the table: B4/65K improved from 3233 us at split 16 to 2975 us
at split 24 in that checkpoint, while B4/4K favored split 16. This justifies a
context-bucket scheduling candidate.

Transferable ideas used for ranking:

- FlashInfer plans split-K from device occupancy, batch/GQA shape, page size,
  and available grid capacity, and reuses planned auxiliary data across
  layers.
- FlashAttention inference divides KV loading across blocks and combines
  partials separately; its KV-cache interface also demonstrates useful
  fusion boundaries around rotary/update work.
- TensorRT-LLM enables multi-block decode only after occupancy and token-count
  thresholds, rather than treating one split policy as universal.
- vLLM's ROCm FlyDSL TurboQuant decode keeps online softmax and PV in the same
  kernel, caps partitions, uses persistent scratch, and arranges V in the
  matrix engine's native transposed layout. These are architectural patterns,
  not code to translate literally.

Primary implementation references:

- <https://github.com/flashinfer-ai/flashinfer/blob/main/include/flashinfer/attention/scheduler.cuh>
- <https://github.com/Dao-AILab/flash-attention/blob/main/csrc/flash_attn/src/flash_fwd_launch_template.h>
- <https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/legacy/advanced/gpt-attention.md>
- `vllm/v1/attention/ops/flydsl_turboquant_decode.py` in the integrated vLLM
  source.

## Round 1 ranked variants

All compatible specializations are emitted by one XPU-kernel build and
selected by the host before launch. The combined decode dispatch matrix is
`q8-scalar` (ID 0), integer-QK (ID 1), `q6-scalar` (ID 2), `q8-vector`
(ID 3), and `q6-vector` (ID 4). The fused-QKV front end is a separate operator
in that same extension, so it can be timed independently without confounding
the decode variants. Page-128 remains reserved and unimplemented. `r1-p0` is
the current DPAS baseline, not a promotion candidate by itself.

| Rank / ID | Isolated change | Expected impact | Cost | Risk | Fast decision |
|---|---|---:|---:|---:|---|
| `r1-p1-dpas-qk-i8u4` | Quantize page-scaled Q to int8 and execute QK directly against packed uint4 codes; leave PV and every other boundary unchanged | 20–45% if BF16 conversion/DPAS dominates | high | high numerical risk | compare QK logits/output against q8 at adversarial pages first; kill quickly on drift |
| `r1-p2-dpas-q6-t64` | Replace the q-packed-8 Xe DPAS tile with an exact GQA-6 repeat/tile | 10–25% core work | medium | medium | compile, ragged/long correctness, then B1/B4 device time |
| `r1-p3-dpas-fused-qkv-store` | Extend the existing K/V H256 scatter launch with independent Q-head workgroups that write rotated Q; do not repeat H256 in every decode split | one launch and much of the measured 60–85 us Q-transform stage, strongest at B1 | medium | medium | exact transformed-Q/K/V boundary compare, then launch/time delta |
| `r1-p4-dpas-bucket-split` | Select split 16/24 from B70 batch/context buckets while retaining the same producer/reducer kernels | up to ~8% from existing B4/65K evidence | low | low | sweep 4K/16K/65K B1/B4 together |
| `r1-p5-dpas-vector-load` | Vector-load each aligned per-lane packed K/V fragment; leave arithmetic unchanged | 5–15% if scalar load/unpack issue-bound | low–medium | low | assembly/kernel-time check plus exact primitive compare |
| `r1-p6-dpas-page128` | One physical page per work iteration with a corrected ReduceK=8/page-128 epilogue | 5–20% through metadata reuse and fewer loop/schedule steps | medium–high | high | known ReduceK=8 boundary/ragged cases first; kill on any mismatch |
| `r1-p0-dpas-q8-t64` | Current Xe2 DPAS-native layout and q-packed-8/tile-64 implementation | baseline | complete | known-correct primitive | control for every primitive sweep |

Keep these next-round experiments ranked but out of round 1 so attribution
stays clear:

1. PV-only uint8 x uint4 DPAS probability/value accumulation, independently
   selectable and correctness-eliminated before combination with integer QK.
2. Prefetched/double-buffered packed K/V tiles after the vector-load result
   shows whether the current kernel is latency- or instruction-bound.
3. Persistent decode across heads/layers only if the XPU trace shows queue
   starvation rather than dominant device work.

## Round 1 B70 result

The sealed one-build factory run
`benchmark-results/kvarn/factory-b70-20260904T060939Z.json` used project
`7b0c7a872a48053aae9a8459e459c607ddd4ffbd`, vLLM
`0d72e5b102a1b5cde06ee97f1ae8304efe9a3dbb`, and kernels
`3df85eecd749fbe4a8b10cd1223c5925b2e765e7`. The top-level package and both
native libraries were independently bound to their real Nix derivers and
closure digests. The hardware preflight named exactly one Arc Pro B70 and
completed a real XPU tensor operation.

The mandatory native kill suite passed 41 tests with zero skips, including
ragged batches, page boundaries, FP16 tails, all round-1 variants, fused QKV,
and 262K addressing. The matched-production fixture then completed all 120
deduplicated B1/B4, context, split, and variant cells. Every timed cell passed
candidate-versus-natural and quantized-natural-versus-auto correctness. The
following ratios are unprofiled XPU event medians; higher is better and 100%
means equal device time to auto. They rank primitives only and are not service
parity evidence.

| Context / batch | Winning variant | Split | Candidate | Auto | Decode performance | Fused device-stage performance |
|---|---|---:|---:|---:|---:|---:|
| 4K / B1 | `q6_scalar` | 32 | 81.9 us | 77.6 us | 94.8% | 97.0% |
| 4K / B4 | `q6_vector` (observed) | 8 | 141.5 us | 162.4 us | 114.8% | 110.1% |
| 16K / B1 | `q6_scalar` | 32 | 156.7 us | 190.7 us | 121.7% | 115.5% |
| 16K / B4 | `q6_scalar` | 8 | 342.3 us | 531.8 us | 155.4% | 147.2% |
| 65K / B1 | `q6_scalar` | 32 | 339.7 us | 521.0 us | 153.4% | 145.5% |
| 65K / B4 | `q6_scalar` | 8 | 1205.2 us | 1888.9 us | 156.7% | 154.2% |

For one coherent service candidate, `q6_scalar` wins five of six raw medians
and is also correct in the sixth: its best decode performance is 94.8%, 104.5%,
121.7%, 155.4%, 153.4%, and 156.7% in table order. Its geometric mean across
those six cells is 128.5% of auto. `q6_vector` is retained as a useful
short-context B4 runner-up, where it beats scalar, but trails scalar by roughly
1--5% elsewhere. The apparent 4K/B4 vector win is not yet promotable: the
short-context candidates continued warming during their recorded samples, and
the scalar last-half median beat the vector last-half median. One fully warmed
confirmation is required before retaining a context-specific vector branch.
The DPAS q8 baseline, integer-QK, and q8 vector candidates
are eliminated from promotion: their best-per-cell geometric means are 70.2%,
58.3%, and 68.3% of auto respectively.

Round-1 promotion is therefore:

- kernel strategy: `q6_scalar`;
- split policy: fixed 32 for B1 and fixed 8 for B4 pending a service-level
  context-bucket result;
- fusion strategy: retain fused QKV because it raises the short B1 floor from
  94.8% for decode alone to 97.0% for the measured fused device stage;
- runner-up: retain `q6_vector` only until the 4K/B4 service cell resolves
  whether its primitive advantage survives end to end;
- dead experiments: do not carry integer-QK or either q8 DPAS variant into a
  finalist service matrix. Natural layout remains the correctness oracle.

Two gaps must be closed before service promotion. The production BF16 model
selects the native decoder's direct BF16 output path, while this factory round
timed the FP16 output path. Output dtype is now an explicit runtime factory
axis, and the finalist confirmation must run BF16. Also, production currently
freezes one split count for
the whole engine, whereas this round consistently wants B1 split 32 and B4
split 8. The next service candidate must allocate for 32 splits and select a
batch-aware `B * splits = 32` policy at decode time; this scheduling choice
does not change the immutable cache-layout ABI.

## Round 2 one-build queue

Round 2 preserves ID 2 byte-for-byte as the Q6 control and compiles new,
independently dispatched IDs into the same Xe2 attention library. The ranked
queue is deliberately wider than one conservative edit:

1. Cache the four Q6 split-reduction softmax weights once per row instead of
   recomputing global max and `exp2` weights for every output-fragment value.
2. Remove loops over the two padded Q8 rows where the compiler has not already
   eliminated them; kill this candidate immediately if generated code and XPU
   time are flat.
3. Reuse scaled Q and page metadata across both 64-token halves of each packed
   128-token record.
4. Stage scaled Q and page-constant K/V metadata cooperatively in workgroup
   memory, isolated from the page-pair experiment until each effect is known.
5. Replace the Q6 epilogue's coordinate scalar output writes with a specialized
   two-row plus one-row block-message path.
6. Prefetch the next packed page only as an isolated long-context candidate;
   reject it if extra GRF/SLM pressure reduces occupancy.

The page-128 topology remains a higher-risk candidate after these. Persistent
decode and a new integer-DPAS cache ABI stay parked until profiling justifies
their complexity. Existing measurements already disprove integer QK over the
current FP16-DPAS-oriented record layout; reviving it requires a distinct
engine-lifetime layout, not a reader flag.

## Round loop and elimination gates

1. Build all round specializations in one package and record its exact source
   and Nix closure.
2. Run every variant against natural Kvarn on the B70 for packed pages and
   FP16 tails, ragged B4, empty/partial/full pages, split boundaries, high
   physical block indices, and 262K addressing. Eliminate any failure.
3. Sweep survivors together at B1/B4 and 4K/16K/65K with identical warmed
   settings. This is a primitive/diagnostic leaderboard, not final parity.
4. Capture separate Kineto XPU traces for likely winners. Report device kernel
   time, launch count, device busy span, and idle fraction. Profiled throughput
   never enters acceptance.
5. Promote at most the best coherent strategy plus useful orthogonal runners
   up. Remove dead specializations before starting the next round.
6. Run full 262K service correctness and 4K–65K ABBA only for finalists.

## Compact leaderboard

The runner writes `primitive_leaderboard` after every completed cell and prints
the final compact ordering. Each row contains only XPU-event measurements the
primitive runner owns: candidate and auto decode device-median ranges,
candidate/auto latency and speed ratios, the separately measured device-stage
ratios, case counts, and primitive correctness status. It does not reserve
columns for service throughput, decode tok/s, TTFT, ITL, launch count, or idle
fraction. Those metrics appear only after their dedicated service or profiler
runners actually measure them.

Ratio directions are explicit. `candidate_latency_over_auto` and
`auto_speed_over_candidate` are numerically equal and are lower-is-better for
the candidate; `candidate_speed_over_auto` is their reciprocal and is
higher-is-better. Legacy `candidate_*_over_auto` latency aliases remain in each
case for existing consumers. The generated leaderboard ranks a complete fused
device-stage measurement when available, otherwise the separate device stage.
That rank is a screening decision signal only, never a service-parity result.

The sealed round-1 artifact produced this historical primitive-only decision
record:

| Variant | Primitive correctness | Measured XPU device-stage evidence | Primitive decision |
|---|---|---|---|
| `auto-control` | control | measured per matched cell | performance control |
| `natural-oracle` | 41/41 suite; reference | measured per cell | correctness only |
| `r1-p0-dpas-q8-t64` | pass through 262K | 105--2572 us decode; 70.2% geometric-mean performance | eliminate |
| `r1-p1-dpas-qk-i8u4` | pass through 262K | 128--3302 us decode; 58.3% geometric-mean performance | eliminate |
| `r1-p2-dpas-q6-t64` | pass through 262K | 81.9--1205 us decode; 128.5% geometric mean | primitive finalist; B1 split32/B4 split8 |
| `r1-p3-dpas-fused-qkv-store` | 41/41 suite; exact boundary | 97.0--154.2% of auto with winning reader | retain for finalist service testing |
| `r1-p4-dpas-bucket-split` | all split cells pass | B1 split32/B4 split8 were the primitive winners | retain; validate service buckets |
| `r1-p5-dpas-vector-load` | pass through 262K | 81.2--1259 us with q6 | runner-up for 4K/B4 only |
| `r1-p6-dpas-page128` | not compiled | no measurement | defer |

Final promotion still requires native Kvarn statistical parity or better with
auto on the B70: paired ratio at least 98%, hard throughput and per-request
decode floor at least 95%, p99 TTFT/ITL no more than 110% of auto, and service
correctness through 262K.
