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
extension contains the baseline and every compatible round specialization.
At engine initialization the host resolves a named cache layout, named kernel
variant, and split policy, freezes them on the attention implementation, and
passes their explicit IDs plus the exact split count to every native operator.
The C++ hot path does not read environment variables. A factory launcher may
set the names, but changing the cache layout always requires a fresh engine and
fresh cache allocation.

The public beta remains simple: selecting a Kvarn KV-cache dtype is sufficient
to enable the conservative natural-layout/reference implementation. Factory
selectors are optional overrides for B70 experiments, not prerequisites for
using Kvarn.

The round-1 library exposes this frozen engine-start selection surface. Kernel
variants and split counts do not require rebuilding the extension:

| Selector | Values in the combined build | Lifetime |
|---|---|---|
| `KVARN_NATIVE_XPU_CACHE_LAYOUT` | `natural`, `xe2_dpas` | engine/cache ABI; restart and allocate a fresh cache to change |
| `KVARN_NATIVE_XPU_KERNEL_VARIANT` | `baseline`, `qk_i8u4`, `q6_scalar`, `q8_vector`, `q6_vector` | frozen at engine initialization |
| `KVARN_NATIVE_XPU_SPLITS` | `1`, `2`, `4`, `8`, `16`, `17`, `24`, `32` | frozen maximum at engine initialization |

The direct B70 factory runner bypasses service startup and passes the same
explicit variant ID, layout bit, and split count to the operator, allowing all
compatible cells to be swept in one process. Variant IDs are `0` through `4`
in the order listed above. ID `5` is reserved for the page-128 experiment and
fails closed in the round-1 library because that specialization is not ready.

Build `.#vllm-xpu-kvarn-factory` for the complete round-1 matrix. It compiles
all five decode variants and the fused-QKV operator into one BMG-AOT attention
library. Runtime selection therefore does not start another Nix build. The
package also freezes the generated upstream FA2 buildout to Brutus's text-only
Qwen3.8 profile: two head-dimension-256 chunk-prefill policies and one
qgroup-8, block-64 paged-decode policy. This reduces the attention target from
663 Ninja actions to about 12 while retaining matched auto and Kvarn paths.

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

Rows are appended from immutable JSON artifacts; `pending` is not a zero.

| Variant | Correctness | B1 throughput | B1 decode tok/s | B4 throughput | B4 decode tok/s | p99 TTFT | p99 ITL | XPU kernel time | launches | idle | Decision |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `auto-control` | control | pending | pending | pending | pending | pending | pending | pending | pending | pending | performance control |
| `natural-oracle` | primitive reference | pending | pending | pending | pending | pending | pending | pending | pending | pending | correctness only |
| `r1-p0-dpas-q8-t64` | primitive pass at 262K | pending | pending | pending | pending | pending | pending | pending | pending | pending | round baseline |
| `r1-p1-dpas-qk-i8u4` | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| `r1-p2-dpas-q6-t64` | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| `r1-p3-dpas-fused-qkv-store` | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| `r1-p4-dpas-bucket-split` | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| `r1-p5-dpas-vector-load` | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |
| `r1-p6-dpas-page128` | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending | pending |

Final promotion still requires native Kvarn statistical parity or better with
auto on the B70: paired ratio at least 98%, hard throughput and per-request
decode floor at least 95%, p99 TTFT/ITL no more than 110% of auto, and service
correctness through 262K.
