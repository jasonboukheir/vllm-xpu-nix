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
| `scheduling_variant` | `tile64`, `tile64_next_page_prefetch`, `tile64_next_page_current_half_v_prefetch`, `tile64_next_page_prefetch_record_cursor` |
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

The combined factory library exposes this frozen engine-start selection
surface. Kernel variants and split counts do not require rebuilding the
extension:

| Selector | Values in the combined build | Lifetime |
|---|---|---|
| `KVARN_NATIVE_XPU_CACHE_LAYOUT` | `natural`, `xe2_dpas` | engine/cache ABI; restart and allocate a fresh cache to change |
| `KVARN_NATIVE_XPU_KERNEL_VARIANT` | `baseline`, `qk_i8u4`, `q6_scalar`, `q8_vector`, `q6_vector`, `q6_cached_weights`, `q6_exact_rows`, `q6_cached_weights_exact_rows`, `q6_page_pair`, `q6_main_grf128`, `q6_split_reducer_specialized`, `q6_next_page_prefetch`, `q6_next_page_prefetch_split_reducer`, `q6_simd_unpack`, `q6_block_output_store`, `q6_current_half_v_prefetch`, `q6_page_record_cursor`, `q6_prefetch_record_cursor` | startup selector; every listed specialization is in the same library |
| `KVARN_NATIVE_XPU_SPLIT_POLICY` | `fixed`, `b70_q6`, `b70_q6_v2` | startup policy; named policies select the effective count per decode call |
| `KVARN_NATIVE_XPU_SPLITS` | `1`, `2`, `4`, `8`, `16`, `17`, `24`, `32` | scratch-allocation maximum; effective count may be selected per call |
| `KVARN_FLUSH_WRITER` | `reference`, `native_xe2` | startup writer; `native_xe2` requires the `xe2_dpas` D256/G128/K4V4/Hkv4 cache ABI |
| `KVARN_NATIVE_XPU_PREFILL_STORE` | `reference`, `hadamard_scatter` | startup multi-token store; unsupported calls fall back to the reference path |

The writer/store selectors are orthogonal to the reader ID. `reference` remains
the public-beta default for both. `native_xe2` replaces the completed-page
Sinkhorn/RTN packer with the native Xe2 balanced-record writer, while
`hadamard_scatter` replaces eligible pure multi-token prefill scatters. Neither
selector permits changing the cache layout after allocation. The factory
harness records both selectors independently so a winning reader is not
mistaken for a writer or prefill-store gain.

`b70_q6` allocates for 32 and selects B1=32, B2=16, B3--4=8,
B5--8=4, and B9--12=2. It is valid only with a Q6 DPAS reader. The named
policy and `KVARN_NATIVE_XPU_SPLITS` are mutually exclusive so a launcher
cannot present two different scheduling contracts.

`b70_q6_v2` is an exploratory, context-aware policy for
`q6_next_page_prefetch` (ID12) and
`q6_next_page_prefetch_split_reducer` (ID13). It allocates scratch for 32 splits,
selects B1=32 at every context, and selects B4=8 through 48 Ki tokens
(49,152 inclusive) or B4=32 above that boundary. It is available only through
the runtime-factory launcher; the historical immutable launchers remain bound
to `b70_q6`.

Harness artifacts record a versioned `native_split_policy_contract` containing
the selection axes, inclusive context bounds, exact rules, scratch ceiling,
and kernel compatibility. `native_nominal_splits_by_batch` is deliberately
`null` for `b70_q6_v2`; each performance/profile workload instead records its
resolved effective split count. This prevents a B4=8 nominal map from hiding
the B4=32 long-context behavior.

The direct B70 factory runner bypasses service startup and passes the same
explicit variant ID, layout bit, and split count to the operator, allowing all
compatible cells to be swept in one process. `--output-dtypes fp16,bf16`
likewise exercises both output paths from that same binary; production BF16 is
the default. Round-1 variant IDs are `0` through `4` in the order listed above;
ID `5` is reserved for the page-128 experiment and fails closed. Round-2 IDs
`6`, `7`, and `8` are cached weights, exact rows, and their combination. The
subsequent independently dispatched experiments are ID `9` page-pair, ID `10`
GRF128 main kernel, ID `11` specialized split reducer, and ID `12` next-page
prefetch. IDs `13` and `14` are independently assigned to the combined
prefetch/reducer and SIMD-unpack experiments; ID `15` isolates two-row plus
one-row block-2D output stores from all of those changes. IDs `16` through `18`
retain ID13 as the reader control and independently select current-half V
prefetch, page-record address reuse, or their composition.

Build `.#vllm-xpu-kvarn-factory` for the complete current-layout matrix. It
compiles every implemented decode specialization through ID18 plus the
fused-QKV operator into one BMG-AOT attention library. Runtime selection
therefore does not start another Nix build. The package also freezes the
generated upstream FA2 buildout to Brutus's text-only Qwen3.8 profile: two
head-dimension-256 chunk-prefill policies and one qgroup-8, block-64
paged-decode policy. This reduces the attention target from 663 Ninja actions
to about 12 while retaining matched auto and Kvarn paths.

### One package-free service launcher

Config branch `feature/kvarn-runtime-factory` at revision
`c3bfa82fe29ac598c63301fedcc6e04ccf56e547` exposes
`vllm-xpu-brutus-kvarn-factory-runtime`. The launcher contains no pinned vLLM
or attention-library store reference. It accepts the already-built candidate
package as its first argument, validates all `KVARN_FACTORY_*` selectors, then
translates them into the engine environment and canonical serve arguments.
Use it through the harness rather than invoking it by hand:

```console
scripts/kvarn_perf_run.py \
  --launcher-mode runtime-factory \
  --candidate-env /nix/store/CURRENT-FACTORY-CANDIDATE \
  --native-layout xe2_dpas \
  --native-kernel-variant q6_next_page_prefetch \
  --native-split-policy b70_q6_v2 \
  --flush-writer native_xe2 \
  --prefill-store hadamard_scatter \
  ...
```

The same `--launcher-mode runtime-factory` switch is supported by
`kvarn_correctness_run.py` and `kvarn_xpu_profile.py`. It makes cache dtype,
layout, kernel, split policy/count, frontend, flush-index strategy, full-page
writer, multi-token prefill store, oneDNN mode,
request-stable projections/RMSNorm, 65K/262K model length, and B1/B4 width
process-start choices. The launcher is
resolved to one immutable Nix-store program once per harness run, while every
service-start record contains the exact selector map. `KVARN_FACTORY_SPLITS`
is recorded as `null` and omitted from the process environment when either
named policy owns split selection.

Use `b70_q6_v2` first with `kvarn_xpu_profile.py` or
`kvarn_perf_run.py --exploratory`. The correctness runner supports the same
runtime-factory selector and records the complete policy contract across its
65K and 262K phases. Formal performance gates are unchanged: they still
require the full correctness manifest, matched factory qualification, exact
candidate identity, B70 execution proof, and the existing throughput/latency
thresholds.

Two additional strict selectors isolate the model-wide request-stability costs
without rebuilding the candidate:

| Factory selector | Child-process selector | Default | Effect when `0` |
| --- | --- | --- | --- |
| `KVARN_FACTORY_REQUEST_STABLE_PROJECTION_ROWS` | `KVARN_REQUEST_STABLE_PROJECTION_ROWS` | `1` | Use ordinary model and logits projection dispatch. |
| `KVARN_FACTORY_REQUEST_STABLE_RMSNORM` | `KVARN_REQUEST_STABLE_RMSNORM` | `1` | Use ordinary Gemma RMSNorm dispatch, including the fused XPU path when available. |

Both factory selectors accept exactly `0` or `1`. They are independent from
each other and from `KVARN_FACTORY_ONEDNN_DETERMINISTIC`, and all three values
are recorded in the selector map, captured engine environment, profile, and
sealed result provenance. The qualified default is `1` for both new axes.
An opt-out is diagnostic: profiling and exploratory performance runs may use
it immediately, but formal parity accepts it only when a correctness artifact
from the same candidate records the identical selectors and has passed the
full replay/restart/isolation/262K suite. The immutable named-launcher path
keeps the qualified defaults; opt-outs require `--launcher-mode
runtime-factory`.

The historical `immutable` launcher mode remains the default for compatibility.
The correctness runner has one explicit exception in runtime-factory mode: its
262K natural-layout, non-native compact-Kvarn oracle still uses the immutable
reference launcher because the package-free config app intentionally supports
only auto or native Kvarn. That oracle remains oneDNN-deterministic even when a
runtime-factory candidate selects the diagnostic non-deterministic mode. This
preserves the exact-token equivalence gate while recording the difference.
Trailing `--kv-cache-dtype`, `--max-model-len`, or `--max-num-seqs` arguments
are rejected because those values belong exclusively to the runtime selectors.

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
  --kernels-repo /tmp/vllm-xpu-kernels-upstream-sync \
  --flush-writer native_xe2 \
  --prefill-store hadamard_scatter
```

This realizes one BMG-AOT package, then tests every selected kernel variant and
the configured split sweep in one pinned Python/Torch/XPU process. The default
`--variants all` is literal: it runs every compiled, runnable ID (`0`--`4` and
`6`--`18`) at split 8 and 32 with direct BF16 output. Use an explicit named
comma-separated shortlist when a smaller sweep is intended. Sixteen warmup
rounds precede twenty
measured rounds per arm because the first B70 sweep showed material
short-context clock settling after four warmups. Override `--variants`,
`--splits`, `--output-dtypes`, `--flush-writer`, `--prefill-store`,
`--warmup-rounds`, or `--sample-rounds` to change
the runtime matrix without rebuilding the native library. Pytest is included
for the mandatory native kill suite, and inherited Python, loader, service,
and Kvarn-selector variables are scrubbed before launch. The launcher
refuses to run beside a vLLM service, against dirty or source-mismatched
repositories, with ambiguous shared libraries, or without exact Nix
derivation and closure attestations. Evidence is written atomically under
`benchmark-results/kvarn/`; `/tmp` and overwrites are rejected.

Use `--service-layer-count 16` to add the service-shaped primitive screen. It
allocates disjoint auto caches and Kvarn packed-cache/tail pools for sixteen
logical attention layers, then times a complete round-robin store/frontend plus
decode sweep inside one outer XPU event. The first layer rotates between sweeps;
metadata and decode scratch are reused sequentially. Results retain both raw
sweep time and the mechanically normalized per-layer time, exact allocation and
pointer evidence, deterministic per-layer seeds, and cache-replication
provenance. The runner reserves ten percent of device memory (at least 1 GiB)
and fails before timing if the estimated allocation does not fit. This remains
a GPU primitive diagnostic: it does not run model projections, MLPs, the vLLM
scheduler, transport, or packed-page flushes, and can never establish service
parity.

The mandatory B70 kill suite is selector-scoped. A `native_xe2` writer run
adds compact/padded byte-exact packing, ragged block IDs, arbitrary values,
ties-to-even, constant rows, and invalid-stride rejection. A
`hadamard_scatter` prefill run adds FP16/BF16, structured rows, deterministic
repeat, allocator reuse, and B1/B4 backend-stride append cases. Incorrect
writers therefore fail before the slower service correctness matrix.

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
