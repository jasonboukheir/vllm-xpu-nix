# KVarN native XPU kernel optimization checklist

This is the living TDD and performance log for replacing the experimental
Triton KVarN decode path with an Xe2/Battlemage-native kernel. A checkbox is
only marked complete when the cited artifact or reproducible command exists.

## Ship gates

- [ ] K4V4 output throughput is at least 95% of matched native-BF16 KV.
- [ ] K4V4 mean TPOT is no more than 1.10 times matched native-BF16 KV.
- [ ] Forced-decode accuracy is no worse than the accepted Triton K4V4 result.
- [ ] Boundary, prefix-cache, MTP, and XPU-graph correctness suites pass.
- [ ] No NaNs, device faults, leaked blocks, or unstable repeated-run results.

## Frozen end-to-end baseline

Workload: Qwen3.6-27B-int4-AutoRound, eager, prefix and MTP disabled, 32
requests, concurrency 4, exactly 6000 input and 128 output tokens, four
warmups, seed 20260808.

| implementation | output tok/s | duration | mean TPOT | mean TTFT | artifact |
|---|---:|---:|---:|---:|---|
| BF16 KV | 22.42 | 182.66 s | 125.19 ms | 6910 ms | `/tmp/bf16-6k-c4-r1.json` |
| Triton K4V4 | 14.66 | 279.38 s | 220.32 ms | 6904 ms | `/tmp/kvarn-6k-c4-r1.json` |
| Triton K4V4, BF16 dot operands | 14.07 | 291.03 s | 231.49 ms | 6939 ms | `/tmp/kvarn-bf16dot-6k-c4-r1.json` |
| Native K4V4, split 16 | 20.13 | 203.50 s | 147.82 ms | pending extraction | `/tmp/kvarn-native-split16-6k-c4-r1.json` |
| Native K4V4, split 16 + fused FWHT cache update | 21.08 | 194.32 s | 141.33 ms | 6314 ms | `/tmp/kvarn-native-fwht-s16-6k-c4-r1.json` |
| Native K4V4, K2 factorized K metadata, run 1 | 20.49 | 199.93 s | 144.46 ms | 6618 ms | `/tmp/kvarn-native-k2-6k-c4-r1.json` |
| Native K4V4, K2 factorized K metadata, run 2 | 20.44 | 200.36 s | 144.71 ms | 6640 ms | `/tmp/kvarn-native-k2-6k-c4-r2.json` |
| Native K4V4, K3 factorized K+V metadata | 20.41 | 200.73 s | 145.07 ms | 6641 ms | `/tmp/kvarn-native-k3-6k-c4-r1.json` |

Minimum passing output throughput is 21.30 tok/s. Maximum passing mean TPOT is
137.71 ms. The BF16-dot experiment was reverted.

Fresh same-build steady-decode workload: four total requests, concurrency 4,
6000 input and 512 output tokens, so no replacement prefills enter during
decode. Pin `temperature=0` (greedy) for all new matched runs; vLLM's current
benchmark client otherwise inherits the server generation config
(`temperature=1`, `top_k=20`, `top_p=0.95`), while older artifacts did not
record enough sampling metadata to establish that they used the same policy.

| implementation | output tok/s | mean TPOT | median ITL | artifact |
|---|---:|---:|---:|---|
| BF16 KV | 49.92 | 59.13 ms | 47.48 ms | `/tmp/bf16-fresh-steady-6k-c4-o512-r1.json` |
| K3+D1 K4V4 | 43.40 | 67.68 ms | 56.50 ms | `/tmp/kvarn-k3d1-fresh-steady-6k-c4-o512-r1.json` |
| K3+D1 + cast bypass + fused split-16 reducer/H256 | 44.10 | 66.25 ms | 55.20 ms | `/tmp/kvarn-k3d1-fused-reduce-castless-steady-6k-c4-o512-r1.json` |

For this matched workload the gates are at least 47.42 tok/s and at most
65.04 ms TPOT. K3+D1 currently reaches 86.9% BF16 throughput and 1.145x BF16
TPOT. Fusing the reducer/H256 and removing the three decode-entry casts raises
this to 88.3% BF16 throughput and 1.120x BF16 TPOT, so the goal remains open.

## Phase 0: reproducible workflow

- [x] Record the matched BF16 and Triton K4V4 end-to-end baselines.
- [x] Localize the regression to decode: TTFT is unchanged while decode ITL is
  roughly 48-50 ms for BF16 and 131-135 ms for K4V4.
- [x] Show that independent hybrid pools are not the cause: prefix caching was
  disabled in the matched benchmark and no GPU pool synchronization is added.
- [x] Test split-K off; it is slower than the current split path.
- [x] Test explicit BF16 `tl.dot` operands; they are slower and were reverted.
- [x] Add one command that runs the kernel microbenchmark and emits JSON:
  `benchmark/benchmark_kvarn_decode.py LIBRARY --batch 4 --context 6000`.
- [x] Add one command that compares candidate JSON against the frozen gates:
  `python scripts/kvarn_perf_gate.py BF16.json CANDIDATE.json`.
- [x] Add an opt-in runtime selector so native and Triton paths can be A/B
  tested without source edits.
- [ ] Preserve every run under a timestamped `benchmark-results/` directory;
  `/tmp` is only scratch space.

## Phase 1: tests before native implementation

- [ ] Add fake/meta registration for the Python-visible native op.
- [x] Define a Python-visible native dequant microbenchmark op.
- [x] Add CPU layout tests for the exact K4V4/G128 packed record:
  K bytes/scales/zero-points/row scales and V equivalents.
- [ ] Add an XPU reference test comparing native output/LSE with the existing
  Triton K4V4 decoder at sequence lengths 1, 127, 128, 129, 255, 256, 6000,
  and a non-multiple final block.
- [ ] Cover batch sizes 1 and 4, GQA 24/4, head dimension 256, arbitrary block
  tables, shared prefix blocks, a BF16 sink, and a partial BF16 tail.
- [ ] Require finite outputs and use explicit max/mean error thresholds.
- [ ] Add deterministic repeated-run and freed-block/reuse cases.
- [x] Add a parametrized XPU runtime suite for page boundaries, permuted page
  tables, ragged B4, the independent FP32 oracle, and mixed FP16-tail/packed
  softmax (`tests/flash_attn/test_kvarn_decode_xpu.py`; 10 device cases).

## Phase 2: minimal native decoder

- [x] Add a narrow `kvarn_decode_xe2` interface and torch schema.
- [x] Dispatch only K4V4, group 128, head dimension 256, Hq/Hkv 24/4, page 128.
- [x] Reuse Xe2 paged-decode scheduling, online-softmax epilogue, and split
  reduction rather than introducing a generic SYCL attention loop.
- [x] Implement packed uint8 K and V loads with register dequantization.
- [x] Consume the existing BF16 sink/tail pools without materializing history.
- [x] Process a page as two 64-token subtiles until page-128 reduction is proven.
- [x] Use subgroup 16 DPAS/XMX QK and PV with FP32 softmax/accumulation.
- [ ] Leave Sinkhorn/full-tile store in Triton for the first decode prototype.
- [x] Wire an environment-controlled vLLM dispatch with Triton fallback.

## Phase 3: profiler-driven iteration

For every variant: run correctness first, then at least 200 warm iterations of
the isolated kernel at B=1 and B=4 for contexts 128, 1024, 6000, and 32768.
Reject variants that regress the primary B=4/6000 kernel median by more than 2%.

- [x] Establish native loader-v0 timing; profiler counters remain pending.
- [x] Compare 128 versus 256 GRFs. The 128-GRF variant passes correctness but
  runs 3.18x slower, consistent with spills/resource pressure; retain 256.
- [ ] Compare 32 versus 64-token K/V subtiles.
- [x] Review pipeline/prefetch depth. Packed KV uses synchronous irregular
  scalar loads directly into MMA fragments; double buffering would increase
  already-sensitive GRF pressure, while Q already has a block-2D prefetch.
  No safe candidate was retained.
- [ ] Reuse each dequantized KV fragment across all six query heads.
- [ ] Compact per-sequence split work so no empty split workgroups launch.
- [ ] Sweep adaptive split counts by context and batch size.
- [x] Sweep uniform split counts at B4/6K with correctness-preserving native
  reduction: S1 3014 us, S2 1522 us, S4 826 us, S8 816 us, S16 695 us.
  S16 is the isolated winner (`/tmp/kvarn-native-split16-b4-6k.json`), 76.9%
  faster than S1 including host scratch allocation and reduction.
- [x] Reject S32: 705 us median versus S16's 695 us
  (`/tmp/kvarn-native-split32-b4-6k.json`).
- [x] Reject the dynamic one-entry packed-word cache: correctness passed but
  B4/6K regressed to 1064 us (`/tmp/kvarn-native-wordcache-s16-b4-6k.json`),
  consistent with extra control flow/GRF pressure.
- [x] Hoist physical-block/tail-slot/record resolution once per tile:
  correctness passes and XPU-event median improves from 702 to 690 us
  (`/tmp/kvarn-native-metadata-s16-b4-6k.json`); retain as a small safe win.
- [x] Attribute query/output Hadamard rotation cost: at B4/H24/D256, one
  rotation is 83.93 us median and Q+output is 167.87 us per layer
  (`/tmp/kvarn-rotations-b4.json`). Across 16 full-attention layers this is
  about 2.69 ms per model step, too small to prioritize fusion.
- [x] Reject preloading all four query fragments outside the K-tile loop:
  13/13 XPU cases pass, but the extra live fragments increase the B4/6K S16
  device median from 690 to 1172 us
  (`/tmp/kvarn-native-qpreload-s16-b4-6k.json`).
- [x] Reject per-fragment subgroup packed-word broadcasts. Broadcasting both
  K and V fails the ragged oracle (3/13 failures, max absolute error 0.0282)
  because PV's value lanes are permuted. K-only broadcast passes 13/13 but
  regresses the B4/6K S16 device median from 690 to 1089 us
  (`/tmp/kvarn-native-kbroadcast-s16-b4-6k.json`). The collective/compiler
  cost outweighs fewer global loads; restore scalar loads.
- [x] Reject cooperative 32 KiB SLM staging. It passes 13/13 and reduces each
  packed K/V tile to 2048 global word loads, but S16 regresses 690 -> 729 us
  and S8 regresses 816 -> 878 us (`/tmp/kvarn-native-slm-s16-b4-6k.json`,
  `/tmp/kvarn-native-slm-s8-b4-6k.json`). SLM traffic, barriers, and occupancy
  outweigh shared dequantization; restore scalar loads.
- [x] Reject branch-free paired uint64 fragment loading/static metadata reuse:
  21/21 combined decoder/FWHT cases pass, but B4/6K S16 regresses 690 -> 1407
  us (`/tmp/kvarn-native-u64static-s16-b4-6k.json`), again indicating severe
  register/compiler pressure. Restore scalar fragment loads.
- [x] Add and retain a native SG16 H256 FWHT fused directly with sparse tail
  scatter. All 8 dtype/oracle/sentinel/determinism tests pass. At B4 BF16 it
  is 49.77 us median versus 137.45 us for dense rotation alone, before the old
  scatter (`/tmp/kvarn-fwht-scatter-b4.json`). The frozen serving run improves
  from 20.13 to 21.08 tok/s and 147.82 to 141.33 ms TPOT.
- [x] Add a KVarN-local split-16 subgroup reducer without touching the generic
  paged-decode header. All 30 combined cases pass; B4/6K improves only 690 ->
  684 us (`/tmp/kvarn-native-fastreduce-s16-b4-6k.json`). Retain as a small
  non-regressing win. A generic-header version was abandoned because it
  invalidated every unrelated paged-decode template and made iteration
  impractical.
- [x] Add native H256 query FWHT. All N=24/48/72/96 FP16/BF16 oracle cases
  pass; device median is about 22 us versus 39-43 us for dense FP16 matmul,
  saving roughly 0.29 ms across 16 full-attention layers. Retain.
- [x] Reject 128-GRF decode: 30/30 cases pass but B4/6K S16 regresses 684 ->
  2175 us (`/tmp/kvarn-native-grf128-fastreduce-s16-b4-6k.json`), consistent
  with severe spilling. Restore the 256-GRF-only launch.
- [x] Hoist packed-K per-token row scaling out of every FP16 MMA fragment and
  apply it once to the FP32 score column. The adversarial 127/128/129 boundary
  test with nonuniform row scales and all 31 combined device cases pass;
  B4/6K improves 684 -> 666.88 us
  (`/tmp/kvarn-native-k1-fastreduce-s16-b4-6k.json`). Retain the 2.5% win.
- [x] Check K1 at B1/6K and B4/32768: 247.55 us and 3773.49 us device medians,
  respectively (`/tmp/kvarn-native-k1-fastreduce-s16-b1-6k.json`,
  `/tmp/kvarn-native-k1-fastreduce-s16-b4-32768.json`). B1 also improves over
  the prior 259 us result; the long-context run is finite and stable.
- [x] Factor packed-K dimension scales onto Q and accumulate the dimension
  zero-point term once per query row (K2). The strengthened metadata oracle
  and all 36 device/planning cases pass; B4/6K improves 666.88 -> 481.59 us
  (`/tmp/kvarn-native-k2-fastreduce-s16-b4-6k.json`). Retain provisionally;
  this is 27.8% faster than K1 and 29.6% faster than the pre-factorization
  specialized-reducer baseline.
- [x] Attribute K2 in serving: bounded steady-state B4 events show native
  decode at 645.46 us/layer versus K1's prior 816 us, with fused cache update
  74.73 us, query FWHT 61.28 us, and output FWHT 5.18 us. Despite that device
  improvement, two frozen end-to-end runs are only 20.49/20.44 tok/s and
  144.46/144.71 ms TPOT. Keep K2 while investigating benchmark wave/prefill
  effects; it has not passed the serving gates.
- [x] Factor packed-V row metadata around PV with one temporary 32-dimension
  accumulator (K3). All 36 cases pass; B4/6K improves 481.59 -> 397.29 us and
  B1/6K improves 175.65 -> 141.56 us
  (`/tmp/kvarn-native-k3-fastreduce-s16-b4-6k.json`,
  `/tmp/kvarn-native-k3-fastreduce-s16-b1-6k.json`). Retain provisionally;
  K3 is 41.9% faster than the 684 us pre-factorization baseline.
- [x] Reject a selectable 32-token KV tile. All 36 cases pass, but B4/6K
  regresses 397.29 -> 600.52 us (+51.2%) because doubled tile iterations,
  online-softmax work, and barriers outweigh occupancy benefits
  (`/tmp/kvarn-native-k3-tile32-s16-b4-6k.json`). Restore tile 64 only.
- [x] Deduplicate invariant-axis factor metadata loads (D1), reducing logical
  FP16 metadata loads per packed workgroup/tile 25,600 -> 3,200 without
  changing arithmetic. All 36 cases pass; B4/6K improves only 397.29 ->
  394.45 us (`/tmp/kvarn-native-k3-d1-s16-b4-6k.json`). Retain as a safe
  0.7% win and simpler memory-access pattern.
- [x] Rebuild after project formatting and rerun the retained final artifact:
  36/36 cases pass and B4/6K is 394.43 us
  (`/tmp/kvarn-native-k3-d1-final-s16-b4-6k.json`), reproducing D1.
- [x] Reject reducing factor-bias state/reductions from eight packed rows to
  six valid GQA rows (D2). All 36 cases, NaN canaries, and all 24 heads pass,
  but added row guards regress B4/6K 394.45 -> 435.00 us (+10.3%)
  (`/tmp/kvarn-native-k3-d2-s16-b4-6k.json`). Restore branch-free eight-row
  handling.
- [x] Reject caching six K zero-point biases across the two page halves in 96
  bytes of KVarN-local shared storage (D3). The new odd-tile split-start oracle
  and all 37 cases pass, but B4/6K regresses 394.45 -> 398.96 us (+1.1%)
  (`/tmp/kvarn-native-k3-d3-s16-b4-6k.json`). Restore local recomputation to
  avoid shared-memory/barrier complexity for no gain.
- [x] Re-sweep split count after K3+D1 changed the mainloop/reducer balance.
  S8 is 485.36 us versus S16 at 394.45 us
  (`/tmp/kvarn-native-k3-d1-s8-b4-6k.json`); S32 is intentionally unsupported.
  Retain S16.
- [x] Reject packed `sycl::vec<float,8>` XOR-tree factor-bias reduction (D4).
  All 36 cases pass, but B4/6K is 395.10 us versus D1's 394.45 us and shows a
  2455 us maximum outlier (`/tmp/kvarn-native-k3-d4-s16-b4-6k.json`). Restore
  scalar collectives; vector shuffle does not improve Xe lowering here.
- [x] Reject folding V dimension scale into packed q/direct output accumulation
  to remove the K3 temporary (D5). It fails 5/36 cases (ragged B2-B4,
  adversarial factors, and hybrid-tail softmax) with max absolute error about
  0.25. Do not benchmark; restore K3's page-local temporary accumulation.
- [x] Fix the Nix KVarN runner to forward additional CLI arguments. This made
  `--profiler-config` visible to vLLM instead of silently dropping it.
- [x] Attempt a warmed Torch/Kineto trace. The XPU profiler fails at start with
  `PTI_ERROR_NOT_IMPLEMENTED` and a callback-thread mismatch in this toolchain,
  so use bounded `torch.xpu.Event` attribution instead.
- [x] Add opt-in bounded serving attribution (`KVARN_XPU_PROFILE_EVENTS=1`),
  including an arming point and skip window so 6K prefill and B1-B3 ramp-up do
  not contaminate steady B4 decode. With 128 full-attention layer calls skipped
  and 256 captured, per-layer means are: native decode 816 us, query rotation
  100 us, output unrotation 5 us, K/V cache rotation 129 us, tail scatter
  125 us. Across 16 full-attention layers these measured KVarN regions total
  about 18.8 ms/model step; native attention itself is about 13.1 ms.
- [ ] Attribute tail/sink handling and split-stage reduction costs.
- [ ] Record kernel launches, device time, EU/XMX utilization, memory bandwidth,
  GRF usage/spills, and occupancy for each retained variant.
- [x] Prototype a same-size DPAS-native packed layout derived from the compiled
  CUTE fragment `tv_layout`. The diagnostic op/XPU bijections pass (K
  `[16,64,2]`, V `[16,32,2]`), and independent property tests cover all nibble
  values, boundaries, round trips, and canonical/swizzled dequant equality.
- [x] A/B a lane-contiguous loader (32-byte K and 16-byte V loads per lane)
  against the retained scalar-u32 loader at contexts 6000, 6145, and 6512.
  Compile-time-isolated DPAS is 38.7-39.6% faster and bit-exact; migrate the
  producer/readers behind an opt-in format selector next.
- [ ] Benchmark a fused RTN + K4V4 pack + record-scatter flush kernel separately;
  treat it as a boundary-latency improvement unless steady-state attribution
  shows that flush work materially contributes to the remaining TPOT gap.

## Phase 4: integration gates

- [x] Run persistent forced-decode BF16-versus-K4V4 accuracy evaluation.
  Safe DPAS cache + Triton reader passes (98.44% top-1, 100% top-5,
  MAE 0.01060, RMSE 0.01604, p95 0.03125) and matches the canonical Triton
  control (98.44%/100%, MAE 0.0115). The experimental native decoder fails
  (17.19% top-1, 39.06% top-5, MAE 1.508) before the GDN scratch-lifetime
  fix. After recording the asynchronous A/W/U scratch tensors on the current
  XPU stream, native compact split-1 passes the full64 comparison: 98.44%
  top-1, 100% top-5, tie-aware agreement 100%, MAE 0.01115, RMSE 0.01643,
  and p95 0.03125
  (`/tmp/kvarn-accuracy-compact-native-recordstream-full64.npz`).
- [x] Re-run exact top-1/top-5/logit metrics and compare with the Triton pilot.
- [ ] Test prefix sharing and block-table aliasing across two requests.
- [x] Test two-token MTP (`qlen=3`: current token `n` plus drafts `n+1` and
  `n+2`) accepted lengths 1, 2, and 3 plus consecutive rejection. The matched
  B4 graph runs exercise all three lengths: compact K4V4 records 26/24/146
  steps and accepts 316/392 drafts (80.61%); BF16-KV records 26/27/144 and
  accepts 315/394 (79.95%). Replayed B4 token IDs are identical. The direct
  persistent-request trace records step lengths beginning `1,1,2,3`, proving
  consecutive first-draft rejection followed by successful recovery; it later
  repeats `1,1`. Artifacts: `/tmp/kvarn-compact-mtp-b4-lifecycle-trace.json`
  and `/tmp/bf16-mtp-b4-lifecycle-current.json`. The retained native
  split-32 reader reproduces the same 30/27/164 length counts, accepts
  355/442 drafts (80.32%), and passes identical B4 replay plus consecutive
  rejection/recovery (`/tmp/kvarn-compact-native-mtp-split32-b4-lifecycle.json`).
- [x] Test eager and XPU graph capture sizes 3 and 6. Compact native split-1
  compiled and captured the requested `[1,3,6]` sizes, selected the Xe2 native
  path at replay, and completed a forced two-token decode with finite logits.
  Its top choices match eager (`68`, `82`); aligned top-51 graph/eager maximum
  absolute drift is 0.09375 and 0.03125. Artifact:
  `/tmp/kvarn-compact-native-recordstream-graph-two.npz`. Keep the eager-only
  finite-value diagnostic disabled during graph compilation because its
  `.item()` checks introduce data-dependent Dynamo guards.
- [ ] Test cancellation, preemption, block reuse, and repeated startup/shutdown.
- [ ] Run the frozen end-to-end benchmark at least three times per mode.
- [ ] Pass both ship gates from the median matched run.
- [ ] Run a long-context capacity/performance check without changing cache size.
- [ ] Package local `vllm` and `vllm-xpu-kernels` source overrides in the Nix
  runner and document the exact commands and derivations.

### Native two-token MTP performance

Here, two-token MTP means a qlen=3 verification row containing the current
token `n` and both speculative successors `n+1` and `n+2`. The native reader
treats B4 verification as twelve virtual one-token rows. The initial split-1
implementation passes correctness but reaches only
61.59 output tok/s and 36.02 ms TPOT. A B12/6K isolated sweep records split
medians of 2502/1454/1476/1754/1612/1599 us for splits 1/2/4/8/16/32. The
isolated split-2 winner does not translate to serving because its reduction
changes draft acceptance; it falls to 58.20 tok/s and 40.24 ms TPOT. Split-32
is the retained serving winner at 66.65 tok/s, 32.64 ms TPOT, and 92.89%
acceptance (`/tmp/kvarn-compact-native-mtp-split32-graph-6k-c4-o512-r1.json`).
The same-source BF16 MTP control reaches 70.54 tok/s and 31.51 ms TPOT
(`/tmp/bf16-mtp-graph-6k-c4-o512-r1.json`). Those initial three-seed results
used the server's stochastic sampling defaults, so their unmatched acceptance
sequences are retained only as exploratory data and are not a valid ship gate.
The corrected deterministic (`temperature=0`) compact runs reach
65.39/70.65/65.40 tok/s, 33.06/31.49/33.66 ms TPOT, and
13.07/12.73/12.78 s TTFT. Matched BF16 runs reach 72.86/77.76/71.26 tok/s,
31.22/29.97/32.99 ms TPOT, and 10.97/10.74/10.74 s TTFT. Their per-seed
throughput ratios are 89.75%, 90.85%, and 91.77%; the ratio of medians is
89.76%. TPOT ratios are 1.059x, 1.051x, and 1.020x (1.051x median). Compact
therefore passes the 1.10x TPOT gate but does not yet pass the 95% throughput
gate. The roughly two-second compact TTFT delta localizes most of the remaining
fixed-output throughput deficit to prefill/cache production rather than MTP
decode. Deterministic artifacts are
`/tmp/kvarn-compact-mtp-deterministic-seed{0,1,2}.json` and
`/tmp/bf16-mtp-deterministic-seed{0,1,2}.json`. Two apparent
136-139 tok/s repeats are excluded: identical random prompts hit the enabled
prefix cache and reduced mean TTFT to about one second. Capturing graph size
12 is also rejected: it halves capacity from 98,304 to 49,152 tokens and falls
to 15.67 tok/s / 102.68 ms TPOT. Restricting native FWHT/scatter to decode-size
inputs and sending large prefill chunks through the tensor-core rotation path
is also rejected: the seed-1 control regresses from 68.96 tok/s and 13.06 s
TTFT to 65.03 tok/s and 14.79 s TTFT. Retain native scatter for all input
sizes; next attribute the valid compact producer and flush path during prefill.
An attempted metadata-publication optimization shared the identical block-ID
and pool-slot device tensors across all flushing attention layers instead of
performing the existing per-layer H2D uploads. A follow-up variant added one
explicit eager producer-to-flush synchronization before sharing those tensors.
On the warmed seed-2 discriminator it reaches 61.00 tok/s, 37.55 ms TPOT, and
12.74 s TTFT versus the retained seed-2 result's 61.61 tok/s, 35.10 ms, and
12.76 s. The explicit wait therefore preserves TTFT but does not improve it;
both experimental variants are removed. Acceptance varies with the server's
sampling sequence, so their unmatched 57.23%/66.59% acceptance values are not
used as corruption evidence. Artifacts:
`/tmp/kvarn-compact-mtp-{shared-flush-indices,explicit-flush-sync}-seed2.json`.
Raising `--max-num-batched-tokens` from the MTP default of 2048 to 8192 is
also rejected. The larger compile range raises peak activation usage to
1.68 GiB, reduces cache capacity from 98,304 tokens / 12x to 40,960 / 5x,
and the first B4 run completes only two requests after roughly 102 seconds
before being stopped. This trades away both capacity and serving performance;
retain 2048 and optimize the compact flush itself.
A fused Triton RTN/K4V4 pack prototype was byte-exact for all K and V packed,
scale, zero-point, and auxiliary fields, but improved startup TTFT by only
about 31 ms and did not materially change serving performance. The prototype
and its validation hooks are removed; the next profile must include the
Sinkhorn transforms and record scatter rather than isolating packing alone.
Full device-event attribution at B4/6K shows a representative 255-pair flush
spends about 4 ms gathering, 76 ms in Sinkhorn plus packing, and 0.6 ms in the
record scatter. Cross-layer batching reduces a 17-pair retirement burst from
about 28 ms to 5.5 ms, but large 255-pair bursts regress to 87-89 ms with one
to four launches and leave TTFT unchanged. The cross-layer batching and timing
hooks are therefore removed. Across the repeated 6K prefill boundaries the
retained Sinkhorn path accounts for roughly 0.8-0.9 seconds, material but still
less than the approximately two-second compact-versus-BF16 TTFT gap.
Reducing Sinkhorn from eight to four iterations is numerically plausible but
still misses the performance gate. Deterministic compact seeds reach
68.29/72.13/66.53 tok/s with 12.82/12.48/12.51 s TTFT and
32.11/30.74/34.13 ms TPOT. Against the matched BF16 seeds the ratio of medians
is 68.29/72.86 = 93.73%, below the required 95%; median TPOT remains safely
within 1.10x. Artifacts:
`/tmp/kvarn-compact-mtp-sinkhorn4-deterministic-seed{0,1,2}.json`. Do not
promote four iterations without the forced-decode accuracy gate; evaluate two
iterations only as a discriminator for whether further Sinkhorn work can close
the remaining throughput gap.
The corrected forced-decode run explicitly loads the locally fixed GDN library;
without it, both the retained eight-iteration control and four-iteration
candidate reproduce all-NaN logits from step zero due asynchronous GDN scratch
reuse, so that failure is not attributable to Sinkhorn. With fixed GDN, the
four-iteration full64/full-vocabulary artifact is fully finite with identical
forced tokens, 100% exact top-1/top-5/tie-aware agreement, MAE 0.006637, RMSE
0.011571, p95 0.03125, and max 0.125 against the paired BF16 baseline. Artifact:
`/tmp/kvarn-accuracy-compact-native-sinkhorn4-fixed-gdn-full64.npz`. Two
iterations is rejected as a performance lever: seed 0 reaches 68.39 tok/s,
12.73 s TTFT, and 31.94 ms TPOT, statistically identical to four iterations.
Zero iterations establishes the fixed-cost floor but also fails the ship gate:
deterministic seeds reach 68.97/73.19/66.75 tok/s, 12.59/12.25/12.28 s TTFT,
and 31.75/30.34/33.37 ms TPOT. The 68.97 tok/s median is 94.66% of the
matched BF16 median, still below 95%, and saves too little over the
accuracy-proven four-iteration candidate to justify removing normalization.
Artifacts: `/tmp/kvarn-compact-mtp-sinkhorn0-deterministic-seed{0,1,2}.json`.
Reducing the four-iteration kernel from eight to four warps is also rejected:
seed 0 regresses to 64.00 tok/s with 12.91 s TTFT, while kernel resource use
reduces capacity from 98,304 tokens / 12x to 90,112 / 11x. The temporary warp
override was removed. Artifact:
`/tmp/kvarn-compact-mtp-sinkhorn4-warps4-seed0.json`.
Reducing Triton's pipeline depth from two stages to one is rejected before a
serving benchmark: startup capacity collapses to 49,152 tokens / 6x. The
temporary stage override was removed.
Combining K and V into one native Hadamard/scatter workgroup is also rejected.
Although it halves the workgroup count and preserves finite qlen=3 execution
with 97.91% acceptance on unseen seed 1, doubled live transform state raises
register pressure and regresses the exact 4x6000/512 run to 61.21 tok/s and
38.19 ms TPOT. The implementation was restored to independent K/V workgroups.
The preceding same-seed warm repeat reached 155.35 tok/s and 1.33 s TTFT
because all four 6K prompts hit prefix cache; retain it as positive cache-hit
behavior, not as comparable performance evidence. Artifacts:
`/tmp/kvarn-compact-mtp-sinkhorn4-paired-kv-producer-seed{0,1}.json` and
`/tmp/kvarn-compact-mtp-sinkhorn4-paired-kv-producer-seed0-warm.json`.
A 32-lane native producer subgroup is capacity-neutral and retains finite
two-token MTP execution, but seed 0 reaches only 67.89 tok/s, 12.84 s TTFT,
and 31.85 ms TPOT versus 68.29 tok/s for the 16-lane control. It is removed.
Artifact: `/tmp/kvarn-compact-mtp-sinkhorn4-scatter-sg32-seed0.json`.
A 3072-token scheduler budget would reduce a 6K prefill from three chunks to
two, but startup activation profiling leaves only 40,960 cache tokens / 5x,
the same capacity cliff observed at 8192. It is rejected without a serving
run, and the profile remains at the MTP-derived 2048 default.
Transposing rectangular V tiles to concatenate them with K in one Sinkhorn
launch is invalid: it reverses V's alternating normalization order, reducing
seed-0 MTP acceptance to 59.05% and throughput to 54.82 tok/s. The
discriminator was removed. Artifact:
`/tmp/kvarn-compact-mtp-sinkhorn4-fused-rect-seed0.json`.
Reusing the column deviation already computed by each iteration's imbalance
check is algebraically redundant-work elimination, but its changed Triton live
range gives no gain (67.99 versus 68.29 tok/s) and shifts acceptance from
92.62% to 92.16%. It too was restored to the bit-stable control. Artifact:
`/tmp/kvarn-compact-mtp-sinkhorn4-reuse-colstd-seed0.json`.
Selecting the fourth normalization pass directly and removing best-so-far
scoring raises seed 0 only to 68.56 tok/s with 12.68 s TTFT and 31.81 ms TPOT.
That cannot reach the 69.22 tok/s three-seed ship threshold and changes
acceptance to 92.36%, so the compile-time discriminator was removed without a
full64 promotion run. Artifact:
`/tmp/kvarn-compact-mtp-sinkhorn4-final-only-seed0.json`.

## Iteration log

| variant | correctness | B4/6K result | decision |
|---|---|---|---|
| Triton baseline | pass | 14.66 tok/s, 220.32 ms TPOT | native kernel needed |
| Triton BF16 dot operands | 64-token smoke pass | 14.07 tok/s, 231.49 ms TPOT | reverted |
| Native scalar materializing dequant v0 | pass | B4/6K records: 844 us, 148 GB/s, 28.5% copy ceiling | reject layout |
| Native coalesced two-launch dequant v1 | pass | 972 us, 129 GB/s, 24.7% copy ceiling | reject launch split |
| Native byte-paired dequant v2 | pass | 557 us, 224 GB/s, 43.1% copy ceiling | retain layout |
| Native uint32 dequant v3 | pass | 509 us, 245 GB/s, 47.2% copy ceiling | current winner; below 60% gate |
| Native uint64 dequant v4 | pass | 662 us, 189 GB/s, 36.3% copy ceiling | reverted to uint32 |
| Native fused DPAS decode v0 | compile gate pass | attention-only Nix derivation builds | wire op and run oracle next |
| Native fused DPAS decode v1 | structured + randomized oracle pass | B4/6K: 4395 us median | add BF16 hybrid pool, then integrate |
| Native fused DPAS decode v2 | build pass; runtime validation interrupted by Xe device entering D3hot after command-stream abort | pending | keep unshipped; validate ragged early-exit rebuild before serving |
| Native fused DPAS decode v2, ragged early exit | 11/11 XPU cases pass | B4/6K: 3010 us median (`/tmp/kvarn-native-ragged-b4-6k.json`) | retain; 31.5% faster than v1 |
| Native fused DPAS decode v3, B1-B4 serving | 13/13 XPU cases pass; native selected for B=1,2,3,4 | 16.22 tok/s, 193.53 ms TPOT (`/tmp/kvarn-native-b1-4-6k-c4-r1.json`) | retain integration; optimize kernel (76.1% throughput and 1.546x TPOT vs BF16) |
| Native split-16 + metadata hoist | 13/13 XPU cases pass | 20.13 tok/s, 147.82 ms TPOT (`/tmp/kvarn-native-split16-6k-c4-r1.json`); isolated 690 us | retain; 89.8% BF16 throughput and 1.181x BF16 TPOT, still short of gates |
| Native split-16 + query preload | 13/13 XPU cases pass | isolated 1172 us device median (`/tmp/kvarn-native-qpreload-s16-b4-6k.json`) | revert; register pressure/occupancy regression |
| Native split-16 + K/V subgroup broadcast | 10/13 XPU cases pass | not benchmarked | revert V; PV lane permutation invalidates assumed grouping |
| Native split-16 + K-only subgroup broadcast | 13/13 XPU cases pass | isolated 1089 us device median (`/tmp/kvarn-native-kbroadcast-s16-b4-6k.json`) | revert; subgroup collectives/compiler effects regress 58% |
| Native split-16 + cooperative 32 KiB SLM staging | 13/13 XPU cases pass | isolated 729 us S16; 878 us S8 (`/tmp/kvarn-native-slm-s16-b4-6k.json`) | revert; 5.7% slower than scalar winner at S16 |
| Native split-16 + paired uint64/static metadata | 21/21 combined cases pass | isolated 1407 us (`/tmp/kvarn-native-u64static-s16-b4-6k.json`) | revert; register/compiler regression |
| Native split-16 + fused FWHT cache update | 21/21 combined cases pass; B1-B4 smoke passes | 21.08 tok/s, 141.33 ms TPOT (`/tmp/kvarn-native-fwht-s16-6k-c4-r1.json`) | retain; 94.0% BF16 throughput, 1.129x BF16 TPOT; close but not through gates |
| Native split-16 + specialized subgroup reducer | 30/30 combined cases pass | isolated 684 us (`/tmp/kvarn-native-fastreduce-s16-b4-6k.json`) | retain as small KVarN-local win |
| Native query FWHT | independent FP16/BF16 oracle pass at B1-B4 row counts | 22 us vs 39-43 us dense rotation | retain as safe ~0.29 ms/token win |
| Native decode at 128 GRFs | 30/30 combined cases pass | isolated 2175 us (`/tmp/kvarn-native-grf128-fastreduce-s16-b4-6k.json`) | revert; 3.18x register-spill regression |
| Native K row-scale hoist (K1) | 31/31 combined cases pass, including adversarial page boundary | isolated 666.88 us (`/tmp/kvarn-native-k1-fastreduce-s16-b4-6k.json`) | retain; 2.5% faster than specialized-reducer baseline |
| Native factorized K metadata (K2) | 36/36 combined device/planning cases pass | isolated 481.59 us (`/tmp/kvarn-native-k2-fastreduce-s16-b4-6k.json`) | retain provisionally; 27.8% faster than K1 |
| Native factorized K+V metadata (K3) | 36/36 combined device/planning cases pass | isolated 397.29 us (`/tmp/kvarn-native-k3-fastreduce-s16-b4-6k.json`) | retain provisionally; 17.5% faster than K2 |
| Native K3 tile 32 | 36/36 combined device/planning cases pass | isolated 600.52 us (`/tmp/kvarn-native-k3-tile32-s16-b4-6k.json`) | revert; 51.2% slower than tile 64 |
| Native K3 invariant metadata dedup (D1) | 36/36 combined device/planning cases pass | isolated 394.45 us (`/tmp/kvarn-native-k3-d1-s16-b4-6k.json`) | retain; safe 0.7% win |
| Native K3+D1 final formatted build | 36/36 combined device/planning cases pass | isolated 394.43 us (`/tmp/kvarn-native-k3-d1-final-s16-b4-6k.json`) | retained handoff candidate |
| Native K3 six-row factor bias (D2) | 36/36 combined device/planning cases pass | isolated 435.00 us (`/tmp/kvarn-native-k3-d2-s16-b4-6k.json`) | revert; row guards regress 10.3% |
| Native K3 half-page K-bias reuse (D3) | 37/37 combined cases pass, including odd split start | isolated 398.96 us (`/tmp/kvarn-native-k3-d3-s16-b4-6k.json`) | revert; 1.1% slower and more synchronization complexity |
| Native K3 packed XOR bias reduction (D4) | 36/36 combined cases pass | isolated 395.10 us (`/tmp/kvarn-native-k3-d4-s16-b4-6k.json`) | revert; no gain and worse tail outlier |
| Native K3 direct-PV/no-temp (D5) | 31/36 cases pass; ragged/factor/hybrid cases fail | not benchmarked | revert at correctness gate; max absolute error ~0.25 |
| Native K3+D1 fused split-16 reducer + H256, decode-entry cast bypass | 44/44 combined device/planning cases pass | isolated 395.91 us (`/tmp/kvarn-native-k3d1-fused-reduce-fwht-s16-b4-6k.json`); steady 44.10 tok/s, 66.25 ms TPOT | retain provisionally; removes output GEMM and three entry casts, but still misses both gates |
| Native K3+D1 fused split-32 reducer + H256 | 47/47 combined cases pass | 415.52 us at 6000, 444.27 us at 6145, 449.14 us at 6512 | reject; 4.9% slower than S16 at 6000 and only 1.3% faster at 6512 |
| Native K3+D1 fused split-24 reducer + H256 | 50/50 combined cases pass | 441.98 us at 6000, 440.05 us at 6145, 442.73 us at 6512 | reject; 11.6% slower than S16 at 6000 and only 2.7% faster at 6512 |
| Native fused direct caller-output copy | serving smoke pass | 43.85 tok/s, 66.77 ms TPOT (`/tmp/kvarn-k3d1-fused-direct-output-steady-6k-c4-o512-r1.json`) | reject; no improvement over 44.10 tok/s, 66.25 ms control |
| Native persistent caller-owned split scratch | 51/51 combined cases pass; old/new exact across repeated ragged B4 calls | 43.87 tok/s, 66.75 ms TPOT (`/tmp/kvarn-k3d1-fused-persistent-scratch-steady-6k-c4-o512-r1.json`) | reject as default; no serving improvement, retained behind `KVARN_NATIVE_XPU_PERSISTENT_SCRATCH=1` |
| Native K3+D1 fused split-17 reducer + H256 | retained S16 suite 55/55 passes; split-17 FP32 reducer oracles pass at 6145/6400/6512/6528 | 397.50 us at 6000, 399.79 us at 6145, 475.05 us at 6400, 512.32 us at 6512 | reject; non-power-of-two scheduling/compiler cost overwhelms the predicted six-tile critical path |
| Native K3+D1 without per-tile UGM fence/workgroup barrier | 55/55 combined cases pass | 394.97 us at 6000, 454.95 us at 6512 (`/tmp/kvarn-no-tile-barrier-s16-b4-c*.json`) | revert; statistically identical to retained control, so compiler already removes/hides it |
| DPAS-native lane-contiguous K4V4 layout | bijection/property/ragged-hybrid exact tests pass | canonical 396.17/451.95/456.90 us vs DPAS 242.81/273.70/275.78 us at 6000/6145/6512 | retain prototype; 38.7-39.6% isolated win with unchanged 16,384-byte K/V payloads; integrate producer/readers behind opt-in format |
| DPAS-native direct-flush serving integration | full 65,536-byte writer oracle exact; ragged/hybrid bit-exact; 4/4 B4 serving requests pass | warm DPAS 46.67-46.91 tok/s, 63.58-63.94 ms TPOT vs matched warm canonical 43.04 tok/s, 68.48 ms | retain opt-in; TPOT gate passes, throughput improves 8.4-9.0% but remains 1.1-1.6% below 47.42 tok/s gate; profile next |
| DPAS-native graph serving | B4 smoke and full 6K/512 pass; native dispatch selected; no capture/pointer fault | warm 59.71 tok/s, 47.45 ms TPOT, 9.89 s TTFT | clears both ship gates; repeat for stability and warm the flush/materialize Triton shapes at startup because the first cold run paid JIT in TTFT |
| Safe DPAS cache + Triton graph reader | full64 BF16 comparison passes; native decoder/scatter defaults off | two warm runs 59.57-59.71 tok/s, 47.45-47.49 ms TPOT | production-safe candidate clears throughput and TPOT gates; graph+prefix remains unsupported and eager prefix coverage is required |

## Current recovery checkpoint

- [x] Add per-row `seq_lens` without a host `.item()` synchronization.
- [x] Stop a row before loading wholly padded ragged tiles; the exit is
  workgroup-uniform and padded block-table entries need not be valid pages.
- [x] Rebuild v2 after the ragged early-exit change:
  `/nix/store/gmyx7achaj3sx6k07wgx7s6kj2c6xvlj-python3.12-vllm-xpu-kernels-0.1.12+unstable.0000.00.00.gdirty`.
- [x] Restore local Xe device discovery and rerun the structured, hybrid, and
  randomized ragged oracle before any serving benchmark.
- [x] Confirm the native serving selector dispatches for every serving batch
  B=1 through B=4 without Triton decode-stage JIT.
- [x] Run the first frozen B4/6K serving benchmark and evaluate both performance
  gates.

## Compact-record iteration (35,072-byte D256 K4V4 records)

- [x] Audit padded-size assumptions. The sole physical inflation is the
  power-of-two slot policy; producers and Triton readers already consume tensor
  strides and semantic offsets.
- [x] Define an immutable cache ABI using the explicit
  `kvarn_k4v4_g128_compact` preset; retain `kvarn_k4v4_g128` as the padded
  65,536-byte fallback.
- [x] Prove allocator accounting: 35,072 / 128 = 274 bytes/head/token, so a
  D256/Hkv4 compact attention page is exactly 140,288 bytes versus 262,144
  padded and 524,288 BF16. Independent KVarN/Mamba pools need no cross-page
  divisibility.
- [x] Pass padded/compact config, cache-shape, adjacent-record canary,
  canonical/DPAS pack-dequant, and exact page-accounting CPU tests.
- [x] Pass XPU writer-to-reader parity at page boundaries 127/128/129 and
  255/256/257, including multiple physical blocks and heads.
- [x] Pass allocator lifecycle, cancellation/deferred-free, reuse, and hybrid
  independent-pool capacity tests (33 focused cases).
- [x] Pass shared-prefix eager serving with compact records. The explicit
  prefix-enabled profile passed 127/128/129/4096-token shared-prefix cases
  with identical decoded token IDs/text/finish reasons, maximum top-5
  logprob drift 0.11685, and a measured server prefix-cache hit rate rising
  to 64.7%. Artifact: `/tmp/kvarn-compact-prefix-gate.json`.
- [x] Confirm the native decoder accepts the natural 35,072-byte head stride
  and allocates 140,288-byte compact attention pages. Keep the safe Triton
  reader as the production default while native full-model accuracy is unresolved.
- [x] Add production DPAS-writer to native-reader tests with real Sinkhorn
  factors, both 35,072/65,536-byte strides, and sequence boundaries
  127/128/129/255/256/257. The correctness-first elementwise reconstruction
  and split-1 direct-output fix pass all 16 focused cases.
- [x] Fix native full-model finite-value correctness. The all-NaN artifact was
  caused upstream of native attention: the asynchronous GDN prefill released
  its function-local A/W/U scratch tensors while `chunk_fwd_o_kernel` still
  consumed them. Recording those allocations on the current XPU stream fixes
  the lifetime without a queue-wide wait. The focused FP16/BF16 allocator-
  pressure regression passes, as do one-step and full64 compact native runs.
- [x] Fix split-16 scratch lifetime and short-context scheduling. The legacy
  wrapper released its local partial-output/statistics tensors while the
  asynchronous reducer still consumed them; recording all three allocations
  on the current XPU stream fixes allocator-driven corruption. Compact and
  padded allocator-pressure cases pass in three fresh processes. Configured
  split-16 now falls back to direct split-1 until the maximum context has at
  least one 64-token tile per split. The adaptive full64 run is finite and
  reaches 100% top-1/tie-aware agreement with BF16 (MAE 0.01170, p95 0.03125,
  max 0.125). Artifact:
  `/tmp/kvarn-accuracy-compact-native-recordstream-adaptive16-full64.npz`.
- [x] Restore K3/D1 factorized packed metadata after the DPAS integration
  accidentally moved scale/zero-point application back into every fragment
  element. The combined canonical/DPAS mainloop retains adaptive splitting and
  allocator `recordStream` protection, passes 48 focused decoder/layout,
  ragged/hybrid, split, and pressure cases, and restores B4/6K split-16 device
  medians from about 1414 us to 390.96 us canonical and 238.75 us DPAS.
  Caller-owned and allocation-owning DPAS scratch paths are 238.75 and
  242.27 us respectively, so the lifetime fix is performance-neutral. Artifacts:
  `/tmp/kvarn-k3d1-restored-{canonical,dpas}-s16-b4-6k.json` and
  `/tmp/kvarn-k3d1-recordstream-dpas-alloc-s16-b4-6k.json`.
- [x] Attribute the restored decoder inside model execution and fix local
  symbol binding. Merely mapping the narrow library was insufficient: the
  packaged full attention library's earlier registration still selected its
  stale implementation (1506.81 us native-decode event mean). Giving the
  narrow library the production `libattn_kernels_xe_2.so` SONAME and
  preloading it before the packaged full library makes narrow symbols win
  while the packaged library supplies unrelated attention symbols. The
  corrected mixed-B1-B4 model window measures 312.26 us, and an isolated call
  through the packaged extension measures 236.43 us. The first graph artifact
  below is retained as a wrong-binding control, not a restored-kernel result:
  `/tmp/kvarn-k3d1-restored-graph-matched-6k-c4-o512-r1.json` and
  `/tmp/kvarn-dual-preload-packaged-extension-dpas-b4-6k.json`.
- [x] Measure realized attention page size, total token capacity, and the
  effective compression ratio versus padded K4V4 and BF16.
- [x] Isolate compact-stride reader cost with identical seeded DPAS records at
  B4/context 6000 and 6512. Materialize and fused split-K medians stay within
  1.3% of padded records, ruling out the 35,072-byte stride as the source of
  the earlier serving regression.
- [x] Repeat the persistent-cache full64/full-vocabulary BF16 comparison after
  fixing the direct-writer head stride. Token-ID-aligned MAE is 0.01136, p95
  absolute drift is 0.03125, and every compact argmax is in the BF16 argmax
  tie set. The three apparent top-1 disagreements are BF16 exact ties, not a
  lower-scoring compact choice.
- [ ] Pass matched eager/graph B4/6K/O512 performance gates. Promote compact
  only after all correctness and performance gates pass. A fresh same-source
  graph BF16 control reaches 59.17 tok/s and 48.62 ms TPOT
  (`/tmp/bf16-graph-matched-6k-c4-o512-r1.json`). With corrected narrow-first
  symbol binding, two compact graph runs reach 55.15/55.02 tok/s and
  50.70/50.83 ms TPOT with 4/4 exact 6000/512 completions. TPOT passes at
  1.043-1.045x BF16; throughput is stable at 93.0-93.2%, about 1.9 percentage
  points below the 95% gate. Mean TTFT is 11.04-11.07 s versus 9.62 s for
  BF16, while enabling native fused Hadamard scatter is neutral (55.23 tok/s,
  50.58 ms TPOT). The next matched discriminator disables only the native
  graph reader while retaining compact cache write/rotation, separating
  prefill/cache synchronization cost from native decode cost. Artifacts:
  `/tmp/kvarn-k3d1-dual-preload-graph-matched-6k-c4-o512-r{1,2}.json` and
  `/tmp/kvarn-k3d1-dual-preload-fused-scatter-graph-matched-6k-c4-o512-r1.json`.
  Earlier stride-only eager and graph replay tests rule out record spacing
  itself.
  The matched compact-cache/Triton-reader discriminator rejects moving decode
  back to Triton: its cold run is 35.95 tok/s and 79.82 ms TPOT, while the
  warm repeat is 40.12 tok/s and 79.90 ms TPOT. Warm TTFT falls to 9.92 s, near
  the 9.62 s BF16 control, but decode remains about 57% slower than the native
  reader. The cold run also JIT-compiles the record-flush and packed-KV
  materialization kernels; the stable warm TPOT proves that JIT is not the
  steady-state gap. Artifacts:
  `/tmp/kvarn-compact-triton-reader-graph-matched-6k-c4-o512-r{1,2}.json`.
  Combining native K/V FWHT-scatter with Q FWHT initially exposed that the
  combined kernel incorrectly required Q and K/V to share a dtype; production
  uses fp16 Q with bf16 K/V. The kernel now dispatches Q and K/V independently,
  and a focused mixed-dtype device oracle passes. The first graph run pays
  flush/materialization JIT (45.69 s TTFT), while the warm run is effectively
  neutral at 55.08 tok/s, 50.70 ms TPOT, and 11.09 s TTFT. Retain the corrected
  opt-in implementation but leave fused query/store disabled by default; launch
  fusion does not close the gate. Artifacts:
  `/tmp/kvarn-k3d1-dual-preload-fused-query-store-graph-matched-6k-c4-o512-r{1,2}.json`.
  Explicit-greedy same-source controls reach 60.40 tok/s, 46.65 ms TPOT, and
  9.92 s TTFT for BF16 versus 55.78 tok/s, 49.53 ms TPOT, and 11.23 s TTFT for
  compact K4V4. Compact therefore passes TPOT at 1.062x but remains at 92.35%
  of BF16 throughput. The previously logged 47.45 ms historical compact TPOT
  cannot be used as a regression oracle because its scratch artifact is no
  longer present and its sampling policy is unverifiable. Artifacts:
  `/tmp/{bf16,kvarn-k3d1-dual-preload}-greedy-graph-matched-6k-c4-o512-r1.json`.
  A device-wide cache-producer synchronization discriminator cannot execute
  inside XPU graph capture: `torch.xpu.synchronize()` fails with `wait cannot
  be called for a queue which is recording to a command graph`. On this path
  `is_current_stream_capturing()` reports false and the B4 decode update is
  statically padded to the compile range, so neither that predicate nor tensor
  shape can guard the wait. The opt-in synchronization hook is now explicitly
  eager-only. Test graph publication ordering with a graph-recordable event or
  a fence outside the captured `unified_kv_cache_update` split instead.
  Skipping the complete compact cache update is an intentionally incorrect
  timing discriminator, not an accuracy candidate. Its cold run pays flush
  and packed-reader JIT, but the warm repeat reaches 57.34 tok/s, 48.22 ms
  TPOT, 10.90 s TTFT, and 35.71 s duration. Relative to the matched greedy
  compact control (55.78 tok/s, 49.53 ms, 11.23 s, 36.71 s), cache production
  therefore accounts for 1.31 ms/step plus 324 ms of mean prefill latency--
  essentially the full one-second duration reduction needed for the 95%
  throughput gate. Split rotation from scatter next with
  `KVARN_DIAGNOSTIC_SKIP_CACHE_SCATTER=1`. Artifacts:
  `/tmp/kvarn-skip-cache-update-greedy-graph-matched-6k-c4-o512-r{1,2}.json`.
  Keeping the two fallback FWHT rotations while skipping only scatter reaches
  56.97 tok/s, 48.55 ms TPOT, and 10.97 s TTFT. The apparent upper bounds are
  0.98 ms/step and 258 ms of TTFT for scatter/publication, plus 0.33 ms/step
  and 66 ms for rotations. These are intentionally invalid-cache runs, though,
  so changed activations can affect downstream timing; use XPU events on the
  valid path before treating the deltas as direct kernel attribution. Artifact:
  `/tmp/kvarn-skip-cache-scatter-greedy-graph-matched-6k-c4-o512-r1.json`.
  The earlier native-scatter result may have reconstructed an AOT graph built
  with the environment branch disabled, so native scatter was recompiled in a
  fresh dedicated cache. Its warm graph run reaches 56.13 tok/s, 49.37 ms TPOT,
  and 11.08 s TTFT: a real but insufficient 0.16 ms/step improvement over the
  fallback control. The standalone native producer still costs one launch per
  full-attention layer; the bypass bound shows that removing publication's
  launch boundary, rather than tuning its FWHT arithmetic, is the remaining
  high-value direction. Artifacts:
  `/tmp/kvarn-native-scatter-freshcache-greedy-graph-matched-6k-c4-o512-r{1,2}.json`.
  Fused query/store was likewise recompiled in a fresh cache so its environment
  branch is authoritative. Its warm graph run reaches 56.21 tok/s, 49.32 ms
  TPOT, and 11.06 s TTFT--only 0.05 ms/step faster than standalone native
  scatter. Folding the two native commands therefore does not reproduce the
  invalid-cache bypass ceiling. Artifact:
  `/tmp/kvarn-fused-query-store-freshcache-greedy-graph-matched-6k-c4-o512-r{1,2}.json`.
  A narrow device-event microbenchmark confirms the combined command is doing
  less standalone work: B4 query FWHT is 57.37 us median, K/V scatter is
  50.91 us, and combined query/store is 50.31 us. The absent ~0.9 ms/layer-set
  saving in graph TPOT shows this work is hidden or amortized during replay;
  it is not the remaining steady bottleneck.
  Enabling the provisionally retained fused split-16 reducer/H256 together
  with fused query/store in another fresh cache reaches 56.31 tok/s, 49.19 ms
  TPOT, and 11.06 s TTFT. That is only 0.13 ms/step faster than the current
  unfused compact control and still 93.23% of the 60.40 tok/s greedy BF16
  control. The historical fused-reducer gain does not reproduce with current
  source, corrected symbol binding, graph capture, and sampling. Artifacts:
  `/tmp/kvarn-fused-all-freshcache-greedy-graph-matched-6k-c4-o512-r{1,2}.json`.
  An exact-shape device comparison now localizes the remaining steady gap to
  the native decoder itself. At B4, four 6000-token rows, QH24/KVH4, D256,
  BF16, and page size 128, packaged FlashAttention repeats at 164.00, 164.18,
  and 164.19 us per call. The current narrow K3/D1 DPAS split-16 decoder is
  237.11 us (unfused reducer) or 238.72 us (fused reducer). The 72.9-74.7 us
  per-layer deficit predicts 2.33-2.39 ms across 32 layers, accounting for
  essentially all of the measured 2.54-2.88 ms compact-versus-BF16 TPOT gap.
  The fused reducer slightly slows the isolated native call, so leave it off.
  Inspect synchronization and packed-fragment work in the mainloop next;
  publication launch fusion is no longer the primary steady-state target.
  Removing the compact mainloop's end-of-tile workgroup barrier is safe in the
  narrow path: packed fragments, softmax state, and output accumulators are
  subgroup-local, and 44 decoder/layout, ragged/hybrid, split-equivalence, and
  allocator-pressure cases pass. (Thirteen standalone reducer/coordinate
  tests require schemas deliberately absent from the benchmark-only shim and
  were not part of that result.) Performance moves only from 237.11 to
  235.68 us, so this synchronization point is unnecessary but explains just
  1.43 us of the roughly 73 us per-layer deficit. Artifact:
  `/tmp/kvarn-current-nobarrier-dpas-s16-b4-6k.json`.
  A timing-only factor bypass then keeps the DPAS-packed nibble reader but
  omits all K3/D1 row/column scale and zero-point application. Its output is
  intentionally invalid, yet the isolated device median falls from 235.68 to
  103.33 us. The correct factor path was restored and rebuilt immediately; a
  short smoke repeat returns to 235.65 us. This 132 us upper bound is larger
  than the full gap to BF16 and makes repeated factor metadata loads,
  arithmetic, subgroup reductions, and their register pressure the next
  optimization target. In particular, each 128-token record is visited as two
  64-token tiles while its K/V column factors are loaded and applied in both.
  Artifact: `/tmp/kvarn-diagnostic-no-k3d1-factors-dpas-s16-b4-6k.json`.
  Separate invalid-output discriminators measure 199.53 us with K factors
  bypassed and 187.01 us with V factors bypassed. Within V, bypassing only
  zero point, token-row scale, or output-column scale reaches 231.77, 231.09,
  and 216.41 us respectively. The non-additive savings confirm that the full
  factor block also changes register allocation; dropping D1 would save only
  about 3.9 us and is not justified by performance. Artifacts:
  `/tmp/kvarn-diagnostic-no-{k-factors,v-factors,v-zp,v-row-scale,v-col-scale}-dpas-s16-b4-6k.json`.
  Retain a representation-preserving V optimization instead. DPAS now
  accumulates directly into the online-softmax output in the current record's
  V column-scale frame, keeping that frame across both 64-token halves of the
  128-token record and closing it at page, split, or sequence boundaries. This
  removes the second full accumulator fragment and halves scale-frame
  conversions. The complete dual-preload decoder suite passes 57/57, including
  nonuniform page factors, ragged/hybrid boundaries, split equivalence, and
  allocator pressure. The B4/6000 device median is 217.50 us versus 235.68 us
  after barrier removal and 237.11 us before it. Artifact:
  `/tmp/kvarn-paired-v-scale-frame-dpas-s16-b4-6k.json`.
  Two matched greedy graph repeats complete 4/4 exact 6000/512 requests at
  56.45/56.48 tok/s, 48.89/48.98 ms TPOT, and 11.12/11.06 s TTFT. This is a
  stable 0.67-0.70 tok/s and 0.55-0.64 ms TPOT improvement over the 55.78
  tok/s, 49.53 ms compact control, consistent with 32 layers times the 18.18
  us isolated gain. Graph capture selects native decode, capacity remains
  270,336 tokens / 33x at 8192, and no device fault or invalid output occurs.
  Throughput is still only 93.5% of the 60.40 tok/s BF16 control and below the
  57.38 tok/s ship gate, so continue with K page-factor reuse. Artifacts:
  `/tmp/kvarn-paired-v-frame-greedy-graph-matched-6k-c4-o512-r{1,2}.json`.
  A new benchmark-only KVarN target reduces local rebuilds from 643 generic
  attention objects to one decoder object plus one registration shim. Split-1
  per-call versus persistent scratch is neutral (5428.594 versus 5428.073 us),
  ruling out allocation overhead. Repaired standard split-16 reaches 1417.812
  us and remains neutral with persistent scratch (1414.479 us), a 3.83x
  isolated speedup over split-1. Artifacts:
  `/tmp/kvarn-recordstream-split16-{alloc,persistent}-b4-6k.json`.
  K-factor attribution on top of the paired-V path reaches 212.86 us with K
  zero point bypassed, 214.64 us with K row scale bypassed, and 132.89 us with
  K column scale bypassed. Moving the K per-dimension column scale from Q onto
  the reconstructed DPAS K operand is algebraically valid and passes the full
  57/57 dual-preload decoder suite. It lowers the B4/6000 isolated median from
  217.50 to 159.53 us, slightly faster than matched native BF16 attention at
  164.00-164.19 us. Artifact:
  `/tmp/kvarn-k-scale-on-dpas-operand-paired-v-s16-b4-6k.json`.
  The isolated gain survives graph replay: matched 4x6000/512 warm repeats are
  57.33/57.36 tok/s with 47.83/47.87 ms mean TPOT (47.08/47.11 ms median),
  native dispatch selected, and unchanged 270,336-token / 33x capacity. This
  is within 0.02 tok/s of the 57.38 ship threshold and materially faster than
  the paired-V control. Artifacts:
  `/tmp/kvarn-k-scale-on-dpas-greedy-graph-matched-6k-c4-o512-r{1,2}.json`.
  An initial full64 run produced all-NaN logits, but this did not implicate the
  decoder: K-scale-on-Q, native decode disabled, fast flush disabled, and
  canonical layout all reproduced the exact failure. Synchronized per-layer
  validation localized the first corruption to the all-NaN input of attention
  layer 11. The test preload had omitted the locally built GDN library carrying
  the A/W/U scratch `recordStream` fix, allowing asynchronous GDN scratch reuse
  to reappear under compact allocator pressure. Preloading the narrow decoder,
  local fixed GDN, then packaged fallback makes the one-step synchronized probe
  finite. The authoritative paired full64/full-vocabulary rerun also passes:
  MAE 0.011389, RMSE 0.016875, p95 0.03125, max 0.171875, 96.875% exact top-1,
  and 100% BF16-tie-aware agreement. This is slightly better than the accepted
  Triton MAE (~0.0115), so retain the K-operand scale optimization. Artifacts:
  `/tmp/kvarn-accuracy-{bf16-paired-fixed-gdn,compact-native-k-scale-fixed-gdn}-full64.npz`.
  Repeating the matched graph profile with the same three-library preload
  confirms that the synchronization fix does not cost decode performance.
  The cold run triggers one-time `_kvarn_flush_record_kernel` and
  `_kvarn_build_packed_kv_kernel` JITs, but the warm 4x6000/512 run completes
  4/4 requests at 57.32 tok/s with 47.97 ms mean TPOT (47.21 ms median) and
  11.05 s mean TTFT. Native decode is selected during full graph capture and
  for B=1..4 replay. At the explicitly matched 0.90 GPU-memory utilization,
  the server reports 204,800 cache tokens / 25x at 8192; the earlier 270,336
  / 33x figure therefore came from a higher memory allowance and must not be
  compared as if it were the same setting. Two immediate 0.98 restarts failed
  during XPU process initialization after shutdown, before model loading, so
  retain 204,800 / 25x as the clean capacity result and retest the larger
  allowance only in a fresh device session. Artifacts:
  `/tmp/kvarn-k-scale-fixed-gdn-greedy-graph-matched-6k-c4-o512-r{1,2}.json`.
  A fresh-AOT timing discriminator with
  `KVARN_DIAGNOSTIC_SKIP_CACHE_UPDATE=1` and the same fixed-GDN preload makes
  the remaining gap unambiguous. The warm invalid-cache run reaches 58.53
  tok/s and 46.82 ms mean TPOT (46.08 ms median), versus 57.32 tok/s and
  47.97 ms on the valid compact path. Thus cache production costs about 1.15
  ms/step in this paired run and removing it brings KVarN TPOT within 0.17 ms
  of the 46.65 ms BF16 control. The cold discriminator also measures 46.63 ms
  TPOT despite its one-time prefill JIT delay. Keep this strictly as timing
  evidence: skipping writes makes attention read invalid cache contents. The
  next corrected-stack discriminator is skip-scatter while retaining both
  rotations, followed by event timing of the valid producer if that split
  still attributes most of the cost. Artifacts:
  `/tmp/kvarn-k-scale-fixed-gdn-skip-cache-update-graph-6k-c4-o512-r{1,2}.json`.
  The corresponding fresh-AOT skip-scatter repeats (rotations retained) reach
  58.25 tok/s and 47.12 ms mean TPOT warm, with the cold run at 46.91 ms.
  Against the valid 47.97 ms path, publication is therefore about 0.85 ms of
  the 1.15 ms producer cost; the two fallback rotations account for only the
  remaining ~0.30 ms. This confirms publication is again the primary target
  after the decoder's K-factor optimization. Re-evaluate the already-correct
  native fused Hadamard/scatter path with the optimized decoder and fixed GDN:
  its earlier ~0.16 ms gain would now be enough to cross the 57.38 tok/s ship
  gate. Artifacts:
  `/tmp/kvarn-k-scale-fixed-gdn-skip-cache-scatter-graph-6k-c4-o512-r{1,2}.json`.
  The real native fused Hadamard/scatter producer then clears the ship gate in
  two stable warm repeats: 57.45/57.43 tok/s with 47.82/47.79 ms mean TPOT
  (47.06/47.03 ms median), versus the 57.38 tok/s threshold. Native decoder
  dispatch remains selected through graph capture and replay; capacity stays
  204,800 tokens / 25x at the matched 0.90 memory setting. Its authoritative
  eager full64/full-vocabulary run is fully finite and improves the paired
  BF16 comparison to MAE 0.010972, RMSE 0.016197, p95 0.03125, max 0.171875,
  with 100% exact and tie-aware top-1 agreement. Promote native scatter in the
  separate native compact eager and graph profiles; keep generic compact
  profiles on their established producer. Artifacts:
  `/tmp/kvarn-k-scale-fixed-gdn-native-scatter-graph-6k-c4-o512-r{1,2,3}.json`
  and
  `/tmp/kvarn-accuracy-compact-native-k-scale-native-scatter-fixed-gdn-full64.npz`.
  The current narrow-first/fallback load order passes all 11 focused native
  Hadamard/scatter cases (fp16/bf16, N=1/4/17, structured invalid rows,
  repeat determinism, non-contiguous production strides, page-boundary
  appends, and mixed query/KV dtypes). Both promoted Nix app wrappers evaluate
  successfully and `nixfmt --check nix/chat-profile.nix` passes.

  A bounded XPU-event trace of the valid fused producer rules it out as the
  remaining long-prefill bottleneck. Across the 17 full-attention consumers
  of a 6K request, the 2048/1664/240-token chunks total roughly 4 ms of device
  time. The temporary synchronizing profiler hook was removed immediately
  after attribution. Combined with the roughly 0.8--0.9 s measured Sinkhorn
  work, the remaining TTFT delta is in retirement/metadata enqueue and queue
  interaction rather than the Hadamard/scatter kernel itself.

  Removing the builder's `slot_mapping.tolist()` synchronization is rejected
  for now. The qlen-3/B4 seed-0 discriminator fell to 65.80 tok/s and 88.07%
  draft acceptance; restoring it returned 68.00 tok/s and 92.50%, matching the
  established 68.29 tok/s / 92.62% baseline. A bounded follow-up trace found
  that slot mapping contributed *no additional block IDs* in the exercised B4
  run. The D2H read is therefore acting as an accidental queue barrier, not an
  ownership-set input, exposing a likely graph/cache cross-stream dependency.
  Keep it until that dependency is represented explicitly. Artifacts:
  `/tmp/kvarn-compact-mtp-noslot-sync-seed0.json` and
  `/tmp/kvarn-compact-mtp-slot-sync-restored-seed0.json`.

  Root cause is pinned-host metadata reuse: the builder overwrote its single
  `cu_seqlens` and verify-plan staging buffers while prior nonblocking H2D
  copies could still consume them. Replace them with an eight-entry staging
  ring, with an XPU event guarding each entry. This removes the accidental
  device-wide D2H barrier while waiting only if eight scheduler steps outrun
  the small metadata copies. The first clean qlen-3/B4 seed-0 run reaches
  67.80 tok/s, 32.11 ms TPOT, and 92.96% acceptance at unchanged 98,304-token
  / 12x capacity. Artifact:
  `/tmp/kvarn-compact-mtp-metadata-ring-seed0.json`. A later seed-1 run hit
  prefix cache because an accidentally concurrent diagnostic had already used
  its prompts; that run and the concurrent pair are excluded from all gates.
  Fresh-server sequential seeds 1 and 2 reach 69.94/65.26 tok/s, 31.58/33.40
  ms TPOT, and 98.48%/78.84% acceptance. Together with seed 0, the ring median
  is 67.80 tok/s and 32.11 ms TPOT. It fixes the race and removes the global
  synchronization, but does not by itself close the 69.22 tok/s throughput
  ship gate. Artifacts:
  `/tmp/kvarn-compact-mtp-metadata-ring-fresh-seed{1,2}.json`.
  Re-recording the eight event objects is rejected: seed 0 falls to 64.48
  tok/s and 86.30% acceptance. Allocate a fresh completion event for each copy
  generation; only the pinned staging storage is reused. Artifact:
  `/tmp/kvarn-compact-mtp-metadata-ring-reuse-seed0.json`.

  Increasing the guarded staging ring from 8 to 256 entries removes nearly
  all periodic scheduler-side event retirement from the 512-token benchmark.
  Deterministic seeds 0/1/2 reach 68.61/70.24/65.59 tok/s with
  32.01/31.41/34.16 ms TPOT and 94.50%/98.12%/75.95% acceptance. The 68.61
  tok/s median improves over 67.80 but remains below the 69.22 ship gate.
  A 1024-entry endpoint is
  rejected: it falls to 66.56 tok/s, 32.55 ms TPOT, and 89.70% acceptance,
  while consuming four times the pinned-host storage. Retain 256 provisionally
  and require its full three-seed median before evaluating the ship gate.
  Artifacts: `/tmp/kvarn-stage{256,1024}-seed0.json`.
  An event-free 8192-generation ring with one synchronization at wraparound is
  also rejected: seed 0 reaches 67.69 tok/s, 32.08 ms TPOT, and 91.90%
  acceptance. Per-step event recording is therefore not the remaining
  throughput limiter. Artifact: `/tmp/kvarn-stage8192-noevents-seed0.json`.

  A fresh matched BF16 control on the current source/system reaches
  72.96/78.04/71.16 tok/s for seeds 0/1/2 (median 72.96), with
  31.05/29.83/33.21 ms TPOT. The historical baseline remains representative.
  Passing the framework-maintained CPU block-table view into common attention
  metadata removes KVarN's remaining unconditional `block_table.cpu()` D2H
  queue barrier. After separating that bookkeeping view from the device table
  consumed by graph kernels, deterministic KVarN seeds reach
  69.29/72.12/66.18 tok/s (median 69.29) and 31.80/30.74/34.44 ms TPOT. The
  median ratio is 94.97%, 0.03 percentage points below the literal 95% gate;
  do not round it to a pass. The qlen-3 lifecycle remains exact after barrier
  removal: two drafts per verification, identical replay, acceptance lengths
  1/2/3, and consecutive-rejection recovery. Artifacts:
  `/tmp/bf16-mtp-current-seed{0,1,2}.json`,
  `/tmp/kvarn-cpu-block-table-seed{0,1,2}.json`, and
  `/tmp/kvarn-cpu-block-table-lifecycle.json`.
  Increasing the guarded ring to 512 after the CPU block-table change is
  rejected: seed 0 falls to 65.91 tok/s, 32.45 ms TPOT, and 88.57% acceptance.
  Retain 256; the larger ring changes queue ordering and numerical acceptance
  rather than providing a simple retirement-wait reduction. Artifact:
  `/tmp/kvarn-stage512-cpu-block-seed0.json`.
  Re-testing the previously corrupt shared-dequant qlen-3 Triton verifier after
  both synchronization fixes confirms the diagnosis: its full lifecycle now
  passes (two drafts, identical replay, acceptance lengths 1/2/3, rejection
  recovery), where the old metadata stack produced invalid proposals. It is
  nevertheless rejected for production because B4 serving reaches only 27.12
  tok/s and 116.32 ms TPOT. Native B12 routing remains the supported path.
  Artifacts: `/tmp/kvarn-shared-verify-post-sync-{lifecycle,seed0}.json`.
  Native Xe2 Hadamard transforms around qlen-3 verification are also rejected:
  seed 0 reaches 68.22 tok/s with no TPOT improvement, as the small numerical
  change lowers MTP acceptance enough to erase the transform saving. Artifact:
  `/tmp/kvarn-native-verify-hadamard-seed0.json`.
  A small-burst cross-layer Sinkhorn flush is rejected as well. Although it
  preserves the exact qlen-3/two-draft verifier contract, batching the usual
  17 decode-boundary layer pairs reaches only 67.38 tok/s, 31.98 ms TPOT, and
  90.85% acceptance on seed 0. Keep the simpler per-layer flush path. Artifact:
  `/tmp/kvarn-small-cross-layer-flush-seed0.json`.
  Hoisting qlen-3's virtual block-table expansion out of all 17 attention
  layers passes the exact two-draft lifecycle, but is also rejected: seed 0
  reaches 67.41 tok/s and 32.32 ms TPOT versus 67.81 tok/s and 32.15 ms for
  the retained four-iteration control. The repeated tiny index operations are
  evidently hidden by queue execution rather than the remaining limiter.
  Artifacts: `/tmp/kvarn-vq-table-hoist-{lifecycle,seed0}.json`.
  Re-testing the faster isolated split-4 verifier after the synchronization
  fixes confirms that synchronization was not the reason it lost in serving.
  Its exact two-draft lifecycle passes, but seed 0 reaches only 64.80 tok/s
  and 34.95 ms TPOT. Retain split 32. Artifacts:
  `/tmp/kvarn-split4-post-sync-{lifecycle,seed0}.json`.
  A KVarN-specific split-32 reduction workgroup preserves the generic
  reducer's serial split order while avoiding its broad launch geometry. At
  B12/context 6000, isolated device median falls from 456.17 to 434.06 us.
  The integrated qlen-3 lifecycle reproduces the authoritative 221 verifier
  steps, 442 drafts, 355 accepted drafts, acceptance-length counts 30/27/164,
  identical replay, and consecutive-rejection recovery. Deterministic serving
  seeds reach 68.99/72.20/65.95 tok/s (68.99 median) and
  31.97/30.91/34.43 ms TPOT (31.97 median). Against the adjacent BF16 medians
  of 71.94 tok/s and 29.98 ms, this is 95.89% throughput and 1.066x TPOT, so
  both ship gates pass. Prefix reuse remains exact at 127/128/129/4096 tokens;
  maximum comparable logprob drift is 0.1821, inside the matched-BF16 0.5
  envelope. Artifacts: `/tmp/kvarn-native-b12-c6000-reduce32-generic-order.json`,
  `/tmp/kvarn-reduce32-generic-order-{lifecycle,prefix}.json`, and
  `/tmp/kvarn-reduce32-generic-order-seed{0,1,2}.json`.
  Remove the abandoned fused reduce-plus-Hadamard flag/tests and the rejected
  shared-dequant verifier path; neither is part of the shipped native B12 MTP
  route.

  The retained fresh-event ring passes the full rejection/cache lifecycle gate
  with the prior authoritative counts exactly: qlen 3, 221 verification steps,
  442 drafts, 355 accepted (80.32%), identical replay, every acceptance length,
  and recovery after consecutive first-draft rejection. Artifact:
  `/tmp/kvarn-compact-mtp-metadata-ring-lifecycle.json`.
  Prefix reuse produces identical decoded tokens. The old global 0.125 logprob
  threshold is invalid as a corruption discriminator: matched BF16 MTP controls
  reach 0.249 at length 127 and 0.437 at length 129 despite identical output.
  Compact measures 0.125029 on the established seed and 0.183944 on a distinct
  seed, both inside the BF16 envelope. Set the numerical bound to a rounded 0.5
  while retaining exact token/text/finish equality as the hard gate.

## MTP capacity redesign checkpoint

- [x] Reproduce the apparent tiny-request cache pressure as exact allocator
  arithmetic. The deployed Mamba pool had eight physical pages, one permanent
  null page, and four peak align-mode MTP2 states per request. One peak request
  therefore reported `4 / 7 = 57.1%`; a second peak needed nine physical pages.
- [x] Fix independent-pool capacity reporting to exclude the null page and log
  physical, null, and usable counts separately.
- [x] Replace maximum-context request bundles with separate policies: reserve
  `max_num_seqs` peak Mamba state bundles, then expose the remaining attention
  pages as a shared token pool. Fail configuration if that residual pool cannot
  hold one `max_model_len` request.
- [x] Reconcile asymmetric multi-worker cache layouts per physical pool instead
  of scaling every independent pool through the legacy aggregate block count.
- [x] Add a real allocator regression proving that two simultaneous align-mode
  MTP2 requests reach a four-state peak only with nine physical Mamba pages.
  Drive two requests through acceptance lengths 1/2/3, require disjoint state
  page identities, and verify complete teardown reclamation.
- [x] Pass both complete CPU allocator suites (115 tests), pinned Ruff, and the
  local-source `vllm-xpu-unstable` Nix build. Local vLLM commits are
  `4adb725b12`, `87c4a24c7d`, and `a403ba1626`; pushing is deferred until SSH
  is available.
- [x] Derive the next-launch capacity expectation from the last 5.47-GiB
  profile (`max_num_seqs=4`): each of the three 16-layer Mamba pools grows from
  the invalid eight pages to 17 physical pages (one null plus 16 usable), and
  the 17-layer compact attention pool retains approximately 1,351 physical
  pages / 172,800 usable tokens. Exact startup values may differ by one page
  because the journal rounds available memory to two decimals.
- [x] Remove the stale shared-dequant verifier interface and routing comments;
  the implementation had already been rejected and replaced by the retained
  native qlen-3 verifier plus virtual-query Triton fallback. Focused KVarN
  metadata/config coverage passes 38/38. Local cleanup commit: `965941d6f3`.
- [x] Audit graph dispatch semantics against the pinned vLLM source. Capture
  sizes are token counts and two-token MTP has uniform decode qlen 3, so sizes
  3/6 cover B1/B2. B3/B4 submit 9/12 tokens, exceed the configured maximum,
  and explicitly dispatch with graph mode `NONE`. This is the intended graph
  VRAM/concurrency tradeoff, not a hidden fallback; live validation must prove
  graph replay for B1/B2 and eager execution plus parity for B3/B4.
- [x] Combine independent pools, align-mode MTP2, prefix reuse, and cancellation
  in one allocator lifecycle: the replay request receives the scheduler's
  shared-prefix boundary, reuses cached attention/recurrent pages, keeps its
  three private target/draft pages separate, and returns those private pages
  on abort. A scheduler-level B4 regression also admits four tiny MTP2
  requests, aborts one, and immediately admits a replacement through the same
  null-aware pools. Local vLLM commits: `c38f6d171e`, `5993077257`.
- [x] With Brutus vLLM still disabled, build the exact local source through the
  deploy-equivalent `vllm-xpu-chat` output and rerun the matched eager
  BF16/native-KV oracle before graph mode. Both modes produce the identical B4
  greedy-token SHA-256
  `b3cbecf4768c455c52cdf7fc42ce050366c13f4404fb38bd53e57d4a16a36ec9`.
  BF16 records 221 verification steps, acceptance lengths 30/27/164, and
  355/442 accepted drafts (80.32%); compact K4V4 records 224 steps,
  33/30/161, and 352/448 (78.57%), a passing -1.75 percentage-point delta.
  Both exercise consecutive rejection/recovery and deterministic replay.
  Artifacts: `/tmp/bf16-mtp-eager-current-source-with-tokens.json` and
  `/tmp/kvarn-compact-mtp-eager-current-source-with-tokens.json`.
- [ ] Re-enable graph sizes 3/6 only after eager parity, prove B1/B2 graph
  replay and B3/B4 intentional eager fallback, then rerun qlen-3
  lifecycle, prefix 127/128/129/4096, one 114688-token request, two/four tiny
  concurrent requests, mixed contexts, cancellation, and B2/B4 performance.
- [x] Root-cause the remaining compact+prefix corruption at the cache boundary.
  Independent physical pools incorrectly retained a 16-token Mamba logical
  boundary while compact attention used 128 tokens. MTP verification could
  therefore publish a recurrent candidate for a different prefix than the
  attention cache. Align-mode KVarN now shares the attention logical boundary
  without padding or merging the physical recurrent pages. The minimal patch,
  with the investigative synchronization barriers removed, exactly matches the
  BF16 B4 greedy-token SHA-256 above in both eager and graph-enabled launches.
  Local vLLM commits: `8dd7421d74`, `7bd0de0484`.
- [x] Capture graph sizes 3 and 6 in both mixed prefill/decode PIECEWISE and
  decode FULL modes. The B4 lifecycle exercises live native KVarN dispatch at
  B=1, B=2, and B=4, produces qlen 3 with two speculative tokens, recovers from
  consecutive rejection, and repeats the BF16 token SHA exactly. Sizes 3/6
  cover B1/B2; the previously audited B3/B4 9/12-token work remains the
  intentional eager fallback. Warm B4 gate duration is about 4.2-5.0 seconds.
  Artifacts: `/tmp/kvarn-compact-mtp-graph-prefix-shared-logical-boundary.json`,
  `/tmp/kvarn-compact-mtp-graph-prefix-shared-logical-boundary-replay.json`,
  and `/tmp/kvarn-compact-mtp-graph-minimal-final.json`.
- [x] Revalidate prefix and cancellation on compact graph+MTP2. Prefix reuse at
  127/128/129/4096 tokens preserves decoded output exactly; maximum comparable
  logprob drift is 0.312, inside the matched BF16 0.5 envelope. A forced client
  disconnect returns reported cache usage to 0%, and the immediate B4 recovery
  request again produces the exact BF16 SHA. Artifacts:
  `/tmp/kvarn-compact-mtp-graph-prefix-reuse-final.json` and
  `/tmp/kvarn-compact-mtp-graph-post-cancel.json`.
- [x] Prove the full-context allocator envelope without reducing the configured
  maximum. At `max_model_len=114688`, startup exposes 341,760 compact attention
  tokens (2.98 maximum-length requests) while each independent Mamba pool still
  admits four requests. This proves one maximum-length request is schedulable
  and four shorter requests can coexist; actually filling a 114,688-token
  request and a fresh matched post-fix B2/B4 performance run remain pending.
- [x] Fill the configured context ceiling with a real request. A compact
  graph+MTP2 server accepted and completed one 114,687-token random prompt plus
  one generated token (114,688 total) in 547.43 seconds at 209.50 total tok/s.
  Live usage rose through 68.75% and 93.75%, then returned to 0% with zero
  running/waiting requests. Artifact:
  `/tmp/kvarn-compact-mtp-graph-max114688-real-request.json`.
- [ ] Pass the fresh post-boundary-fix B4/6K/O512 performance gate. The warm
  seed-20260809 compact run completes 4/4 at 51.53 output tok/s, 32.33 ms TPOT,
  15.72 s TTFT, and 84.45% draft acceptance. Its matched BF16/native-KV run
  reaches 74.94 tok/s, 30.74 ms TPOT, 10.72 s TTFT, and 92.43% acceptance.
  Compact passes the 1.10x TPOT limit at 1.052x but fails throughput at 68.76%
  of BF16. Do not ship based on the older passing median; isolate the
  long-prefix acceptance and TTFT gaps first. Artifacts:
  `/tmp/{kvarn-compact,bf16}-mtp-boundaryfix-graph-6k-c4-o512-warm.json`.
- [x] Separate captured B2 execution from intentional B4 eager fallback with a
  literal seed-0 matched run. At B2/6K/O512 compact reaches 58.78 tok/s,
  19.27 ms TPOT, 7.53 s TTFT, and 95.45% acceptance versus BF16's 61.07 tok/s,
  19.80 ms, 6.52 s, and 97.41%. The 96.25% throughput and 0.973x TPOT ratios
  pass both gates, so graph size 6 and qlen-3 state ownership are not the broad
  regression. Artifacts:
  `/tmp/{kvarn-compact,bf16}-mtp-boundaryfix-b2-graph-6k-o512-seed0.json`.
- [ ] Restore B4 prefill/TTFT performance after the shared-boundary correctness
  fix. A direct literal seed-0 compact repeat reaches 51.24 tok/s, 31.53 ms
  TPOT, 16.36 s TTFT, and 93.62% acceptance, versus the pre-fix artifact's
  68.99 tok/s, 31.97 ms, 12.57 s, and 91.97%. Stable TPOT and acceptance with
  slower/staggered first tokens localize the regression to chunked prefill and
  scheduling, not decode arithmetic. The scheduler admits 6K requests in
  2048-token chunks (temporarily reporting 2 running/2 waiting), while Mamba
  usage remains within the four-page-per-request contract. Artifact:
  `/tmp/kvarn-compact-mtp-boundaryfix-b4-graph-6k-o512-seed0.json`.
