# D256 attention: query tile, adversarial review and adjacent K64 experiment

The Q128/16-subgroup hypothesis in issue #11 is rejected for the frozen B70
workload. The implementation is numerically correct but slower. Restoring
baseline prefetch packets recovers part of the loss without producing a win.
An adjacent Q256/K64 policy also passes correctness but loses the bounded
long-context screen. Keep the serving Q256/K32 policy. No candidate was
advanced to service qualification, deployment or release.

## Results

The first A/B predicts these changes in **attention-call cost**, counting every
observed prefill layer call and interpolating unmeasured full-query extents:

| Cache format | 16383-token prompt | 65023-token prompt |
| --- | ---: | ---: |
| Auto, paged BF16 | 26.71–37.21% slower | 58.53–63.76% slower |
| Compact KVarN, dense FP16 | 33.98–38.97% slower | 54.56–57.98% slower |

Ranges cover both starts and warm, streaming-conditioned and interleaved
conditions. These are operator predictions, not TTFT or model-service results.
The inventory includes 128 calls for 16K and 512 calls for 65K, spanning all
chunks and 16 ordinary attention layers. Respectively 80 and 432 calls use
interpolated anchors. The two ragged final chunks use measured costs directly.

Three skeptical agents reviewed implementation, measurement and code generation.
The strongest implementation challenge was an implicit prefetch change. A
bounded packet control restored K packets of 32x1 elements and V packets of
32x8, with two packets per subgroup and complete coverage. At Q2048/K63488:

| Follow-up | Paged BF16 vs original | Dense FP16 vs original |
| --- | ---: | ---: |
| Q128/SG16, restored packets | 54.94–62.53% slower | 44.19–54.55% slower |
| Q256/SG32, key tile 64 | 1.73–2.81% slower | 4.25–5.96% slower |

The packet control reduces latency relative to the first Q128 candidate by
4.08–5.28% paged and 6.13–9.87% dense. That is partial recovery, not a speedup
over the existing kernel. Its compiler also unrolls more packet-loop code, so
it isolates an implementation alternative rather than just hardware packet size.

## What the experiment established

Frozen substrate: xpu-v1.7.2 profiling runtime
`/nix/store/88fvzjx1ar98dlm9mdw715c22zz1r2i0-vllm-xpu-profile-env`, vLLM
`9bfcf123f675c6b668391cb536ba06949174afd4`, kernels
`23d8103a97982c169986f65bba593885c91ad444`, Torch 2.13.0+xpu, oneAPI
2026.0.1.27, Intel Arc Pro B70. Checkpoint revision:
`6b0622f4354481d5d04577d48ba0db844efc1330` of the pinned BF16/W4A16 Qwen27B.

The six Q/K extents are 2048/2048, 2048/8192, 2047/16383, 2048/32768,
2048/63488 and 1535/65023, with Hq24/Hkv4/D256 and bottom-right causal masking.
Captures preserve values, strides, physical pages, K/V aliasing, every call
extent, exact prompt IDs and 512 output IDs per diagnostic request. Auto uses
832-token pages, not an assumed 64-token page size. Its K/V views share storage
with offset 256 and strides `[1703936,2048,512,1]`. Compact KVarN supplies dense
FP16 at every selected extent.

The four-line Q128 change compiles legally and selects the intended native
kernel. It retains eight query rows per subgroup, DPAS8 and the same accumulator
shape. Q2048/H24 changes from 192 to 384 workgroups while total subgroups remain
6144. All selected original/control/Q128 kernels report GRF256, 4096-byte SLM
allocation and no emitted spill buffers. Thus the change did not reduce each
subgroup's register demand. More workgroups request overlapping K/V tiles;
actual occupancy and DRAM traffic were not measured.

The prefetch algorithm derives packets from subgroup count: the Q128 candidate
changes K packet height 1 to 2 and V height 8 to 16. Offline ISA confirms the
descriptor changes. The restored-packet implementation covers the complete tile
through two original-sized packets per subgroup; simply claiming 32 prefetch
subgroups while launching 16 would miss coordinates.

Matched control/candidate DSOs have identical reduced build coverage. Exact
traces are joined to embedded ELF resource metadata from the same mapped DSO.
Original-full and narrowed-control executed ISA is identical through end of
thread for both selected kernels; complete code hashes differ only because of
trailing discarded immediates. Two fresh calibration starts differ by only
-0.55% to +0.72%, excluding build coverage as an explanation for the large loss.

## Correctness and measurement limits

All 36 initial gates pass: original, rebuilt control and Q128 each reproduce all
12 captured outputs bit for bit. Twelve additional packet-control gates also
pass bit for bit. K64 passes all 12 fixed-tolerance reference gates and exact
same-input repeats, but is not bit-identical across policies because its online
softmax blocking changes. No numerical tolerance was relaxed: independent
blocked FP32 attention uses atol 0.02 and rtol 0.01 from the frozen ordinary
attention test. Gates also cover NaN-filled output, surrounding canaries,
changed operands, dirty allocation conditioning and input noninterference.

GPU jobs were serial, with the competing serving service inactive. The initial
screen uses two fresh process starts, reversed case/arm order, 30 symmetric
ABBA/BAAB blocks and three cache conditions. Complete-call latency includes the
native wrapper, internal workspace allocation, launch and completion wait;
service-shaped output is preallocated. IPC, preparation and conditioning are
outside timing. Device traces and reference calculations are separate.

The initial source snapshot uses at least one second of balanced total warmup,
then three 100ms windows with medians spanning at most 1%, capped at ten seconds.
The committed runner strengthens this to at least one second per arm and records
per-arm duration. Seven of 24 initial case/start combinations fail stabilization
(21 of 72 condition rows). All eight long packet-control condition rows fail
that strict warmup rule; their ranges are descriptive, not qualified wins.
One of four K64 case/start combinations fails (two of eight condition rows).
Even the initial 27 stable long-context rows regress 38.8–62.2%.

The reviewer found two reporting issues and an important conditioning limit:

- Stability/CV previously included unused anchors. It now considers only actual
  contributors; nine 16K rows correctly become stable. All numeric predictions
  are unchanged. Original and reviewed derived files are both retained.
- The runner now reports ratio of aggregate costs and bootstraps paired blocks,
  avoiding inflation from averaging noisy per-block ratios. Original raw samples
  and summaries are retained; full-prefill costs already used aggregate latency.
- Short interleaved calls depend on block position and on the arm-specific decoy.
  Their apparent gains are not reliable. Follow-ups use the identical streaming
  conditioner across arms. Long-context regression also appears inside the
  native kernel in separate diagnostic traces, not only in host timing.

## Adjacent solutions and the resource-gate amendment

K64 retains Q256/SG32 and halves key-loop trips, barriers, Q reloads and output
rescales. Its larger fragments require a new correctness/resource check.
Page size 832 is divisible by 64; K128 would need page-routing changes and was
not attempted.

K64 fails the initial no-spill-anywhere screen: compiler metadata reports
64-byte dense and 256-byte paged scratch/spills. Offline ISA then established
that the actual service paths have **no inner-key-loop scratch instructions**.
Dense setup/cleanup performs one 64-byte store and load; normal paged setup/
cleanup totals 128 bytes stored and loaded. Additional loop spills occur in an
unused overflow-surface branch. Captured effective paged surface is 154752 rows,
below its 16777216-row threshold.

That evidence prompted a recorded amendment before K64 GPU execution: preserve
the initial failure and measure whether fewer iterations outweigh bounded setup
scratch. It was never labeled a no-spill kernel. All numerical gates stayed
fixed. The two long anchors then failed to show a gain, ending this candidate.

Two smaller adjacent ideas were subsequently tested independently; see the
[barrier and pure-prefill results](d256-barriers-dispatch-experiment.md):

1. For confirmed pure B1 prefill, use the existing non-mixed path to avoid
   metadata kernels, decode scratch and a decode launch that skips its only
   sequence. Traces show only 4–14 microseconds of removable device work per call;
   host overhead may add benefit for short calls. Never infer pure prefill merely
   from maximum query length in a mixed batch. Moving an unused dense dummy
   allocation into the paged branch remains a separate, untested wrapper cleanup.
2. Test removal of only the two mainloop split-barrier calls for ordinary D256
   with no cross-subgroup reduction. Mainloop storage is empty and arithmetic is
   subgroup-local, but the barriers may preserve useful cooperative prefetch
   timing/cache reuse. Preserve initial workgroup reductions and prove correctness
   and performance before removing them from serving code.

A Q128/SG32, DPAS4, proven-GRF128 policy would actually reduce per-subgroup
accumulators, but adds workgroups/subgroups and may lose DPAS efficiency. It is a
separate, lower-confidence hypothesis. This experiment does not prove that every
smaller-query kernel or every nearby optimization is slower.

## Reproduction and evidence

Reusable capture, replay, resource/dispatch inspection and weighting commands
are in `docs/kvarn-profiling.md`. The original fixture plan and three frozen-source
patches are committed. Experiment scripts, exact plan snapshots, builds/derivations,
four experimental DSOs, raw captures/token IDs, references, samples, traces and
three adversarial reviews are retained locally on Brutus under
`benchmark-results/issue11-d256-q128-20260909/`. The evidence archive is local,
not a hosted attachment. `SHA256SUMS` covers retained files; runtime caches and
Python bytecode are excluded from the archive.

The subsequent barrier/dispatch evidence uses a separate directory and archive;
it does not modify this original retained experiment. Its plan and fourth
frozen-source reproduction patch are also committed.

No serving source change is proposed, so there is no kernel-fork PR to merge.
The negative prototype patches are retained for reproduction. Existing model,
replay/state, MTP2, image and paired service qualification were not triggered;
operator failures already meet the experiment's stop rule. No service speedup,
deployment or release is claimed. AI assistance included Codex and three
explicitly requested skeptical reviewer agents.
