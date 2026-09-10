# D256 barrier and pure-prefill follow-up plan

Declared before GPU testing on 2026-09-09. Evidence is separate from the
immutable issue11 archive: `benchmark-results/d256-barriers-dispatch-20260909/`.

Test two interventions independently against frozen kernels `23d8103` and
vLLM `9bfcf123`, original Q256/K32/SG32, pinned B70 profiling runtime:

1. Remove only the per-key-loop workgroup split-barrier arrive/wait for ordinary
   Xe2 D256 causal, non-local FP16/BF16 prefill, ReduceK=1. Keep initial reductions,
   prefetches, arithmetic, policy, LSE and sink variants intact. Inspect generated
   code/resources before GPU use. Scratch is classified by location, not rejected
   solely because metadata reports a nonzero number.
2. Select existing `is_mix_batch=False` for confirmed paged B1 prefill, using the
   original full DSO. Preserve allocation and every other argument. Eligibility
   requires saved CPU cumulative query lengths `[0,M]`, one block-table row,
   one KV length `K>=M`, and packed Q with `M=max_query_len>16`. Decode, speculative
   queries, B2 and inconsistent/padded metadata stay outside this experiment.

Use all six original captured extents for each affected cache format, preserving
physical pages, strides, interleaved K/V aliasing and output addresses. References
come from the original checksummed FP32 reference files. Numerical gates remain
atol=0.02/rtol=0.01, finite output, dirty allocator/exact same-input repeats,
canaries, unchanged inputs and explicit output-pointer ownership. Pure dispatch
also requires exact agreement between flags with LSE off and on, finite FP32
head-major LSE and exact LSE repeats. Separate templates may differ between LSE
off/on. Trace actual native dispatch and skipped work outside timed samples.

No timed GPU work while CPU compilation or another GPU workload runs. Measure
completed-call wall time with synchronization before/after, excluding conditioning
and IPC. Pure dispatch uses one process, immutable DSO and the same operands for
both arms; balanced randomized AB/BA pairs, identical default-path warm conditioning
or identical 512 MiB streaming conditioning or a fixed default-path decoy. Record
submission time separately. Run queued-window checks on shortest/longest anchors.
Barrier timing uses the existing separate-process DSO runner with balanced ABBA
blocks. Both use 30 paired blocks and two fresh starts with reversed case order.
Warmup is at least one second, then three >=100 ms windows within 1%, capped at
10 seconds; retain all failures and control CV/drift instead of hiding them.

Report ratios of aggregate times, paired bootstrap intervals and raw samples.
Weight all observed attention calls using the established six-anchor interpolation
without extrapolation. Distinguish attention-call prediction, submission savings,
queued throughput and service performance. Advance only a correct, repeatable
improvement with meaningful weighted benefit to integration/service qualification.
If either result is null/negative, preserve the patch and evidence in workflow PR16.
Do not combine changes to rescue an independently failed experiment.
