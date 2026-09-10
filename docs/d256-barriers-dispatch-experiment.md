# D256 follow-up: loop barriers and pure-prefill dispatch

Removing the loop barriers is a regression; retain them.
Selecting pure-prefill dispatch removes redundant work and yields a small
repeatable 16K operator benefit, but does not establish a useful long-context
or service-level speedup. This operator experiment stopped before service
qualification; the barrier source remains a reproduction patch. The user
subsequently advanced the small dispatch cleanup to a separate
[serving integration stage](pure-prefill-serving.md).

These are two independent follow-ups to the [issue11 tile experiment](d256-attention-experiment.md).
They use its original Q256/K32/SG32 policy, frozen kernels `23d8103`, vLLM
`9bfcf123`, B70 runtime and six captured extents per cache format. The
[prospective plan](../fixtures/d256-adjacent-followup-plan.md) fixes the scope
and gates before GPU testing. They do not combine either intervention with
the rejected Q128 or K64 policies.

## Results

These ranges include both fresh starts and all three cache conditions. Positive
values mean higher predicted attention-call cost; they are not TTFT changes.

| Intervention and cache | 16383-token request | 65023-token request |
| --- | ---: | ---: |
| Remove loop barriers, paged BF16 | -0.42% to +6.57% | +5.99% to +14.65%* |
| Remove loop barriers, dense KVarN FP16 | +0.08% to +11.46% | +5.41% to +15.17%* |
| Pure-prefill dispatch, paged BF16 | -0.73% to -0.49% | -0.55% to +0.05% |

\* All 65K barrier predictions include at least one anchor whose warmup did
not stabilize. Treat their magnitude as descriptive, not qualified performance.
Four of 24 barrier case/start warmups fail: paged index4 in both starts, dense
index3 in start0, and dense index4 in start1. All flags remain in the reports.
Short two-process cold/interleaved samples also have high variation (up to
12.9% control CV), making tiny changes there inconclusive.

The rejection does not depend on those noisy rows. All eight stable long-shape
warm measurements regress by 1.33–7.34%, with paired 95% intervals excluding
a speedup. Preserving cooperative prefetch coordination is a plausible benefit
of the barriers, but no cache-traffic or stall counter measurement establishes
the cause. Empty shared arithmetic storage alone did not imply a performance win.

All twelve pure-dispatch case/start warmups stabilize. Its shortest synchronized
call improves by 1.65–2.67%; the full 16K prediction saves only 3.7–5.5 ms.
All six weighted 16K bootstrap intervals exclude no gain. At 65K all three
second-start intervals include no gain; the first-start interleaved interval
also includes zero. These intervals retain pairing within each anchor but do
not include interpolation uncertainty or covariance across different anchors.
At 65K the warm prediction changes sign across starts (-0.34%, then +0.05%).
Queued windows reduce the apparent benefit: shortest-case latency improves
0.94%/1.12%; longest-case latency improves 0.11%/0.026%, with the latter interval
including no gain. Submission savings must not be counted again as GPU savings
or projected directly onto service latency.

## What was actually changed

The barrier candidate removes only the per-key-loop workgroup split-barrier
arrive/wait pair. Mainloop storage is empty, arithmetic reductions are within
subgroups, and the selected ReduceK=1 epilogue uses register fragments. Initial
workgroup reductions remain synchronized. Prefetches remain cache hints with
no shared intermediate result. Source review found no producer/consumer
dependency requiring these loop barriers in the selected path; finite tests
do not establish safety for every legal shape or compiler/driver.

Independent review caught and fixed a scope error before the timed build:
D512 shares the D256 tile geometry, and the named D256 policy also serves
smaller logical heads. The final candidate checks both policy identity and
`args.head_size == 256` on the host, selecting a distinct native specialization.
It leaves no runtime branch inside the key loop. D224, D512, page16 policies,
FP8/mixed precision, local/noncausal, sink and LSE variants retain the default.
Only the reduced capture-relevant configuration set was compiled; this is
not a full-configuration build qualification.

The selected paged BF16 and dense FP16 kernels retain GRF256, SLM4096 and no
emitted spills. The compiler reports one barrier resource instead of two.
Static ISA signal/wait sites each decrease from ten to eight: one pair is
removed from each of the two surface paths. Prefetch sites remain seven and
DPAS sites remain 88. The preserved fallback variants have identical ISA to
the original control through end-of-thread. Actual candidate traces are
matched to the generated code and resource metadata from their mapped DSO.

The dispatch experiment uses the unchanged original full DSO and only switches
`is_mix_batch` from true to false. Saved CPU metadata must establish packed
paged B1 prefill: cumulative query lengths `[0,M]`, `M=max_query_len>16`, one
block-table row and one KV length `K>=M`. Dense input, decode, speculative short
queries, B2 and inconsistent/padded metadata are excluded. No dummy allocation
cleanup is bundled into the measurement.

Every paired trace selects the exact same prefill template. The candidate
removes exactly the subtraction, comparison and skipped decode kernels:
four device kernel events become one. Those removed events total about
4.5–11.6 microseconds per call in these profiler-on traces; the durations are
diagnostic, not service performance evidence. The trace validator checks exact
kernel fingerprints, CPU correlation and unchanged unrelated device work.

## Correctness and measurement

The final barrier candidate passes all twelve captured cases bit for bit
against original outputs. Gates use unchanged independent FP32 references
(atol 0.02, rtol 0.01), dirty allocator/operand conditioning, exact repeats,
output-pointer ownership, canaries and Q/K/V/metadata noninterference. The
initial candidate also passed, illustrating why numeric testing alone did
not reveal its broader dispatch-scope error.

All six pure-dispatch cases pass those gates. Separately, both flags agree
exactly with LSE enabled: finite FP32 head-major `[24,M]` LSE, exact paired
outputs/LSE and dirty-state repeats, intact output canaries and unchanged
inputs. LSE-on output also satisfies the retained FP32 output reference.
No equality between the distinct LSE-on/off templates is required.

Pure-dispatch timing uses one process, one immutable DSO, and identical tensor
addresses for both arms. Each case has 30 balanced, randomized AB/BA pairs
under warm, 512 MiB streaming-conditioned and interleaved conditions. Both
arms receive the same fixed default-path conditioning. Warmup uses the same
warm condition and retains failed stability flags. Two fresh processes reverse
case order. Completed-call and host submission time are recorded separately;
shortest/longest cases also use eight-call queued windows.

Barrier timing uses the established two-process immutable-DSO runner with
30 balanced ABBA/BAAB blocks, the same three conditions and two fresh starts.
Numerical tests and profiling precede timing; compilation and other GPU work
do not overlap timed samples. Control CV, drift, raw samples and every warmup
result are retained. Complete-call timing includes wrapper/workspace overhead
and completion wait, excluding conditioning and IPC.

Whole-prefill predictions count all 128/512 observed attention calls for the
16383/65023-token requests. They retain the original interpolation between
full-chunk anchors and exact ragged-tail measurements, without extrapolation.
These predictions are attention-call costs, not TTFT or model-service timings.

## Reproduction and retained evidence

Evidence is local on Brutus under
`benchmark-results/d256-barriers-dispatch-20260909/`. It is separate from the
immutable issue11 archive. The new evidence retains the initial scope-bug
patch/build/audit, corrected candidate, resource/ISA and native dispatch
attestations, strengthened gates, paired samples, source snapshots and reviews.
Captured tensors and FP32 references are reused with checksums from
`benchmark-results/issue11-d256-q128-20260909/`; reproduction needs that original
evidence as well.

The new tools are `scripts/xpu_attention_pure_prefill.py` for the single-process
flag experiment, `scripts/xpu_attention_prefill_trace.py` for dispatch attestation,
and `scripts/xpu_attention_regate.py` for testing a fresh DSO against verified
retained original references. `xpu_attention_dispatch.py` accepts explicit
candidate tile/subgroup values for policies that preserve Q256/SG32.
`xpu_attention_pure_prefill_bootstrap.py` adds weighted uncertainty estimates
that preserve pairs and independently resample anchors with fixed interpolation.
Those intervals do not model cross-anchor covariance, interpolation error or
service overlap; starts and cache conditions remain separate estimates.

For the flag experiment, run the frozen runtime Python with `--library` pointing
to the original full DSO, `--capture` to `capture-auto-layout/manifest.json`,
`--references` to the old `gates/` directory and a fresh `--output`. Then use
`--gates-from <new-gates>/report.json` and separate fresh outputs for `--start 0`
and `--start 1`. Preserve the experiment's ICD environment. See the retained
run commands and [profiling workflow](kvarn-profiling.md) for the DSO build,
candidate gating, dispatch attestation, timing and weighting sequence.

A production pure-prefill change needs trusted host metadata and explicit
forwarding through vLLM's XPU wrapper: the frozen version's `**kwargs` discards
`is_mix_batch`. `num_decode_reqs == 0` is not a valid general proof because that
field can retain a default zero outside DCP handling. Merely passing the flag
at the attention backend would not apply this experiment to serving.

No native fork PR was proposed at this stage. The workflow PR retained the
experiment and small dispatch result. Model/replay/state, MTP2, image and
profiler-off service qualification were not run in this operator experiment;
see the subsequent [serving stage](pure-prefill-serving.md) for that work.
Nothing was deployed or released. AI assistance
included Codex and three explicitly requested skeptical reviewer agents.
