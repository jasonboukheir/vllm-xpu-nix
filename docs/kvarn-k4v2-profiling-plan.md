# K4V2 profiling prerequisite for the coordinated release

The user requested a replacement release goal that prioritizes getting profiling
working and understanding performance before more tuning. This is a required
milestone of [the release plan](kvarn-k4v2-plan.md). Completing profiling does
not complete the goal: qualification, clean commits, paired packaging, and
verified publication in all three repositories still follow.

**User update, 2026-09-14 UTC:** Performance comparisons and optimization use
MTP disabled. MTP must still work and pass applicable correctness, quality, and
lifecycle checks. Retain all prior MTP performance failures, but do not use them
as release speed gates or tune acceptance to achieve parity. CPU sampling is a
supporting diagnostic when useful; matched MTP-off XPU critical-path and kernel
resource evidence are the profiling requirement. Existing tool repairs and
validated ISA/counter evidence remain reusable within their actual coverage.

## Resume from evidence

The private run directory is
`benchmark-results/kvarn-k4v2/20260913T051252Z/`, relative to the packaging
repository on Brutus. Read its `CURRENT.md` and `status.json` first, followed by
the named stage plans and reports. Check the actual goal status and live process
identities before acting. Never infer that a job is live from an archived session
number. Preserve raw captures, failures, frozen inputs, and measured source trees.

At this milestone's creation, the full four-column reducer MTP2 comparison has
finished. Its short B1 decode ratio is 0.9368519285150829 versus matched K4V4,
failing the 0.95 outer floor. The complete result and exact same-format output
review are retained as historical MTP evidence. Under the updated scope that
MTP2 speed failure is not a release performance gate. The candidate still needs
MTP-off performance and remaining release qualification. Use CURRENT.md for
actual completed profiling, live processes, and next actions. Do not resume the
old stopped coordinator or run an old continuation with superseded MTP2 gates.

## 1. Unblock and qualify the tools

Read [the existing profiling workflow](kvarn-profiling.md), especially trace
analysis, codegen/resource inspection, and hardware counters. Reuse these tools:

| Evidence | Existing starting points |
| --- | --- |
| Model trace and callers | `scripts/kvarn_xpu_trace.py`, `scripts/kvarn_xpu_trace_stacks.py`; run-local `profile-mtp-reducer16-queue.py` and its frozen plan/reviewer |
| Compiler resources and native code | `scripts/xpu_attention_codegen.py`, `scripts/kvarn_resource_audit.py`; run-local `inspect-k4v2-compiler-resources.py` and `reducer16-compiler-resource-comparison.json` |
| Hardware metrics | `nix/xpu-metrics-probe.nix`, `nix/unitrace.nix`, `scripts/kvarn_xpu_metrics_probe.py`, `scripts/kvarn_xpu_counter_run.py`, `scripts/kvarn_xpu_counter_workload.py`, `scripts/kvarn_xpu_counter_report.py` |
| Real service performance | Existing frozen ABBA service harness, stream audits, MTP counters, and retained comparison plans |

Record a tool-readiness report with exact executable/library versions and hashes,
driver/runtime pairing, commands, output locations, successful validation, and
remaining limitations. The pinned oneAPI compiler, IGC `iga64`, and PTI libraries
are installed; installation alone does not validate collection. VTune is not a
prerequisite. Do not change the upstream bases or unrelated dependency pins to
obtain a profiler. Use a separate diagnostic environment if necessary, documenting
its differences and proving that it loads the intended candidate libraries.

For traces, require real XPU kernel events, host/runtime correlation, expected
phase coverage, and audited model outputs. For ComputeBasic metrics, require
actual finite samples with definitions, units, device identity, kernel coverage,
and matched off/on/off controls. Enumeration and an opened metric stream are
insufficient. Validate a retained known workload before adapting the collector
to the implicated attention workload. Inspect available bandwidth, cache,
occupancy, execution/stall and shared-function metrics; do not assume a name or
metric exists on this device. Instruction-level stall-PC sampling remains
unqualified until demonstrated.

If a tool fails, retain the error, determine whether the cause is permissions,
runtime compatibility, configuration, unsupported hardware, or sampling coverage,
and repair and retest the supported path. Do not skip this work merely because it
is inconvenient. Do not weaken host permissions or reset the GPU unattended.
Where a metric is demonstrably unsupported, document the attempted paths and
use validated alternative measurements that answer the same optimization
question. An unresolved measurement essential to the chosen hypothesis keeps
that hypothesis unready for implementation.

## 2. Explain request time and extra work

Start with matched MTP-off benchmarks of the current candidate, using both
formats, identical prompts, frozen settings, and the existing harness. Capture
representative XPU traces with warmup/before/profiled/after controls and audit
within-format token trajectories. Confirm MTP is disabled in actual arguments,
configuration and request metrics. Preserve the distinction between full-request
work and captured worker rounds. Inspect model forward, sampling,
cache writes/flush/reconstruction, attention producer/reducer, weight GEMMs,
recurrent layers, CPU scheduling, synchronization, launches, and idle gaps.

Attribute kernels using external IDs, runtime correlation, and CPU ancestry.
Keep unresolved events visible. Build a timeline and quantify the critical path,
host/device overlap, round count, per-round cost, and completed-request cost.
Overlapping device-duration sums are not wall time; do not add synchronization
waits to the work they wait for. Profiled CPU durations and request times include
instrumentation overhead and cannot substitute for unprofiled serving results.

Compare equal generated-token work with MTP off, then distinguish per-step
device costs from launch, synchronization, and other host costs. Prior MTP
captures have trajectory-dependent extra work and are not the basis for this
comparison. Keep their work-count findings as historical evidence. Use additional
phase captures or matched-work native replay only to resolve an MTP-off question.

After the failing anchor, use representative longer and concurrent workloads
where needed to test whether the selected MTP-off bottleneck generalizes. Reuse completed
valid evidence. Do not repeat the whole performance matrix merely to obtain a
passing sample.

## 3. Connect hot kernels to registers, memory, and instructions

Join exact traced kernel names to the actual mapped DSO and embedded device ELF.
Record SIMD width, allocated GRFs, declared spill/scratch, static SLM, separately
determined dynamic launch scratch/SLM, workgroup shape, split count, and code size.
Disassemble selected hot kernels using the pinned IGC tool after verifying its
Xe2 platform setting. Preserve raw code, metadata, hashes, and relocation context.
Use compiler diagnostics for register liveness where supported; otherwise state
that allocated GRFs do not measure peak live registers or occupancy.

The completed resource audit reports 256 GRFs for the inspected producers. The
rejected query-hoist K4V2 binary declares a 64-byte spill and scratch buffer.
This supports a register-pressure hypothesis, but does not establish the fraction
of its slowdown caused by spilling. Missing spill metadata in other binaries is
not a measurement of zero spill traffic. Baseline and current producer code have
equal lengths but different hashes despite unchanged producer source: resolve
instruction/relocation differences before claiming compiled equivalence.

Use validated counters and controlled replay to distinguish bandwidth/cache
limits, dependency stalls, occupancy constraints, unpacking/instruction cost,
barriers, launch overhead, and insufficient parallel work. Record capture coverage
and uncertainty; short kernels may require repeated matched replay for sampling.
Counter collection perturbs execution and is separate from performance timing.
Static resource counts and stall counters alone do not prove the bottleneck.

## 4. Required diagnosis before another performance change

Write `profiling-diagnosis.md` and a machine-readable evidence index in the run
directory. Link raw artifacts rather than relying on conversational memory.
The report must include:

1. A tool-readiness table, validation results, and explicit unavailable metrics.
2. A matched K4V4/K4V2 critical-path and work-count comparison, with profiler
   overhead, attribution coverage, variation, and unresolved gaps stated.
3. Hot-kernel resource/ISA findings joined to trace identities, supported by
   available validated memory/occupancy/stall measurements.
4. A ranked set of hypotheses with supporting and contrary evidence, a plausible
   bound on end-to-end benefit, confidence, and the next falsifying experiment.
5. A concrete selected experiment, fixed semantics/configuration, exact source
   and runtime baseline, affected correctness checks, and unprofiled comparison.

Spend the time needed to produce a defensible diagnosis. If the evidence remains
inconclusive, the next experiment is a diagnostic measurement. Do not declare
this milestone complete merely because files exist or a profiler command ran.
Record its completion and the selected experiment in `CURRENT.md` and
`status.json` before implementing another performance change.

Potential ideas include resident FP16 block loads, V unpacking, reducer scheduling,
register lifetimes, and host synchronization. These are hypotheses to rank from
the evidence. The existing tail-block-copy note is not an approved implementation
plan. Preserve the validated quantization/history semantics; rejected higher
retention floors are not candidates for another trial under this plan.

## 5. Optimize, qualify, and publish

Change one supported cause at a time in a separate candidate. Inspect the generated
code/resources, pass affected correctness checks, then measure with the profiler
off using balanced order and fresh starts. A native microbenchmark win must
translate to the real-model workload; retain unsuccessful attempts and avoid
retry-until-pass selection. Update the diagnosis as evidence changes.

Continue the full [release workflow](kvarn-k4v2-plan.md). Target parity and the
0–3% tuning range; investigate repeatable larger regressions. Keep the formal
0.95 throughput/decode and 1.10 latency/TTFT limits and all quality thresholds.
Do not silently waive an unresolved floor failure. Complete affected model
quality, MTP-off performance, MTP functionality, released MTP-off controls,
API/tools/images/concurrency/cancellation,
fixed-allocation memory, occupied capacity, and long-context checks. Startup
allocation is not proof of live capacity; quantify the actual mixed K4V2-target /
K4V4-draft configuration when reporting memory savings.

Clean up both feature commit stacks on the retained bases, qualify the exact final
paired Nix package, publish the coordinated source and packaging release, and
verify all three repositories' immutable release refs and resolved source pair.
Host deployment remains outside the goal. Keep concise current checkpoints with
owned process identities, completed evidence, next action, and remaining gates
so subsequent agents can continue without reconstructing the conversation.
