# Native XPU Kvarn gate sequence

This is the short operational checklist for moving the Brutus Qwen3.8 service
from the accepted Triton Kvarn reader to the native Xe2 decoder. The existing
non-native Kvarn service remains the rollback until every correctness gate
passes. Performance has two deliberately separate comparisons at the common
65,536-token limit. Compact Kvarn non-native versus compact Kvarn native
isolates the reader kernel. BF16 `--kv-cache-dtype auto` versus compact native
Kvarn is the end-system comparison the deployed feature ultimately has to
match; it changes both the cache representation and reader and must not be
used to attribute a speedup to the native kernel alone.

The candidate keeps the accepted service shape fixed: the pinned
`jasonboukheir/Qwen3.8-27B-AEON-Ultimate-Uncensored-BF16-W4A16-AutoRound`
revision, compressed-tensors weights, eager execution, text only, no MTP, no
prefix cache, no XPU graph, and at most four sequences. Only the KV dtype and
native-reader switch may differ in an A/B.

## Gate order

1. Build the narrowed BMG kernel package. Run the native primitive suite,
   including the GDN allocator-pressure regression and the 262K ragged B4
   decoder oracle at split counts 1 and 16.
2. Start native Kvarn at 65,536/B1 with split count 1. Require the native Xe2
   dispatch log and reject any fallback, exception, non-finite value, or device
   assertion. Replay the short and boundary fixtures twice and require exact
   token IDs.
3. Restart the same B1 service and replay the fixtures. Exercise cancellation,
   immediate slot reuse, and another exact replay.
4. Repeat at 65,536/B4 with distinct concurrent prompts. Require each request
   to match its own B1 replay; this catches cross-request state leakage.
5. Start non-native Kvarn at 262,144/B1. Use a 261,631-token prompt plus 512
   output tokens, ending at 262,143, and freeze its exact token IDs. Start
   native Kvarn with the same closure/input and require its first and restarted
   runs to match that reference exactly. Exact native replay alone is not
   sufficient because it cannot detect a deterministic decoder error.
6. Start 262,144/B4 and mix shorter requests around the long incumbent. Cancel
   and replace one request, then require the unaffected streams to retain their
   B1 token sequences. Four simultaneous maximum-length requests are not a
   requirement; the scheduler limit does not reserve one quarter of the cache
   for each slot.
7. Only after correctness, run the kernel-isolating compact
   non-native/native trials, followed by BF16-auto/native end-system trials at
   65,536. Establish split 1 first. Other split counts are eligible only after
   their long primitive and all service gates pass.

The Brutus flake exposes isolated foreground launchers for these phases:

```text
vllm-xpu-brutus-auto-b1
vllm-xpu-brutus-auto-b4
vllm-xpu-brutus-kvarn-b1
vllm-xpu-brutus-kvarn-b4
vllm-xpu-brutus-kvarn-native-b1
vllm-xpu-brutus-kvarn-native-b4
vllm-xpu-brutus-kvarn-262k-b1
vllm-xpu-brutus-kvarn-262k-b4
vllm-xpu-brutus-kvarn-native-262k-b1
vllm-xpu-brutus-kvarn-native-262k-b4
```

The native launchers explicitly set the decode, materializer, persistent
scratch, natural-layout, and split-count switches. The auto launchers
explicitly disable them. That makes an inherited shell variable unable to
silently change one arm.

## Matched performance trials

Run at least four recorded repeats per arm in repeated balanced order
`reference, native, native, reference`, restarting the service between arms.
Four repeats per arm therefore use two complete ABBA cycles. Use the same
candidate closure and warmed compilation cache. Preserve the detailed
benchmark JSON, engine log, command line, source revisions, and store path for
every run.

Use two primary workloads:

- latency/B1: fixed 16,384-token input, 512-token output, concurrency 1;
- aggregate/B4: fixed 8,192-token input, 512-token output, concurrency 4.

Use `--random-range-ratio 0`, `--ignore-eos`, a fixed seed, detailed result
output, and identical prompt counts. A separate mixed-arrival trace should
hold three decoding incumbents while injecting a delayed long prefill; record
incumbent ITL before, during, and after the prefill. This directly covers the
observed temporary 7 -> 0.1 -> 7 tok/s behavior instead of misclassifying it
as steady decoder speed.

Compare one matched workload at a time:

```bash
scripts/kvarn_perf_gate.py \
  --reference results/auto-1.json \
  --reference results/auto-2.json \
  --reference results/auto-3.json \
  --reference results/auto-4.json \
  --candidate results/native-1.json \
  --candidate results/native-2.json \
  --candidate results/native-3.json \
  --candidate results/native-4.json \
  --reference-log results/reference-1.log \
  --reference-log results/reference-2.log \
  --reference-log results/reference-3.log \
  --reference-log results/reference-4.log \
  --candidate-log results/native-1.log \
  --candidate-log results/native-2.log \
  --candidate-log results/native-3.log \
  --candidate-log results/native-4.log \
  --correctness results/native-correctness.json \
  --comparison-kind kernel \
  --output results/native-vs-nonnative.json \
  --mode match
```

Use `--comparison-kind kernel` with a compact non-native reference, and
`--comparison-kind end-to-end` with a BF16-auto reference. `match` requires at
least 95% of the reference's output, total-token, request, median-request, and
p10-request decode throughput, with p99 TTFT and p99 ITL no worse than 110%.
`--mode win` requires no regression on any of those axes plus at least a 5%
gain in output or median per-request decode throughput. It therefore cannot
declare a win by trading latency for throughput, or throughput for latency.

Each detailed benchmark JSON must carry the gate's `kvarn_*` provenance
metadata: candidate/store identity, model revision, service and workload IDs,
seed, model length, sequence limit, eager/prefix/MTP/graph settings, cache
dtype, native switch, split count, arm, scheduler peak-running evidence,
unique run UUID, timezone-qualified start time, global ABBA run order, the
engine-log SHA-256, and the SHA-256 of the passed correctness artifact.
Built-in model, tokenizer, backend, request-rate, prompt-count, duration,
token totals, and concurrency fields must also match. The gate requires four
unique results and logs per arm, full completion, observed B1 or B4
concurrency, internally consistent throughput arithmetic, clean logs, and
positive native-dispatch evidence in every candidate log.

The correctness artifact is not a boolean checklist. Every required gate is
an object containing `status: passed`, a durable evidence path, and that
file's SHA-256. The performance gate re-reads and hashes all evidence,
including the current-build GDN allocator-pressure pass, its no-fix mutation
failure, short and 262K native decoder primitives, B1/restart/cancel/B4
service results, the non-native/native near-262K equivalence, and the restarted
near-262K result.

Do not promote based on a single aggregate tok/s number. Retain p99 TTFT, p99
ITL, median per-request decode tok/s, request throughput, output throughput,
and total-token throughput. A failed request, workload-shape mismatch, missing
detailed timing, or unequal repeat count invalidates the comparison.

If a native gate fails, stop the foreground process, start
`vllm-xpu-brutus-kvarn-b1` or `vllm-xpu-brutus-kvarn-b4` from the last accepted
closure, wait for `/health`, replay one frozen B1 fixture, and scan the new
engine log before restoring the systemd service. Do not replace the deployed
non-native profile until that recovery sequence and all native gates pass.
