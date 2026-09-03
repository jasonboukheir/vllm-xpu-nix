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
prefix cache, no XPU graph, a 2,048-token scheduler budget, and at most four
sequences. The runner passes `--max-num-batched-tokens 2048` to every launcher,
verifies the live service argv, and seals the value into each result. Only the
KV dtype, native-reader identity switches, and the native-only decoder split
count may differ in an A/B. Use the runner's `--max-num-batched-tokens` option
only when deliberately creating a new matched workload; its value still applies
identically to both arms.

## Gate order

1. Build the narrowed BMG kernel package. Run the native primitive suite,
   including packed K-column-scale coverage across every token subgroup and
   the 262K ragged B4 decoder oracle at split counts 1 and 16.
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

Run at least eight recorded repeats per arm in repeated balanced order
`reference, native, native, reference`, restarting the service between arms.
Eight repeats per arm provide four complete ABBA blocks, the minimum used for
the one-sided paired confidence result. Four repeats per arm remain useful for
an exploratory floor check, but the harness reports `insufficient_evidence`
rather than claiming statistical parity from only two blocks. Use the same
candidate closure and warmed compilation cache. Preserve the detailed
benchmark JSON, engine log, redacted command line, source revisions, actual
process package, and order-independent Nix closure digest for every run.

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
`--comparison-kind end-to-end` with a BF16-auto reference. `match` is the hard
floor: it requires at least 95% of the reference's output, total-token,
request, median-request, and p10-request decode throughput, with p99 TTFT and
p99 ITL no worse than 110%. The automated runner replaces the legacy
median-of-per-run p99 comparison with p99 computed from all detailed TTFT or
ITL samples pooled within each arm. A separate target result uses each
four-run ABBA block as one paired observation. It passes only when the
one-sided 95% lower confidence bound of the candidate/reference geometric
mean is at least 0.98 for both output throughput and median per-request decode
throughput. This is a non-inferiority test with no upper bound: a candidate
that is significantly faster than auto passes rather than being rejected as
"not equivalent."

The gate JSON records `hard_floor.status` and `statistical_parity.status`
separately. A floor pass does not establish the 1:1 performance goal, and an
insufficient parity sample is not silently treated as a pass.

`--mode win` requires no regression on any of those axes plus at least a 5%
gain in output or median per-request decode throughput. It therefore cannot
declare a win by trading latency for throughput, or throughput for latency.

Each detailed benchmark JSON must carry the gate's `kvarn_*` provenance
metadata: candidate/store identity, model revision, service and workload IDs,
seed, model length, sequence limit, eager/prefix/MTP/graph settings, cache
dtype, native switch, split count, arm, scheduler peak-running evidence,
unique run UUID, timezone-qualified start time, global ABBA run order, the
engine-log SHA-256, and the SHA-256 of the passed correctness artifact.
It also seals the executable and package observed in `/proc`, the actual
process-package closure digest, the candidate closure digest, and the
canonical matched-profile digest. Closure paths are sorted and deduplicated
before hashing, so Nix query order cannot alter the identity.

The canonical profile comes from the actual service argv and effective
allowlisted environment. The only normalized arm differences are
`--kv-cache-dtype`, the five native identity switches (`KVARN_NATIVE_XPU`,
decode, DPAS layout, materialize, and persistent scratch), and exactly
`KVARN_NATIVE_XPU_SPLITS`. Any other argument or effective
performance-environment difference invalidates the cell. Secret-looking
argument values are redacted before argv or profile evidence is written.

Both arms must keep the beta's validated natural layout. The auto reference
uses the neutral split value `1`; the native candidate uses the empirically
selected value `24` for B1 and `16` for B4. The split
variable is unreachable when the native reader is disabled, so copying the
candidate's tuning value into auto would add misleading provenance without
making the execution more closely matched. The gate instead permits only this
one named native-only difference and verifies both values exactly.

Built-in model, tokenizer, backend, request-rate, prompt-count, duration,
token totals, and concurrency fields must also match. The gate requires four
unique results and logs per arm, full completion, observed B1 or B4
concurrency, internally consistent throughput arithmetic, clean logs, and
positive native-dispatch evidence in every candidate log.

The correctness artifact is not a boolean checklist. Every required gate is
an object containing `status: passed`, a durable evidence path, and that
file's SHA-256. The performance gate re-reads and hashes all evidence,
including short and 262K native decoder primitives, B1/restart/cancel/B4
service results, the non-native/native near-262K equivalence, and the
restarted near-262K result. GDN allocator-pressure and no-fix mutation
artifacts are diagnostics, not release evidence: the retained-scratch
hypothesis did not explain the observed failure.

Do not promote based on a single aggregate tok/s number. Retain pooled p99
TTFT, pooled p99 ITL, median per-request decode tok/s, request throughput,
output throughput, and total-token throughput. A failed request,
workload-shape mismatch, missing detailed timing, or unequal repeat count
invalidates the comparison.

### Automated auto/native matrix

`kvarn_perf_run.py` owns the foreground process lifecycle for the end-system
comparison. It refuses to start while the loopback service port is occupied,
uses the fixed-seed random workload, and restarts the service for every entry
in each repeated ABBA sequence. The defaults cover B1 and B4 at prompt lengths
4,096, 16,384, 32,768, and 65,023 with 512 output tokens and eight repeats per
arm. Each service start gets one separate full-width warmup wave and two
measured waves; all of those values are configurable without changing the
matched service profile. The scheduler sampler starts only after the warmup
process finishes and the engine returns to idle, so its peak is evidence from
the measured requests.

Launcher realization and evaluation explicitly use the daemon store. On
Determinate Nix installations, raw app metadata can retain a contextual logical
path rooted at `/`; the runner therefore joins the app's executable basename to
the physical output reported by `nix build --json`. It also requires the app
string context and built package to name the same derivation and output, so a
same-named executable from an unrelated package cannot satisfy resolution.
Benchmark manifests always retain that verified physical immutable program
path.

```bash
run_stamp=$(date -u +%Y%m%dT%H%M%SZ)
scripts/kvarn_perf_run.py \
  --candidate-env "$(readlink -f benchmark-results/kvarn/CANDIDATE/candidate-env)" \
  --correctness benchmark-results/kvarn/CANDIDATE/native-correctness.json \
  --output-dir "benchmark-results/kvarn-perf/$run_stamp"
```

Before a current-build eight-gate correctness manifest exists, use the
explicitly non-promotable exploratory mode rather than weakening or fabricating
formal evidence:

```bash
run_stamp=$(date -u +%Y%m%dT%H%M%SZ)
scripts/kvarn_perf_run.py \
  --exploratory \
  --candidate-env /nix/store/CURRENT-BETA-CANDIDATE \
  --context 4096 \
  --batch 1 \
  --batch 4 \
  --repeats 2 \
  --output-dir "benchmark-results/kvarn-perf/${run_stamp}-exploratory-4k"
```

Only exploratory mode permits omitting `--correctness`, and two repeats per
arm are its minimum: one complete ABBA block. It retains the same immutable
launchers, warmups, raw and detailed results, engine-log checks, actual-process
identity, closure digests, scheduler-overlap evidence, and strict matched
service profile. Each sealed benchmark says `kvarn_evidence_mode: exploratory`
and `kvarn_promotable: false`, while the session carries the corresponding
unprefixed fields; no correctness digest, hard-floor status, or
statistical-parity status is invented. The descriptive
per-cell summary records throughput ratios and pooled tail latency without a
pass/fail claim. A completed exploratory collection exits zero regardless of
the ratios; infrastructure, result-shape, native-dispatch, or provenance
failures still exit two.

Those defaults expect B1 split 24 and B4 split 16. To state them explicitly,
use `--native-splits 1=24 --native-splits 4=16`; one unqualified value applies
to every selected batch. This option declares what the immutable native
launcher must export—it does not override the launcher. Auto must still export
the neutral value `1`, and a mismatch in either arm aborts the trial before a
measurement is accepted.

Use `--plan-only` to realize immutable launcher programs and materialize and
inspect `session.json` without starting a service. `--context` and `--batch`
are repeatable and also accept comma-separated values, for example
`--context 4096,16384 --batch 1`.

For every recorded trial the runner:

1. resolves each mutable flake app once to an immutable Nix store program;
2. verifies the actual API-process argv and allowlisted environment, including
   KV dtype, native switches, eager mode, graph/MTP/prefix-cache exclusions,
   model revision, model length, and sequence limit;
3. validates and retains the detailed result and digest for the separate
   warmup wave, then waits for scheduler idle;
4. polls `vllm:num_requests_running` only until the requested B1/B4 overlap is
   observed, avoiding continuous metrics traffic during the measurement;
5. stops the complete foreground process group and waits for every log writer;
6. scans the final engine log and then hashes it; and
7. leaves vLLM's `benchmark.raw.json` unchanged while atomically writing the
   provenance-augmented `benchmark.json` consumed by `kvarn_perf_gate.py`.

Each context directory receives its gate result after all eight repeats per
arm finish. `session.json` distinguishes execution completion, the 95% hard
floor, and the 98% paired parity target; `SHA256SUMS` seals the entire completed
matrix. A floor or parity miss exits one after retaining every measurement; an
invalid or interrupted run exits two and retains its last durable run manifest.
SIGINT, SIGTERM, and SIGHUP are forwarded to the independent service, warmup,
and measured-benchmark process groups before failure is recorded. The runner
never starts, stops, or reconfigures a systemd service. Safe evidence-preserving
resume is not yet supported: after an interruption, use a fresh output
directory rather than editing or appending to the sealed partial session.

If a native gate fails, stop the foreground process, start
`vllm-xpu-brutus-kvarn-b1` or `vllm-xpu-brutus-kvarn-b4` from the last accepted
closure, wait for `/health`, replay one frozen B1 fixture, and scan the new
engine log before restoring the systemd service. Do not replace the deployed
non-native profile until that recovery sequence and all native gates pass.
