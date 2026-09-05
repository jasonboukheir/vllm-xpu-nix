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
   the 262K ragged B4 decoder oracle at split counts 1, 16, and 24.
2. Start native Kvarn at 65,536/B1 with the selected split count 24. Require
   the native Xe2 dispatch log and reject any fallback, exception, non-finite
   value, or device assertion. Replay the short and boundary fixtures twice
   and require exact token IDs.
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
   65,536. The current selected split counts, B1=24 and B4=16, are eligible
   only after their long primitive and all service gates pass; split 1 remains
   the primitive oracle/control.

The Brutus flake exposes isolated foreground launchers for these phases:

```text
vllm-xpu-brutus-auto-b1
vllm-xpu-brutus-auto-b4
vllm-xpu-brutus-kvarn-b1
vllm-xpu-brutus-kvarn-b4
vllm-xpu-brutus-kvarn-native-b1
vllm-xpu-brutus-kvarn-native-b4
vllm-xpu-brutus-kvarn-native-dpas-b1
vllm-xpu-brutus-kvarn-native-dpas-b4
vllm-xpu-brutus-kvarn-262k-b1
vllm-xpu-brutus-kvarn-262k-b4
vllm-xpu-brutus-kvarn-native-262k-b1
vllm-xpu-brutus-kvarn-native-262k-b4
vllm-xpu-brutus-kvarn-native-dpas-262k-b1
vllm-xpu-brutus-kvarn-native-dpas-262k-b4
```

The natural and DPAS native launchers explicitly set the decode, materializer,
persistent-scratch, layout, and split-count switches. The auto launchers
explicitly disable native execution and keep the natural layout. Dedicated
`native-dpas` launcher outputs are required because the foreground wrapper
scrubs inherited Kvarn behavior variables before exporting its own values; a
runner environment override cannot select DPAS honestly. That makes an
inherited shell variable unable to silently change one arm.

## Automated correctness manifest

`kvarn_correctness_run.py` produces the eight-gate manifest consumed by the
formal performance runner. It runs two native primitive pytest phases and six
foreground service starts, in this fixed order:

1. native 65K/B1 first pass: four exact-length fixtures, same-process replay,
   and cancellation/reuse at the 65,023-token boundary;
2. native 65K/B1 restart: exact comparison with the first process;
3. native 65K/B4: one full-width concurrent wave, observed scheduler overlap,
   and exact comparison of every stream with its B1 result;
4. compact non-native 262K/B1: one 261,631 + 512 reference completion;
5. native 262K/B1: one completion compared exactly with the reference; and
6. restarted native 262K/B1: one completion compared with both prior results.

The long prompt is generated deterministically in memory from the checked-in,
SHA-pinned reasoning fixture. Its length and token-ID SHA-256 are retained,
but the 261,631 prompt IDs are not copied into result JSON. The HTTP request
necessarily contains the IDs. Output token IDs, comparison results, redacted service
profile, actual process package, candidate/process closure digests, engine log,
and engine-log scan remain durable.
Every completion must also report OpenAI usage matching the exact prompt
length and 512 generated tokens, terminate for `length`, and return exactly
512 token IDs.

The runner independently resolves `vllm-xpu-brutus` from the pinned
configuration, requires that exact package in the supplied candidate
environment's closure, and then requires every captured engine process to use
that exact package. A look-alike environment containing a different vLLM build
therefore fails before a manifest can pass. `--config-ref` must resolve to the
same local tree as `--config-repo`; the runner will not verify one checkout and
evaluate another. That configuration checkout must be clean; its HEAD and
complete tracked-tree digest are captured and rechecked around every package
or launcher resolution and once more before manifest emission.

The correctness harness is a separate identity from the candidate packaging
input pinned in that lock. Its own checkout must be clean, and the manifest
records and rechecks that checkout's actual HEAD and complete tracked-tree
digest. It also snapshots the flake lock, correctness runner,
performance-gate and performance-runner helpers, service helper, and
engine-log scanner into the result tree and retains their SHA-256 values. The
harness HEAD is deliberately not required to equal the older
`vllm-xpu-release` candidate revision.

For every service phase it also pins the bounded continuation-prefill window
to 16 fp16 blocks and removes the unbounded
`VLLM_KVARN_DEFER_PREFILL_FLUSH` diagnostic from the inherited environment.
The captured process environment must prove both conditions.

The primitive Python must contain pytest, Transformers, the candidate vLLM
package, and the candidate kernel extension, while retaining the package's
pinned Level Zero/oneAPI wrapper. This expression builds such an environment
from the exact Brutus package without running the GPU:

```bash
candidate_env="$(
  nix build --store daemon --no-link --print-out-paths --impure --expr '
    let
      config = builtins.getFlake "path:/home/jasonbk/.config/nix";
      pkgs = config.inputs.nixpkgs.legacyPackages.x86_64-linux;
      package = config.packages.x86_64-linux.vllm-xpu-brutus;
      pythonEnv = package.pythonModule.withPackages (ps: [ package ps.pytest ]);
      wrapperArgs = builtins.filter
        (arg: !(pkgs.lib.hasPrefix "--prefix PYTHONPATH " arg))
        package.makeWrapperArgs;
    in pkgs.symlinkJoin {
      name = "vllm-kvarn-correctness-env";
      paths = [ pythonEnv ];
      nativeBuildInputs = [ pkgs.makeWrapper ];
      postBuild =
        "rm -f \"$out/bin/python\" \"$out/bin/python3\" \"$out/bin/python3.12\"\n"
        + "makeWrapper ${pythonEnv}/bin/python \"$out/bin/python\" "
        + builtins.concatStringsSep " " wrapperArgs
        + "\nln -s python \"$out/bin/python3\""
        + "\nln -s python \"$out/bin/python3.12\"";
    }
  '
)"
test -x "$candidate_env/bin/vllm"
test -x "$candidate_env/bin/python"
```

Wait for that build to finish before starting any correctness phase. A
CPU-only plan resolves and seals all immutable launchers and package-import
origins without initializing an XPU or starting a service:

```bash
run_stamp=$(date -u +%Y%m%dT%H%M%SZ)
scripts/kvarn_correctness_run.py \
  --candidate-env "$candidate_env" \
  --primitive-python "$candidate_env/bin/python" \
  --plan-only \
  --output-dir "benchmark-results/kvarn/${run_stamp}-correctness-plan"
```

The real run owns the foreground processes and refuses to begin unless the
loopback API port is free and both deployed vLLM units are inactive:

```bash
sudo systemctl stop vllm-xpu-chat.service vllm-xpu-embedding.service
test "$(systemctl is-active vllm-xpu-chat.service)" = inactive
test "$(systemctl is-active vllm-xpu-embedding.service)" = inactive

run_stamp=$(date -u +%Y%m%dT%H%M%SZ)
scripts/kvarn_correctness_run.py \
  --candidate-env "$candidate_env" \
  --primitive-python "$candidate_env/bin/python" \
  --output-dir "benchmark-results/kvarn/${run_stamp}-native-correctness"
```

The production/formal default is `--native-layout natural`. To validate the
separate Xe2 DPAS cache layout, add `--native-layout xe2_dpas`; this selects
the dedicated `native-dpas-b1`, `native-dpas-b4`, and
`native-dpas-262k-b1` launchers while the non-native reference remains
natural. The correctness manifest seals the selected layout, every service
phase seals its actual launcher and captured
`KVARN_NATIVE_XPU_DPAS_LAYOUT`, and primitive evidence records the selection.

The command exits zero only after writing
`native-correctness.json`, whose `candidate_id` is the resolved correctness
environment store path. Use that same `candidate_env` for formal performance
trials. An all-skipped primitive suite, mismatched flake lock, modified native
oracle source, extension outside the candidate closure, service/profile/closure
mismatch, missing native dispatch, fallback, fatal log finding, missed B4
overlap, or token mismatch prevents a passing manifest. Interrupt signals are
forwarded to the complete primitive or service process group, and partial
phase evidence is retained without a passing manifest.

Primitive phases disable external pytest plugins and scrub Python/pytest path
injection, probe every imported package origin, and hash the complete tracked
kernel checkout before and after pytest. The two deployed units remain
mandatory even when extra inactive units are requested and are rechecked
before every primitive or service process; a startup failure is not retried.
Cancellation closes after exactly 257 returned delta token IDs, then requires
running, waiting, and KV-cache-usage gauges all to reach zero before reuse.
Before emission—and again during performance-gate loading—every nested JSON
artifact reference is recursively re-read and rehashed. Both producer and
consumer also validate each gate-specific evidence schema: primitive JUnit
counts and source references; exact replay/restart comparisons; the exact
257-token cancellation checkpoint and zero idle gauges; four-way B4 overlap;
and the reference/native/restarted near-262K results. A correctly hashed but
empty, incomplete, or mislabeled gate file is not accepted.

The separate mixed 262K/B4 cancellation scenario in step 6 remains extended
release evidence. The current performance-gate schema has no ninth field for
it; the eight-gate manifest therefore does not silently claim that scenario
ran. Its 262K B4 coverage is the native ragged primitive oracle, while exact
service equivalence is B1.

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

For quick manual probes, use two primary workloads. These do not replace the
complete automated formal matrix below:

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
dtype, native switch, explicit cache layout and effective layout environment,
split count, arm, scheduler peak-running evidence,
unique run UUID, timezone-qualified start time, global ABBA run order, the
engine-log SHA-256, and the SHA-256 of the passed correctness artifact.
It also seals the executable and package observed in `/proc`, the actual
process-package closure digest, the candidate closure digest, and the
canonical matched-profile digest. Closure paths are sorted and deduplicated
before hashing, so Nix query order cannot alter the identity.

Optimization evidence also carries explicit `kernel_strategy`, `split_policy`,
`fusion_strategy`, `scheduling_variant`, and `variant_id` fields. It separately
binds `kvarn_flush_writer`, `kvarn_prefill_store`, `kvarn_native_frontend`, and
`kvarn_forward_pool_ensure` to the corresponding captured process values. These
values are derived from the selected harness settings rather than accepted as
free-form labels. The current native identity records the Xe2 qlen=1 reader, fixed
B1=24/B4=16 split policy, native materializer with persistent scratch, and the
eager MNBT=2048 schedule; the auto arm is labeled separately as the performance
control. This is deliberately a small provenance record, not a generalized
experiment framework.

Use focused exploratory cells to screen optimization variants. Only finalists
should pay for the full selected-layout correctness suite and B70-only repeated
ABBA timing matrix. In those final gates, natural-layout non-native Kvarn
remains the correctness reference, and vLLM `auto` remains the performance
control.

The canonical profile comes from the actual service argv and effective
allowlisted environment. The only normalized arm differences are
`--kv-cache-dtype`, the five native identity switches (`KVARN_NATIVE_XPU`,
decode, DPAS layout, materialize, and persistent scratch), and the explicitly
selected split, writer, prefill-store, frontend, and forward-pool axes. Auto is
forced to the `reference` frontend/store paths and
`KVARN_FORWARD_POOL_ENSURE=always`. Any other argument or effective
performance-environment difference invalidates the cell. Secret-looking
argument values are redacted before argv or profile evidence is written.
The runner pins `KVARN_PREFILL_FP16_WINDOW_BLOCKS=16` identically in both
arms, removes `VLLM_KVARN_DEFER_PREFILL_FLUSH` from its launch environment,
and captures both variables from `/proc`; either a different window or an
active full-defer diagnostic invalidates the trial.
It also removes ambient `VLLM_*`, Python path/startup/user-base variables, and
`LD_PRELOAD`, and disables the Python user site. The foreground launcher then
restores its declared vLLM environment, which is what the runner captures.

The auto reference always keeps natural layout. The native candidate uses the
layout explicitly selected by `--native-layout`; natural remains the default,
while `xe2_dpas` is a separate validation mode and must match the supplied
correctness manifest. The auto reference uses the neutral split value `1`;
the native candidate uses the empirically selected value `24` for B1 and `16`
for B4. The split
variable is unreachable when the native reader is disabled, so copying the
candidate's tuning value into auto would add misleading provenance without
making the execution more closely matched. The gate instead permits only this
one named native-only difference and verifies both values exactly.

Current vLLM emits the exact
`Using the native Xe2 KVarN qlen=1 decoder` marker but does not emit a
layout-specific marker. Accordingly, the harness verifies the exact native
marker plus the effective layout variable captured from the engine process,
and seals `native_layout_log_marker: unavailable`. This proves the selected
configuration and native dispatch without pretending that the log itself
distinguishes natural from DPAS. It does not establish that DPAS is the
no-environment production default.

Built-in model, tokenizer, backend, request-rate, prompt-count, duration,
token totals, and concurrency fields must also match. The gate requires eight
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
arm. Formal qualification requires that complete matrix; context or batch
subsets are available only in explicitly non-promotable exploratory mode. Each
service start gets one separate full-width warmup wave and two measured waves.
An explicit formal warmup count must be at least four so the B4 warmup remains
full-width. The scheduler sampler starts only after the warmup process finishes
and the engine returns to idle, so its peak is evidence from the measured
requests. Formal mode fixes the decode at 512 output tokens and requires at
least two measured waves. Its command-line acceptance values may be made
stricter, but cannot weaken the 95% throughput/per-request floor, 110% p99
latency ceiling, 98% paired parity target, or four-pair minimum. The repeat
count must supply every requested ABBA pair.

Before starting the matrix, the runner uses the candidate's pinned Python and
Torch to allocate, synchronize, and read back an XPU tensor. Formal or
exploratory measurements proceed only when exactly one visible device is
`Intel(R) Arc(TM) Pro B70 Graphics`. Every sealed run references the hashed
hardware preflight and its own hashed warmup. Its final engine log must report
`device_config=xpu`, positive consumed-model memory, and positive KV-cache
memory; native candidates must additionally report the Xe2 KVarN decoder
dispatch. Therefore CPU measurements, another accelerator model, a cold run,
or an XPU-configured process without resident model/KV state cannot enter a
performance or parity result.

The enforced GPU proof is the candidate-Torch B70 compute operation plus the
resident XPU service and, for Kvarn, native decoder dispatch. Per-run Level Zero
Sysman utilization is intentionally not an acceptance input yet: it commonly
requires elevated permissions, and high-frequency sampling can perturb the
latencies being measured. Do not describe the artifacts as sampled engine
utilization. Model-weight offload and KV-cache offload/transfer options are
rejected so positive XPU residency cannot hide an offloaded comparison.

CPU-only checks may still be useful as correctness diagnostics, but they are
not promotable performance evidence and cannot satisfy the parity gate.

Launcher realization and evaluation explicitly use the daemon store. On
Determinate Nix installations, raw app metadata can retain a contextual logical
path rooted at `/`; the runner therefore joins the app's executable basename to
the physical output reported by `nix build --json`. It also requires the app
string context and built package to name the same derivation and output, so a
same-named executable from an unrelated package cannot satisfy resolution.
Benchmark manifests always retain that verified physical immutable program
path. The supplied 262K correctness manifest must identify that same candidate
environment, process package, process-closure digest, and candidate-closure
digest; performance from a rebuilt or substituted candidate cannot inherit an
older correctness result.

```bash
run_stamp=$(date -u +%Y%m%dT%H%M%SZ)
scripts/kvarn_perf_run.py \
  --candidate-env "$(readlink -f benchmark-results/kvarn/CANDIDATE/candidate-env)" \
  --correctness benchmark-results/kvarn/CANDIDATE/native-correctness.json \
  --output-dir "benchmark-results/kvarn-perf/$run_stamp"
```

For DPAS validation, pass `--native-layout xe2_dpas` to both the correctness
and performance commands. The performance runner refuses a correctness
manifest from another layout, uses the dedicated DPAS launchers only for the
candidate, and restarts every reference and candidate service exactly as in
natural mode.

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

`--native-frontend qkv_scatter_inline` requires two independent runtime proofs:
the existing fused-QKV active marker and
`[KVARN_FRONTEND_INLINE] active=qkv_scatter_inline; wrapper=unified_qkv_attention_with_output;`,
which is emitted only when the combined wrapper executes.
`--native-frontend qkv_scatter_inline_current_stream` requires those same two
proofs plus an active qlen=1 frontend line containing
`native_op=kvarn_hadamard_qkv_scatter_current_stream; qlen=1;`. This experiment
is screened only on the single eager, in-order XPU stream until a separate
stream-identity receipt qualifies it for promotion.
`--qlen1-inline-plan bound_native_v2` requires the exact
`[KVARN_BOUND_QLEN1_INLINE] active=bound_native_v2;` execution marker. ID22
requires `[KVARN_FACTORY] ID22 last-arrival fused reduction active;`; selecting
ID22 at startup but downgrading to ID18 at runtime fails the selected-variant
gate.
The service-only `--forward-pool-ensure always|epoch_latch|fused_qkv_proof`
selector is
captured in service profiles and sealed provenance. `fused_qkv_proof` requires
a fused QKV frontend and the runtime marker
`[KVARN_FORWARD_POOL_ENSURE] active=fused_qkv_proof; action=elide_ensure_pool;`.
Both service-only axes are deliberately absent from direct primitive results;
reference and inapplicable paths must not emit their active markers.
Round-7 service runs additionally expose `--metadata-lifecycle
reference|incremental_qlen1`. Variant A is `epoch_latch` with reference
metadata, Variant B is `always` with incremental metadata, and the combined
candidate selects both. Auto/non-native reference phases always remain
`always` plus `reference`. Incremental results require the exact
`[KVARN_METADATA_LIFECYCLE] active=incremental_qlen1;
action=elide_full_lifecycle_scan` marker; reference phases reject it.
Every sealed performance result records the verified execution booleans and
marker strings plus the run-local engine-log scan path and SHA-256. The scan
also binds the engine-log SHA-256, so either artifact changing invalidates the
result before comparison.

Use `--plan-only` to realize immutable launcher programs and materialize and
inspect `session.json` without starting a service. `--context` and `--batch`
are repeatable and also accept comma-separated values, for example
`--context 4096,16384 --batch 1`.

For every recorded trial the runner:

1. resolves each mutable flake app once to an immutable Nix store program;
2. verifies the actual API-process argv and allowlisted environment, including
   KV dtype, native switches, eager mode, graph/MTP/prefix-cache exclusions,
   bounded prefill window, disabled full-defer diagnostic, model revision,
   model length, and sequence limit;
3. validates and retains the detailed result and digest for the separate
   full-width warmup wave, binding it to the run UUID, arm, workload, raw
   result, process/candidate closures, and matched service profile, then waits
   for scheduler idle;
4. polls `vllm:num_requests_running` only until the requested B1/B4 overlap is
   observed, avoiding continuous metrics traffic during the measurement;
5. stops the complete foreground process group and waits for every log writer;
6. scans the final engine log, requires positive XPU residency, and then hashes
   it; and
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

## GPU timeline diagnostics (not a performance gate)

Use `kvarn_xpu_profile.py` to explain a gap found by the unprofiled parity
runner. It captures a bounded 20–50-step Kineto device timeline from a fresh
foreground service. The runner skips the conservative upper bound for chunked
prefill plus four settled decode iterations, disables stacks, shapes, memory,
FLOP accounting, and frontend tracing, and then summarizes only
positive-duration XPU kernel events and per-stream idle gaps.

This is intentionally separate from the performance gate. Kineto perturbs the
workload, so neither the raw benchmark result nor any time in
`profile-summary.json` is eligible for throughput, latency, parity, promotion,
or acceptance conclusions. The summary repeats that restriction in machine
readable fields. CPU annotations label steady decode steps and native Kvarn
regions; CPU durations are not reported as GPU performance evidence.

The command refuses to run unless the candidate's pinned Torch completes an
XPU tensor operation on exactly one
`Intel(R) Arc(TM) Pro B70 Graphics`. It also requires the final service log to
show `device_config=xpu`, positive model and KV residency, and, for the
candidate arm, native Xe2 dispatch with no fallback. A trace is valid only when
it contains the requested number of all-generation B1/B4 steps and at least one
positive XPU kernel in every step.

Capture auto and natural-layout native diagnostics into separate durable
directories:

```bash
stamp=$(date -u +%Y%m%dT%H%M%SZ)
scripts/kvarn_xpu_profile.py \
  --candidate-env /nix/store/CURRENT-BETA-CANDIDATE \
  --arm reference --context 65023 --batch 1 --profile-steps 32 \
  --variant-id auto-natural-b1 \
  --launcher vllm-xpu-brutus-auto-b1 \
  --output-dir "benchmark-results/kvarn-profile/${stamp}-auto-b1-65k"

scripts/kvarn_xpu_profile.py \
  --candidate-env /nix/store/CURRENT-BETA-CANDIDATE \
  --arm candidate --context 65023 --batch 1 --profile-steps 32 \
  --variant-id native-xe2-natural-split24-b1 \
  --native-layout natural --native-splits 24 \
  --launcher vllm-xpu-brutus-kvarn-native-b1 \
  --output-dir "benchmark-results/kvarn-profile/${stamp}-native-b1-65k"
```

For the DPAS cache-layout experiment, make the layout and its dedicated
launcher explicit. B4 currently uses split count 16:

```bash
stamp=$(date -u +%Y%m%dT%H%M%SZ)
scripts/kvarn_xpu_profile.py \
  --candidate-env /nix/store/CURRENT-BETA-CANDIDATE \
  --arm candidate --context 65023 --batch 4 --profile-steps 32 \
  --variant-id native-xe2-xe2_dpas-split16-b4 \
  --native-layout xe2_dpas --native-splits 16 \
  --launcher vllm-xpu-brutus-kvarn-native-dpas-b4 \
  --output-dir "benchmark-results/kvarn-profile/${stamp}-dpas-b4-65k"
```

The explicit launcher is a provenance assertion, not an arbitrary override:
it must match the selected arm, layout, and batch. The runner resolves it to a
realized immutable Nix-store program and records both names, the captured
layout environment, split count, process/candidate closure digests, engine-log
digest, hardware-preflight digest, Kineto trace digest, profiler configuration,
per-step device-kernel totals, top kernels by device time, and queue idle gaps.
The variant block also records the cache layout, kernel strategy, split count,
fusion selection, frontend, forward-pool guard, and scheduling selection. A compact GPU-only leaderboard
block provides total device-kernel time, launch count, union-busy time, device
span, and idle fraction. These are useful for ranking factory experiments
inside the same profiler setup, but remain diagnostic and non-promotable.
