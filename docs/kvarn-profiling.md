# CPU/XPU profiling

Keep reusable profiling tools here; kernel implementations and their correctness
tests belong in the source forks. Generated traces and reports belong under the
ignored `benchmark-results/` directory, not in repository progress notes.

Run GPU commands serially with competing services stopped. Never change host
counter permissions automatically. Choose fresh output paths for every capture.

## Runtime and service captures

Wrap an immutable candidate environment with its matching PTI libraries:

```bash
nix build --impure --file nix/profile-env.nix \
  --argstr candidateEnv '/nix/store/<candidate-env>' \
  --argstr ptiDir '/nix/store/<intel-pti>' \
  --arg enableMetrics true --out-link /tmp/kvarn-profile-env

/tmp/kvarn-profile-env/bin/python -m scripts.kvarn_xpu_profile_smoke \
  --output benchmark-results/profile-new/smoke.pt.trace.json

/tmp/kvarn-profile-env/bin/python -m scripts.kvarn_vision_run \
  --service-env /tmp/kvarn-profile-env \
  --cache-dtype kvarn_k4v4_g128_compact --draft-tokens 2 \
  --max-model-len 131072 --profile-workload text-4k \
  --output benchmark-results/profile-new/kvarn
```

Substitute actual store paths. The service runner supports `text-4k`,
`short-image`, and `long-image` warmed off/on/off capture brackets. Repeat with
`--cache-dtype auto` and a separate output directory for a matched comparison.
Use a context limit that fits both arms; all other workload/runtime settings
must match. Add `--profile-stacks` for source callers, and measure its overhead
separately from lightweight capture. `--draft-tokens 0` disables MTP.

The older `kvarn_xpu_profile` service CLI uses historical factory launchers and
is not the launch path for current releases. Its shared trace/config helpers
and the offline analyzers remain usable. The retired experiment-plan generator
and Sinkhorn trial selectors are not part of this workflow.

### Matched text-only prefill

Use the same immutable service environment for both cache dtypes. The bounded
prefill suite disables image inputs and requires MTP off. It constructs an exact
token length through the pinned model's tokenizer, warms the identical request,
then records three profiler-off requests with 512 generated tokens and EOS
stopping disabled. Raw token IDs and streaming arrival times are retained.

```bash
/tmp/kvarn-profile-env/bin/python -m scripts.kvarn_vision_run \
  --service-env /tmp/kvarn-profile-env --cache-dtype auto --draft-tokens 0 \
  --max-model-len 65536 --prefill-suite --prefill-tokens 65023 \
  --output benchmark-results/prefill-new/auto

/tmp/kvarn-profile-env/bin/python -m scripts.kvarn_vision_run \
  --service-env /tmp/kvarn-profile-env \
  --cache-dtype kvarn_k4v4_g128_compact --draft-tokens 0 \
  --max-model-len 65536 --prefill-suite --prefill-tokens 65023 \
  --output benchmark-results/prefill-new/kvarn

python -m scripts.kvarn_prefill_compare \
  --pair benchmark-results/prefill-new/auto benchmark-results/prefill-new/kvarn \
  --output benchmark-results/prefill-new/service-comparison.json
```

Repeat in reversed arm order with fresh directories, and include another
representative length such as `--prefill-tokens 16383`. Pass each matched pair
with another `--pair`. The comparator validates manifests, runtime identity,
exact requests and processed prompt IDs. It reports individual request timings,
run variation, token-based decode throughput and client inter-token latency.
It rejects bundled token events as insufficient evidence for individual token
latency. Cross-dtype output equality is reported separately from performance.

For separate diagnostic captures, replace `--prefill-suite` with
`--profile-workload text-prefill` and choose fresh directories. This records an
off/on/off request bracket after warmup. Profiling begins at the first worker
step and covers every prefill chunk plus two decode boundary steps. The capture
includes operator shapes for checking attention and materializer inputs. The
shared profiler helper limits captures to 50 worker steps; longer workloads need an
explicitly qualified extension. Profiled request time includes trace-export
overhead and cannot establish service performance.

```bash
python -m scripts.kvarn_prefill_trace \
  --run benchmark-results/prefill-new/kvarn-profile \
  --output benchmark-results/prefill-new/kvarn-attribution.json
```

The analyzer checks exact chunk extents, the prefill-to-decode boundary, trace
hashes and correlated CPU origins. It retains initial, early, middle, late and
boundary slices, device-busy unions, family duration sums and unresolved
out-of-scope work. Worker traces do not cover frontend or scheduler work outside
the worker. Neither uncovered time nor CPU wait totals prove removable overhead.

### Chunk-budget experiment workflow

For a scheduler experiment, pass `--max-num-batched-tokens 2048`, `4096`, or
`8192` to the same service runner. The default remains 2048. The runner records
the actual service argument and each workload's expected chunk extents and
profile step count. Attribution reads that recorded budget and rejects missing
chunks, unexpected scheduler alignment, or incomplete decode boundary steps.
Expected extents in an unprofiled workload are a plan, not an observed schedule.
Full traces also retain the packing operation sequence and input shapes;
these do not substitute for checking committed page bytes and tail state.

1. Freeze the paired runtime and checkpoint revision. Record the service
   environment store path, source identities, commands, and acceptance thresholds
   in a fresh artifact directory before timing. Use B1, context 65536, 512 output
   tokens, no EOS stopping, eager mode, no prefix caching, MTP off, and the same
   0.90 memory-utilization limit. Run serially on an idle GPU.
2. Screen 2K/4K/8K with both 16383 and 65023 input tokens, separately for `auto`
   and `kvarn_k4v4_g128_compact`. Each service invocation warms the exact request
   and collects three measured repetitions. Repeat eligible arms with fresh
   service starts in reversed budget order. Retain 2K controls even when larger
   arms fail. Stop an arm on OOM, preemption, incorrect output/state, unacceptable
   memory growth, or absence of repeatable TTFT gain. A change in freely
   generated wording is a diagnostic, not proof of incorrect output.
3. Compare each candidate against 2K **within its cache dtype**:

   ```bash
   python -m scripts.kvarn_chunk_compare \
     --pair benchmark-results/chunks-new/auto-16k-2k-r1 \
            benchmark-results/chunks-new/auto-16k-4k-r1 \
     --pair benchmark-results/chunks-new/auto-16k-2k-r2 \
            benchmark-results/chunks-new/auto-16k-4k-r2 \
     --output benchmark-results/chunks-new/auto-16k-4k-comparison.json
   ```

   The audit preserves dtype and all non-budget service settings, checks raw
   requests/prompt IDs, and reports output equality including warmups, TTFT,
   total time, decode throughput, client p95/p99 inter-token latency, sampled DRM
   allocation/residency peaks, preemptions, and fatal engine-log findings. It
   never marks a candidate qualified. Compare measured gain against the
   predeclared threshold and control variation, including each separate start.
   DRM sampling is not an instantaneous Torch allocator high-water mark.
4. Assess numerical accuracy against the model with **unquantized KV at each
   budget**. Keep the same model weights, pinned runtime, prompt IDs, and one
   frozen reference continuation for every arm. Use persistent teacher forcing
   to compare scores before selecting the next token, so earlier word choices
   cannot confound later errors. Compare KVarN versus `auto` at 2K and at 8K,
   report the change in compression error, and separately compare `auto` across
   budgets to measure ordinary numerical sensitivity. The 2K KVarN output is
   not an accuracy oracle, and exact free-generation equality is not a gate.
   Apply the retained numerical thresholds unchanged; candidates still require
   replay, ordinary-attention, GDN-continuation, and committed packed-history/tail
   gates. Exact state invariants remain exact. Do not relax a gate because
   changing chunks changes the compression schedule.
5. Only carry a correctness-passing, repeatable service winner into a matched
   full trace using `--profile-workload text-prefill` and the same budget.
   Attribute GEMM, attention, reconstruction, and host work separately, then
   qualify supported MTP2 and representative images before recommendation.
   Record rejected arms and unperformed gates explicitly. This workflow does
   not authorize a deployment or release.

### Start the model-reference replay stage

`scripts.kvarn_chunk_accuracy` freezes prompts and a 512-token continuation from
retained unquantized-KV service captures, snapshots its source, and runs every
arm serially in an owned process group. It prioritizes 2K/8K at both lengths
before 4K, then reverses budget/dtype order on the second start. It records a
plan before submitting GPU work and updates `status.json` and
`comparison-summary.json` after each completed arm.

```bash
/tmp/kvarn-profile-env/bin/python -m scripts.kvarn_chunk_accuracy \
  --service-env /tmp/kvarn-profile-env \
  --reference-run benchmark-results/<auto-16k-control> \
  --reference-run benchmark-results/<auto-65k-control> \
  --threshold-report benchmark-results/<retained-thresholded-comparison.json> \
  --budgets 2048 8192 4096 --starts 2 \
  --output benchmark-results/chunk-accuracy-new
```

Use actual retained paths and the same immutable environment that produced the
references. `--plan-only` prepares a separate reviewed plan without GPU work;
normal execution requires a fresh output directory. The threshold report must
contain the complete existing numerical profile, not newly loosened cutoffs.
All arms receive identical forced tokens for a given prompt, including after
any position where free generation would have diverged.

This bounded stage captures the top 50 raw logits plus the forced token. It
reports top-1/top-5 agreement, matched/selected score errors, and coverage;
it does not estimate full-vocabulary KL. The same W4A16 weights are used for
both cache dtypes: `auto` is an unquantized **KV** reference, not unquantized
model weights. A numerical-screen pass does not qualify deployment or replace
the full six-fixture model gate, state checks, profiler-off performance trials,
MTP2, or images. Preserve numerical failures for diagnosis; stop execution on
invalid logits, engine failure, or changed artifact identity.

### Native single-group W4A16 experiment

Issue [#10](https://git.sunnycareboo.com/jasonbk/vllm-xpu-nix/issues/10)
screens the existing Xe2 grouped INT4 GEMM as an E=1 prefill backend. The
workflow below captures the actual loaded model operands, checks exact format
adaptation and correctness, then compares complete operator calls. It does not
install a serving backend. Keep the native policy and determinism settings fixed.

Use the immutable xpu-v1.7.2 profiling environment and the pinned checkpoint
revision from [the predeclared plan](../fixtures/xpu-w4a16-grouped-plan.json).
The plan retains the frozen grouped-GEMM reference tolerance (`atol=rtol=0.01`)
and existing BF16 batch-separability tolerance (`atol=rtol=0.015625`). The tools
reject changed numerical thresholds. Run serially on an idle B70, with fresh
output directories. For a deliberately newer control, record and qualify its
source/runtime pair separately before comparing candidates.

```bash
EXPERIMENT=benchmark-results/grouped-int4-new
RUNTIME=/nix/store/88fvzjx1ar98dlm9mdw715c22zz1r2i0-vllm-xpu-profile-env
RETAINED=benchmark-results/kvarn/issue7-prefill-20260908
export HF_HOME=/var/cache/huggingface HF_HUB_OFFLINE=1
export OCL_ICD_VENDORS=/nix/store/gcwa832p0329y80drcqhslmnzhw49bh8-intel-compute-runtime-26.27.39122.11/etc/OpenCL/vendors
export VLLM_CACHE_ROOT="$PWD/$EXPERIMENT/runtime-cache"

"$RUNTIME/bin/python" -m scripts.xpu_w4a16_capture \
  --output "$EXPERIMENT/capture" \
  --projection language_model.model.layers.0.mlp.gate_up_proj \
  --projection language_model.model.layers.0.mlp.down_proj \
  --tokenize "$RETAINED/auto-16k-r1/prefill-16383-0-tokenize.json" \
  --tokenize "$RETAINED/auto-65k-r1/prefill-65023-0-tokenize.json"

"$RUNTIME/bin/python" -m scripts.xpu_w4a16_grouped_probe \
  --capture "$EXPERIMENT/capture" \
  --plan fixtures/xpu-w4a16-grouped-plan.json \
  --output "$EXPERIMENT/screen"

"$RUNTIME/bin/python" -m scripts.xpu_w4a16_grouped_bench \
  --capture "$EXPERIMENT/capture" --screen "$EXPERIMENT/screen" \
  --output "$EXPERIMENT/operator-r1"

"$RUNTIME/bin/python" -m scripts.xpu_w4a16_grouped_bench \
  --capture "$EXPERIMENT/capture" --screen "$EXPERIMENT/screen" \
  --output "$EXPERIMENT/operator-r2" --reverse
```

The retained tokenization files are local evidence, not repository fixtures.
If unavailable, first obtain exact tokenizations using the matched text-only
prefill workflow above. Choose the projections from the actual trace's operator
attribution and confirm their loaded dimensions. The example selects gate/up
K/N=5120/34816 and down K/N=17408/5120. Capture preserves raw prompt/output IDs,
actual M=2048/1535/2047 operands, all loaded W4A16 layer metadata, tensor hashes,
and the kernel binary identity. It uses a diagnostic in-process engine with
ordinary KV, eager execution, B1, MTP off, no prefix cache and 512 output tokens;
its timings do not qualify service performance.

The bounded adapter accepts the loaded NT int32 `[K/8,N]` view, contiguous BF16
G128 scales, scalar int8 zero point 8 and absent `g_idx`. It rejects other
formats. XOR with `0x88` produces signed int8 storage `[1,N,K/2]`; scales are
transposed to `[1,N,K/128]` without changing their values, including finite
negative scales present in this checkpoint. Independent word and byte unpackers
compare **every** logical integer and dequantized weight. No requantization or
second full-model GPU copy is involved in the screen.

The first screen checks the native dispatch trace, output ownership/canaries,
reference accuracy, captured oneDNN compatibility and exact repeated output.
The second stage requires exact oneDNN replay, M=4 versus 4×M=1 and sampled
prefill-row separability, optional bias, changed intervening weights/scales,
allocation pressure and input noninterference. Native and oneDNN are each warmed
three times, then measured in 20 pairs with alternating order. The second
process reverses projection, M and initial arm order. Timings include output
allocation, native rows metadata, operator scratch/launch and synchronization;
prepacking/upload are outside steady-state timing and reported separately.
Reports retain raw pair timings and allocator memory. Check exit status and the
report's `status` before advancing; no script declares service qualification.

Stop on failed correctness or absent repeatable weighted operator benefit.
If a candidate qualifies, account for retained fallback formats across all
selected model layers **before** integration. Then run paired profiler-off
16K/65K control/candidate trials within each cache dtype, reverse arm order over
multiple service starts, and apply the existing model/replay/state gates before
MTP2 and images. The predeclared service threshold is greater than both 3% and
twice measured relative control variation. TTFT, total time, decode throughput,
client p95/p99 token latency, memory and preemptions all remain required.
Keep evidence and a negative-result decision under the ignored artifact path;
record its hashes and limitations in the PR. No release/deployment is authorized.

### Follow-up: specialized dense INT4 policies

The modified-kernel follow-up has a separate
[plan](../fixtures/xpu-w4a16-dense-plan.json). Its two opt-in policies use
128x128 and 256x128 output tiles, each with a 4x4 subgroup layout. They also
explicitly round dequantized BF16 weights to nearest-even. A synthetic identity
probe found that the original kernel's BF16 scale multiplication truncates;
the original path is retained as a control. Candidate numerical thresholds
remain unchanged. This compares the combined tile and rounding modifications.

Build only the grouped Xe2 library from the experimental kernels checkout:

```bash
nix build --impure --file nix/xpu-grouped-int4-experiment.nix \
  --argstr kernelsSrc /absolute/path/to/experimental-kernels \
  --cores 4 --max-jobs 1 --out-link /tmp/dense-int4-library -L
```

Use the frozen base revision from the plan plus the experiment's source patch
for the controlled build. Applying that patch to a newer fork main also brings
in newer shared build sources; record that as a different build. The helper
preserves the flake's pinned Torch/toolchain and existing dependent libraries.
It does not rebuild the Python binding or install a serving runtime.

Run the fork's targeted `int4_dense` tests first, with the candidate library
loaded. Then reuse the captured operands and passing reference screen:

```bash
LIBRARY=$(readlink -f /tmp/dense-int4-library)/lib/libgrouped_gemm_xe_2.so
export LD_LIBRARY_PATH="$(dirname "$LIBRARY"):${LD_LIBRARY_PATH:-}"

"$RUNTIME/bin/python" -m scripts.xpu_w4a16_dense_bench \
  --capture "$EXPERIMENT/capture" --reference-screen "$EXPERIMENT/screen" \
  --library "$LIBRARY" --baseline-info fixtures/xpu-w4a16-dense-plan.json \
  --output "$EXPERIMENT/dense-r1"

"$RUNTIME/bin/python" -m scripts.xpu_w4a16_dense_bench \
  --capture "$EXPERIMENT/capture" --reference-screen "$EXPERIMENT/screen" \
  --library "$LIBRARY" --baseline-info fixtures/xpu-w4a16-dense-plan.json \
  --output "$EXPERIMENT/dense-r2" --reverse
```

The runner checks the unchanged binding's hash, the actual mapped grouped DSO's
path/hash, and each native policy's device trace. Every shape passes the
correctness screen before timing begins. The five arms are oneDNN, original
native with fresh rows metadata, original native with cached rows, and both
modified policies with cached rows. Caching immutable rows is measured
separately; output allocation, internal scratch, launch and synchronization
remain inside every complete-call timing.

Each arm must stabilize across three consecutive windows of at least 100 ms,
after at least one second of warmup. Each arm's window medians must span at most
1%; a ten-second cap records failure to stabilize. Thirty balanced blocks place
each arm twice at symmetric positions and rotate positions across blocks.
Reports preserve raw calls, paired block changes, block bootstrap intervals and
first-versus-last-third drift flags. An unstable report does not qualify a win.

Also run a fresh process with the original immutable DSO and
`--arms onednn original_uncached original_cached`; compare its default path to
the rebuilt default control. Passing operator gates still requires the same
memory, model and service qualification described above before recommendation.

The September 9 B70 follow-up in
[kernels PR #3](https://git.sunnycareboo.com/jasonbk/vllm-xpu-kernels/pulls/3)
did not qualify either policy. Across two stable starts, corrected 128x128 was
29.56–29.96% slower than oneDNN on gate/up and 63.63–84.23% slower on down;
256x128 was 89.61–90.56% and 158.43–187.44% slower, respectively. The larger
tile spills registers. An explicitly disqualified, original-rounding diagnostic
at M=2048 found no useful tile-only gain either. Cached rows save 0.97–3.11%
against uncached native calls, but cached gate/up's 1.18–2.23% advantage over
oneDNN remains below the predeclared 3% threshold. Preserve the current serving
dispatch. Reports and immutable evidence identities are in
[workflow PR #15](https://git.sunnycareboo.com/jasonbk/vllm-xpu-nix/pulls/15).

## Trace analysis

```bash
/tmp/kvarn-profile-env/bin/python -m scripts.kvarn_xpu_trace \
  --trace /path/to/captured.pt.trace.json.gz \
  --output benchmark-results/profile-new/attribution.json

/tmp/kvarn-profile-env/bin/python -m scripts.kvarn_xpu_trace_stacks \
  --trace /path/to/captured.pt.trace.json.gz \
  --event zeEventHostSynchronize --limit 4 \
  --output benchmark-results/profile-new/callers.json
```

Use correlation-based attribution, not CPU timestamp buckets, for GPU ownership.
Do not add synchronization waits to GPU duration as independent costs. Keep
unresolved and out-of-scope events explicit. With captured guard steps, select
the interior using `--first-step`/`--last-step` (one-based within the trace).

`kvarn_xpu_profile_compare` validates matched captures in the historical
`run.json`/`profile-summary.json` format; it does not consume the vision/MTP runner's
manifest directly. `kvarn_xpu_profile_overhead` provides bracket analysis for
the benchmark-result format. Neither converts profiled timing into service
performance evidence. Use unprofiled runs to confirm an improvement.

## Hardware counters

```bash
nix build --impure --file nix/xpu-metrics-probe.nix \
  --out-link /tmp/kvarn-metrics-probe
nix build --impure --expr 'let pkgs = (builtins.getFlake (toString ./.)).inputs.nixpkgs.legacyPackages.${builtins.currentSystem}; in pkgs.callPackage ./nix/unitrace.nix {}' \
  --out-link /tmp/kvarn-unitrace

/tmp/kvarn-profile-env/bin/python -m scripts.kvarn_xpu_metrics_probe \
  --probe-binary /tmp/kvarn-metrics-probe/bin/metrics-probe \
  --output benchmark-results/counters-new/definitions.json

/tmp/kvarn-profile-env/bin/python -m scripts.kvarn_xpu_counter_run \
  --unitrace /tmp/kvarn-unitrace/bin/unitrace \
  --workload gemm --iterations 1000 \
  --output-dir benchmark-results/counters-new/gemm

/tmp/kvarn-profile-env/bin/python -m scripts.kvarn_xpu_counter_run \
  --unitrace /tmp/kvarn-unitrace/bin/unitrace \
  --workload sinkhorn --iterations 100 \
  --output-dir benchmark-results/counters-new/sinkhorn

/tmp/kvarn-profile-env/bin/python -m scripts.kvarn_xpu_counter_report \
  --capture-dir benchmark-results/counters-new/sinkhorn \
  --definitions benchmark-results/counters-new/definitions.json \
  --output benchmark-results/counters-new/sinkhorn-report.json
```

Validate GEMM with the same report command. Enumeration or opening a stream is
not collection validation: require actual finite samples, metric definitions
and units, kernel coverage, and matched off/on/off workloads. Raw samples,
source snapshots, tool identities and commands are retained with each capture.
Short kernels can escape sampling; report coverage and unsupported metrics.
Stall counters alone do not prove a bottleneck, and sampled intervals are not
additive per-kernel traffic totals.
