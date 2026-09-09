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
