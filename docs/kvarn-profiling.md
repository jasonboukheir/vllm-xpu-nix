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
