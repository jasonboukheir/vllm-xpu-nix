# Brutus prefill investigation — 2026-09-08

Initial read-only investigation of deployment configuration, journal history, installed
sources, repository history, and retained benchmarks, split across three agents.
No service was started or restarted and no serving cache or deployment was changed.
The follow-up below runs synthetic matmuls on the idle GPU with separate
diagnostic cache settings.

## Conclusion

There is no demonstrated regression that discards Triton disk caches between
requests. The existing JIT warnings cannot establish that: they also fire when
Triton loads a specialization from its disk cache. Actual cache-directory health
and full-model repeat-request performance remain unverified. A subsequent
microbenchmark reproduces shape-specific oneDNN setup costs, detailed below.

There are separate mechanisms for cold starts, first-seen input shapes, and
repeated full-context prefill. The historical timing rules out disabled prefix
caching as the explanation for a slowdown noticed before August 30.

## Established findings

### JIT warnings include disk-cache hits

Installed Triton XPU 3.7.2 is
`/nix/store/vfy5vd2f1xnfq9rl9azm30jhjmndwgmf-python3.12-triton-xpu-3.7.2`.
Under its `lib/python3.12/site-packages/triton` directory:

- `runtime/jit.py:901–904` calls `self.compile(...)` and then unconditionally
  invokes `jit_post_compile_hook` when adding a specialization to process memory.
- `compiler/compiler.py:275–289` returns a disk-cached `CompiledKernel` without
  compiling when cached metadata exists.
- The compiler listener reports `cache_hit=True` on that path and
  `cache_hit=False` after compilation at lines 370–372, with timing information.

Deployed vLLM's `utils/jit_monitor.py:240–277` logs through the post-compile hook.
Its default `warning_once` at line 141 also suppresses subsequent warnings for
the same kernel name. Consequently, warning presence does not prove a disk miss,
and warning absence does not prove an absence of additional specializations.

### Package changes deliberately discard accumulated caches

`nix/modules/vllm-xpu.nix:604–627` sets `HOME` and `VLLM_CACHE_ROOT` to
`<instance cacheDir>/build/<package store hash>`. Triton defaults to
`$HOME/.triton/cache`. The service's `ReadWritePaths` permits this directory, and
tmpfiles rules create it with service-user ownership.

Line 645 deletes all other build-hash directories at service startup. Any
package-hash change starts fresh, and rolling back cannot reuse a directory that
was already deleted. This behavior dates to May 27 commits `1b50c96` and
`74447dc`, introduced to prevent loading a stale Triton helper linked against an
older SYCL ABI. It explains cold starts during frequent deployments, not
per-request eviction in one process.

Installed Triton 3.7.0 and 3.7.2 `runtime/cache.py` files are byte-identical.
The module does not set `TRITON_ALWAYS_COMPILE`. Triton autotuning result
persistence is separately opt-in (`cache_results` or `TRITON_CACHE_AUTOTUNING`)
in both versions; cached kernel binaries do not imply persisted tuning results.

### Prefix reuse changed after the reported earlier slowdown

Production journal inspection found `enable_prefix_caching=True` with ordinary
KV caching through the August 25 startup. All 1,145 sampled August 28 log
entries and all 72 August 29 entries reported nonzero prefix hit rates; maximum
reported rates were 72.8% and 56.5%, respectively.

The first observed disabled-prefix startup was August 30 at 23:10:53 PDT with
KVarN. August 31 at 11:18 switched back to ordinary KV caching but kept prefix
reuse disabled. Current configuration explicitly passes
`--no-enable-prefix-caching`; current logs additionally identify an MTP/Mamba
draft-group correctness constraint on prefix reuse.

This makes repeated conversation turns prefill their supplied history again.
It can explain an additional recent slowdown, but not one predating this change.
Enabling it blindly would ignore the recorded correctness constraint.

### Long prefill exists with ordinary KV caching too

Retained September 5 measurements in
`benchmark-results/kvarn/prefill-causal-20260905/service-analysis.json` show:

| 65,023-token benchmark | Ordinary KV TTFT | KVarN TTFT |
| --- | ---: | ---: |
| Historical comparison | 51.46 s | 54.06 s |
| Finalist comparison | 51.57 s | 53.59 s |

The ordinary-KV reference repeated requests at approximately 51.3–51.6 seconds.
Thus most of that benchmark's delay existed without KVarN, whose observed
increment was approximately 2–2.6 seconds. These are retained measurements, not
a new regression test. They used eager mode, 2,048-token chunks, no prefix reuse,
text-only inputs, and no speculation; current production enables vision and
MTP. They do not establish performance against an older stack.

## Remaining candidates

**Model and execution changes need a matched comparison.** Retained production
logs begin August 11 and show dense 27B models throughout, rather than a recent
sparse-to-dense switch. The current AEON compressed-tensors model first appears
August 19 at 08:29 PDT, following several INC/AWQ/GPTQ model changes. From August
19 at 19:21 through August 29, recorded runs used ordinary KV, a 65K context
limit, `enforce_eager=False`, and prefix reuse. August 30 switched to eager mode
as well as disabling prefix reuse. A slowdown beginning before August 30 should
therefore be compared against the August 19 model/quantization transition and
August 25 stack update; the current execution settings add further differences.

**Shape-specific oneDNN primitive creation.** The deployed compressed-tensors
W4A16 route selects `XPUwNa16LinearKernel` and calls
`torch.ops._xpu_C.int4_gemm_w4a16`. The kernels source
`csrc/xpu/onednn/onednn_ext.h:797–889` caches primitives using exact M/N/K and
strides, in a 512-entry thread-local memory cache. New prompt-tail sizes can
therefore require new primitive creation even with a healthy Triton disk cache.
The mechanism predates recent changes; neither eviction nor expensive creation
has been measured in the affected requests.

**August 25 dependency update.** Packaging commit `b0f6705` changed Torch from
`2.13.0.dev20260524+xpu` to `2.13.0+xpu`, and Triton from 3.7.0 to 3.7.2, alongside
vLLM/kernel changes. This is a useful historical comparison boundary if the
reported onset matches it, not an identified culprit.

**Warmup coverage.** September 3 and September 7 vLLM changes explicitly extend
Qwen kernel warmup to avoid first-request spikes. The installed source includes
these fixes, and the current model's `qwen3_5_text` metadata matches the warmup
allowlist. Eager mode still runs kernel warmup. These fixes support investigating
missed shapes but do not demonstrate per-request cache loss.

## Limits and decisive follow-up

At inspection, `vllm-xpu-chat.service` was inactive, stopped September 8 at
08:12:22 PDT after starting September 7 at 23:40:11. Its recorded restart count
was zero. Service cache directories require additional OS access; the current
user could not read them and noninteractive sudo required a password. There was
therefore no live metrics endpoint or direct cache file/permission inspection.

The smallest useful controlled follow-up is:

1. Instrument Triton's compilation listener for actual disk hits/misses and
   compilation time, and record oneDNN primitive-creation time separately.
2. In one process, repeat identical token lengths, then vary only the final
   prefill-chunk length. Keep prefix reuse off for this comparison so it cannot
   hide the prefill cost. Record TTFT, queue time, and prefill time.
3. Restart the exact same package with its cache preserved, and repeat those
   lengths. Inspect cache file counts, timestamps, ownership, and driver-cache
   growth before and after. Do not clear caches.
4. If warm, same-shape prefill remains slow, compare old and new stacks with the
   same model, quantization, prompt lengths, and scheduler configuration.

This distinguishes missing persistence from new-shape compilation and a slower
steady-state prefill path without treating JIT warning counts as timings.

## Follow-up: direct W4A16 shape measurement

The user's strongest suspicion was the first-seen prompt-length effect. On
September 8, the stopped service left Brutus's Arc Pro B70 available for a
bounded synthetic test of the deployed `int4_gemm_w4a16` op. The probe uses
Torch 2.13.0+xpu, kernels `23d8103`, and oneDNN 3.13.0 from the deployed package.
It covers the six projection dimensions in the kernels repository's production
W4A16 determinism tests, with BF16 activations/scales, group size 128, packed NT
weights, and scalar zero point 8.

**The shape effect is reproduced.** For each shape, the probe times the first
call and three repeats, then returns to length 107 after other lengths.
Allocation and input initialization happen outside the timed region; each call
is followed by XPU synchronization. A second process repeats the same sequence
without deleting any caches. A third process samples a wider range of lengths.

| Token count and context | Extra first-use wall time summed across six projection shapes |
| --- | ---: |
| 107, first occurrence | 49–50 ms in the two matched processes |
| 108 / 109, following 107 | about 10 ms |
| 511, first occurrence | 184–191 ms |
| 513, first occurrence | 112–115 ms |
| 107, revisited after other lengths | 0.28–0.39 ms |
| 239, wider sweep | 431 ms |
| 2,048, first occurrence | 292–495 ms in the two matched processes |

These are sums of first-call minus median-repeat time for six distinct matrix
shapes, **not model TTFT**. Do not multiply them by the number of model layers:
the kernel's primitive cache key excludes layer identity and weight pointers,
so layers with matching dimensions reuse these entries.

oneDNN's `profile_create` trace directly attributes creation events. Each of the
two matched processes logged 12 GEMM `create:cache_miss` events, 42
`create:kernel_cache_hit` events, and 54 outer matmul
`create:nested_primitive_cache_hit` events. Repeating an exact shape emitted no
new creation event because the vLLM kernel wrapper's cache served it. Moving
from 107 to 108/109 often reused oneDNN's underlying generated kernel even
though a new primitive was needed. Thus a new exact dimension does not always
mean a new kernel compilation. See the
[oneDNN verbose documentation](https://uxlfoundation.github.io/oneDNN/dev_guide_verbose.html)
for the creation statuses and timing fields.

Across 32 distinct tested token counts and three completed processes, the
largest summed first-use penalty was approximately 0.5 seconds. This establishes
a real contributor to prompt-length-dependent latency, but does not reproduce
multi-second stalls in the W4A16 matmuls alone. The sweep is not exhaustive,
does not exercise long-run cache eviction, and does not measure attention,
GDN, vision, scheduling, or other model operations. The exact first-use order
also affects lower-level kernel reuse. No full-model request or production
cache read was performed.

The initial standalone attempt failed at MLP-down M=2,048 because OpenCL vendor
discovery was absent. The completed runs explicitly set `OCL_ICD_VENDORS` to
the package's pinned Intel driver. That diagnostic setup failure is retained
and excluded from the result tables; it is not evidence of a serving failure.

Reproduction and retained evidence:

- Probe: `scripts/xpu_w4a16_shape_probe.py` (ruff checks and format pass).
- Results: `benchmark-results/prefill-shape-20260908/{process1,process2,length-sweep}.json`.
- oneDNN traces: adjacent `.log` files.
- Aggregates: `benchmark-results/prefill-shape-20260908/summary.json`.
- Package/environment provenance, a copy of the probe, and a Python launcher
  derived from the deployed package wrapper are retained in the same directory.

Example, from the repository root with the recorded environment applied:

```sh
bash benchmark-results/prefill-shape-20260908/python-launcher.sh \
  scripts/xpu_w4a16_shape_probe.py --output /tmp/w4a16-shape-results.json
```

The next useful measurement is full-model first-versus-repeat prefill at fixed
lengths, attributing non-W4A16 work as well. Bucketing or prewarming shapes is a
potential mitigation only after identifying which creation costs dominate;
changing dimensions can also change quantized numerical behavior and needs
correctness validation.

## Cache-only follow-up (no padding)

The user ruled out padding and asked whether caching itself was defective.
An additional probe exercised the wrapper's 512-entry LRU using the small
GDN-BA projection, at exact lengths 1 through 512, then 1, 513, 2, 1.

- Revisiting length 1 at capacity reused it without a oneDNN creation event.
- Inserting 513 evicted the least-recently-used entry, length 2, as expected.
- Revisiting 2 then reached oneDNN's own primitive cache, which reported
  `create:cache_hit`; its extra first-call time was approximately 0.047 ms.
- The final revisit of 1 still hit the wrapper cache.

This controlled test found normal LRU behavior, not a cache reset or broken
lookup. It does not rule out production-specific thread changes, restarts, or
long-run pressure on lower-level caches. Raw data is in
`benchmark-results/prefill-shape-20260908/eviction.{json,log}`. No padding or
serving/kernel change was made.

## AOT scope

The deployed native components already build with both
`VLLM_XPU_AOT_DEVICES="bmg"` and `VLLM_XPU_XE2_AOT_DEVICES="bmg"`, verified in:

- `/nix/store/h7xj5iqsrbd11snlcfhxs8p6c6lrgkac-python3.12-vllm-xpu-kernels-base-glue-0.1.14.1+src.9jjgx9lb29bsc7s5rsm5zyqnjgvhycv6.drv`
- `/nix/store/ac093v4k5d0hhja003v2z4swnlpx7yry-python3.12-vllm-xpu-kernels-fa2-binding-0.1.14.1+src.ri57zj3pb7rcss30rnjhr3xm8lvzl5b9.drv`

These are dependencies of the exact deployed kernel package. This setting
compiles the packaged SYCL kernels for the target device; it does not
preinstantiate oneDNN's runtime-generated, shape-dependent GEMMs or Triton
specializations. The observed oneDNN creation delays therefore occurred with
the package's native AOT setting already enabled. Prewarming unchanged shapes
and retaining compatible generated kernels are separate possibilities from
this AOT switch; no warmed-throughput gain from changing AOT has been measured.
