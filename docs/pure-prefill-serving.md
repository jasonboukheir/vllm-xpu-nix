# Pure-prefill serving integration

The vLLM fix forwards the existing XPU `is_mix_batch` flag and selects the
pure-prefill route for a CPU-validated, unpadded single request. Actual service
traces confirm that every selected attention call drops three redundant
launches and retains the identical prefill kernel. This is the serving follow-up
to the [operator experiment](d256-barriers-dispatch-experiment.md), authorized
separately as a small overhead cleanup. Native kernel code is unchanged.

The implementation is reviewable as a bounded cleanup, not a demonstrated
service speedup. Measurements retain a possible small 65K cost, and ordinary-KV
MTP2 exact qualification remains failed with inconclusive attribution because
the unchanged control also varies across starts. Complete serving qualification
and performance non-regression are not claimed.

The [prospective plan](../fixtures/xpu-pure-prefill-serving-plan.md) uses the
[profiling workflow](kvarn-profiling.md#qualify-a-python-dispatch-change).
Evidence is local on Brutus under
`benchmark-results/pure-prefill-serving-20260909/`.
The complete evidence archive is
`benchmark-results/pure-prefill-serving-20260909-evidence.tar.zst`
(164,852,727 bytes, 996 checksummed files), SHA256
`cdea88f4b5c9d18c24aa780ed8f3c0a7b73b4763cc37630fcd4ced0257209121`.
Its `SHA256SUMS` was verified before and after archiving; that manifest's SHA256
is `8d3ee56a88a983f6b342a95b8aee2ebbb0476e5636c31d66d19347696601deb1`.

## Implementation and scope

Control is fork main `6700b7cad87b884a49ce2e5e6c33c86f1b5529b3`;
candidate is `b29cd13ef9c904155f36476ef8829cf417869a2f`. The selected path
requires query length greater than 16, one physical request row, matching
unpadded query extents, CPU prefill-state metadata and an existing CPU sequence
length known to be exact for prefills. It does not synchronize device metadata.
Missing or inconsistent evidence retains default dispatch.

The implementation is published in [vLLM PR #2](https://git.sunnycareboo.com/jasonbk/vllm/pulls/2)
on `fix/xpu-pure-prefill-dispatch`, marked WIP for the unresolved qualification
and performance limits. [Workflow PR #16](https://git.sunnycareboo.com/jasonbk/vllm-xpu-nix/pulls/16)
contains the reproduction tools and experimental results.

Selection is restricted to causal D256 attention with FP16/BF16 KV. Cascade,
DCP, graph capture, draft builds, sliding windows, sinks, ALiBi, softcap and
custom masks retain default routing. Draft metadata reuse clears the hint even
when no scheduler metadata exists. CUDA/ROCm call signatures remain unchanged.
Compact KVarN's separate dense attention path is not redirected.

The source commit includes 39 targeted tests in existing attention and worker
suites: metadata/padding checks, capture/draft invalidation, native wrapper
forwarding and ten final-forward feature routes. All pass. Applicable pinned
pre-commit hooks, including Ruff 0.14.0, typos and mypy, pass. Private generic
Ruff/typos executables needed NixOS ELF-loader repair; repository hook settings
were not changed. The inapplicable Markdown hook was skipped, and the Docker
graph hook was invoked with Bash because this host has no `/bin/bash`.

## Native and service correctness

Eight real wrapper cases pass both dispatch flags: all six captured paged BF16
extents and FP16 casts at Q/K=2048/2048 and 1535/65023. Paired output is bit
identical; independent FP32 references, dirty repeats, output ownership,
canaries and input/metadata integrity pass at the existing tolerances. FP16
casts change tiny operand values, so their references were regenerated rather
than reusing BF16 references. The failed reference-reuse preparation remains
retained alongside the completed gates.

This narrow validation DSO has no native LSE prefill templates. Its existing
reference fallback gives identical output/LSE with either flag, and LSE matches
the independent reference. Four output values in the short BF16 fallback case
miss the existing 0.02 + 1% tolerance in both arms (maximum absolute error
0.032739). That baseline failure remains recorded; no tolerance was relaxed.
The selected ordinary serving call requests no LSE. These results do not claim
that the fallback's output-accuracy gate passes.

Separate paired 16383-token service diagnostics establish:

- All 128 prefill calls change from four native launches to one: 384 launches
  removed, with identical prefill specializations and input layouts.
- All 32 calls at the two decode boundary steps retain both decode and
  `ReduceSplitK` kernels.
- Every device origin resolves through CPU correlation; unrelated device work
  matches. All four requests, including the off/on/off bracket and warmup,
  have identical prompt and output token IDs.

Profiled time is diagnostic only. It is excluded from performance results.

## Profiler-off service measurements

All eight fresh services complete the planned B1/eager/MTP0 workload at budget
2048, with prefix caching disabled: one warmup and three measured requests,
512 output tokens each, at both lengths and in reversed arm order. All 32
requests have exact paired prompt/output IDs, content, usage and finish reason.
The source-aware audit passes with zero preemptions, fatal engine findings or
memory-sampling errors.

These are medians of three measured requests per service. Positive TTFT delta
means the candidate is slower; these tiny ratios are descriptive.

| Input tokens / start | Control TTFT (s) | Candidate TTFT (s) | TTFT delta | Control / candidate total (s) | Control / candidate decode (tokens/s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 16383 / 1 | 10.55833 | 10.55692 | -0.0133% | 27.45526 / 27.45407 | 30.2555 / 30.2420 |
| 16383 / 2 | 10.56762 | 10.57044 | +0.0267% | 27.47013 / 27.46597 | 30.2326 / 30.2451 |
| 65023 / 1 | 50.38918 | 50.39292 | +0.0074% | 70.07527 / 70.07557 | 25.9429 / 25.9450 |
| 65023 / 2 | 50.38386 | 50.40761 | +0.0472% | 70.07990 / 70.09551 | 25.9468 / 25.9439 |

No end-to-end speedup is demonstrated. The 16K TTFT change switches sign
between starts. At 65K all six request-position paired TTFT deltas are positive
despite reversing service order: 3.74–23.76 ms, with per-start paired-delta
medians of 17.28/19.33 ms. This is a weak but consistent adverse signal;
a small real 65K cost cannot be ruled out. It must not be described as proof
of no regression simply because the percentages are small. Request-position
pairs are separate service executions, not randomized same-process trials.

Decode-rate changes also switch sign (within 0.045%). Client median p95/p99
token gaps are about 33.3/33.8 ms at 16K and 38.8/39.1–39.2 ms at 65K; all
raw distributions are retained. Sampled peak DRM allocation/residency is about
29.588 GB in each arm, with no material growth. These are 0.5-second samples,
not instantaneous allocator high-water marks. The change removes measured
native overhead, but it is not recommended as a demonstrated service speedup.

## MTP2 qualification and baseline instability

The paired compact KVarN MTP2 service passes all 16 requests, with exact
generated tokens, target logprobs and speculative counters. Ordinary KV fails
the strict cross-service gate: 6/16 generated sequences and 13/16 target-score
streams differ. All eight semantic correctness fixtures still pass, but that
does not clear the exact gate. Score differences begin at the first token,
before later speculative verification, and cannot be dismissed as ULP noise.

A prospectively recorded diagnostic repeats the unchanged ordinary-KV control
in a fresh service. It also differs from its original run: 6/16 output
sequences and 14/16 score streams change. Short text varies within both source
versions; repeated image requests are stable within each service but differ
between fresh control services. This establishes baseline MTP2 instability on
the actual fixtures. It does not turn the original failed qualification into
a pass or establish equivalence for every supported workload.

Additional native gates cover Q=17/25/31/64/222/231/255/256, both initial
K=Q and cached K=65023. All 16 cases pass unchanged reference tolerances and
exact paired/repeated outputs. All 192 restored-output checks also pass when
Q and real KV pages are mutated/restored on device and attention is queued
without an intervening host synchronization. No hint-lifecycle or native
short-query defect was found in the skeptical source review. The original
failure, all follow-up diagnostics and their limits remain retained.

An instrumented service follow-up replays actual first-prefill operands from
two image fixtures and one text fixture, Q=231/222/25. All 48 target-layer
calls produce byte-identical outputs with alternate default dispatch and a
repeated selected call. Full logical input bytes remain unchanged across the
replays. The original selected native call runs before any diagnostic clone,
hash or synchronization; the input-integrity baseline is captured after that
original call. Observed vision, draft and target-verification calls retain
default dispatch.

The diagnostic records its hook sources, direct immutable-package launcher
and actual worker identity. Earlier attempts failed to install the hook and
have zero replay coverage; they remain recorded separately. These instrumented
results narrow the native flag hypothesis on the actual fixtures. They do not
clear the uninstrumented cross-service MTP2 gate, qualify arbitrary stream
interactions or provide performance measurements.

## Runtime and reproduction

Both arms use the pinned Qwen27B BF16/W4A16/G128 checkpoint revision
`6b0622f4354481d5d04577d48ba0db844efc1330`, B70, Torch 2.13.0+xpu and oneAPI
2026.0.1.27. The Nix builder changes only vLLM Python source. Complete native
process dependencies match; the only closure differences are vLLM itself and
the two outer Python/service environments. Both package version labels inherit
`9bfcf12`; the actual source revisions above and installed-file hashes provide
the source identity. Installed candidate/control files match their commits
byte for byte.

The attention DSO is
`/nix/store/ih92ljdk363rljg87jxavdwn1pmazs0x-vllm-xpu-attn-kernels-xe-2-0.1.14.1+src.dqw1a0l8w7c2xazw03bip6yfyqq25bhc/lib/libattn_kernels_xe_2.so`,
SHA256 `7b94d7573ead6773cb966c85c354cc097fb68ec71cc0a8d128e130563453cd92`.
This is the validation build of kernels `23d8103`, not the historical full
profile package. Its actual native wrapper and service routes were regated.

Use `run-services.py` and the recorded stage command files in the evidence
archive to reproduce the immutable environments and exact requests. Source
snapshots, the vLLM format-patch, native gate references, failed preparations,
traces, raw streams and skeptical reviews are retained. Original BF16 captures
and references are dependencies in the separately checksummed issue11 archive;
they were not modified. Nothing was deployed or released. AI assistance:
Codex and three user-requested skeptical reviewers.
