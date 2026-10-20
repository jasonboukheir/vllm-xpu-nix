# V2, DFlash2 and idle latency: source and measurement audit

Read-only investigation on Brutus, 2026-10-19 PDT / 2026-10-20 UTC. No
model downloads, GPU jobs, requests, service changes, source changes or host
configuration changes. This report records implementation evidence and proposed
measurements; it does not qualify a runtime or claim a speedup. Journal evidence
below contains only timing, process and cache metadata.

## Decision and frozen inputs

Proceed toward one `xpu-v1.13.0` with vLLM 0.32, V2, DFlash2, the existing
compact K4V2 target and RAM prefix cache. Preserve `xpu-v1.12.2` as rollback.
The primary unknown is the complete local cache integration, not the existence
of a V2 class upstream. Treat the first-request-after-idle delay as a separate
measured regression target; faster steady-state decoding need not fix it.

- vLLM frozen upstream: `5df63a7203cdb9328d824a27b682e77c7b9928be` (`v0.32.0`).
- Existing vLLM fork: `dcba4007ef6af8ba143d2794c194c36b58397339`.
- Released compatible kernel base:
  `ddf336d86e3c8602888572a3502f951abd51df12` (`0.1.15.4`); fork
  `dbb684f4c9b38a8b3bbf52bda0c490ee47049221`.
- Actual profile:
  `/home/jasonbk/.config/nix/hosts/brutus/services/vllm-xpu/models/RedHatAI__Qwen3.8-27B-INT4.nix`.
  Target revision `bf08f3dbd9a324e53956920aad378a1f1b6dd24a`, BF16 compute,
  compressed-tensors INT4, `kvarn_k4v2_g128_compact`, 262144 combined input/output
  ceiling, four scheduler slots, 2048 batched tokens, memory utilization 0.96,
  eager V1, bundled K4V2 MTP2, two images at 200704 pixels each, video disabled.
  Prefix caching is enabled with `IndependentCPUOffloadingSpec`, 8 GiB host
  budget, `gpu_cache_policy=active_only`, `offload_prompt_only=false`.
- The October 10 performance review predates RAM prefix caching. Its
  no-prefix-cache profile and suggested smaller-context/BF16-target prototype
  are not this release's acceptance profile.
- Paused issue #17 is useful engineering evidence, not an automatic reinstatement
  of every historical constraint. Current scope permits selecting a compatible
  drafter and independently choosing its cache format while preserving target
  context, RAM caching, vision and supported concurrency. In particular, a
  bounded BF16 **draft** cache is worth measuring; it does not require replacing
  the K4V2 **target** cache or reducing its 262K context.

## What the exact 0.32 source establishes

All vLLM paths in this section are relative to `/home/jasonbk/Projects/vllm`
at the frozen upstream commit, not a claim about a subsequently edited checkout.

| Component | Evidence and practical consequence |
| --- | --- |
| V2 on XPU | `vllm/v1/worker/xpu_worker.py:217` selects `XPUModelRunnerV2`; `xpu_model_runner.py:29` wraps the shared GPU V2 runner with XPU stream/event/graph interfaces. V1 is still present and selectable in this release; the earlier planned V1 removal is not a fact about this frozen tree. |
| Hybrid target and vision | `vllm/v1/worker/gpu/model_states/__init__.py:52` selects `MambaHybridModelState`; shared runner/model-state code supplies recurrent-state and encoder handling. These paths require testing with native XPU GDN, speculation, prefix hits and real images. |
| DFlash2 selection | `vllm/v1/worker/gpu/spec_decode/__init__.py:34` selects `DFlash2Speculator` by `DFlash2DraftModel`. `vllm/config/vllm.py:_get_v1_model_runner_unsupported_features` rejects candidate-head drafts on V1. Set V2 explicitly and verify the selected candidate-head path. |
| Quantized context projection | `vllm/model_executor/models/qwen3_dflash.py:523` recognizes `UnquantizedLinearMethod`; otherwise `:620` calls each quantized QKV module and extracts K/V results. The September report's unconditional packed-weight slicing blocker is superseded. Keep this working path before considering a fused dequantized context buffer as a measured optimization. |
| Draft quantization mapping | `vllm/model_executor/models/utils.py:985` still returns a fresh draft quant config without `configure_quant_config`. `qwen3_dflash.py:446` uses it and adds layer-exclusion aliases. Verify actual fused-module schemes/weights after load, especially QKV and gate/up mappings and ignored conv/selector modules. Do not equate passing the quantization config with verified INT4 dispatch. |
| Native INT4 route | `vllm/model_executor/kernels/linear/mixed_precision/xpu.py:XPUwNa16LinearKernel` exists for supported packed types. Log the realized kernel/quant method per relevant draft projection and compare its output with a small numerical reference. |
| Context K normalization | Commit `754aa29c00` (#58896) uses stacked `ops.rms_norm` in `qwen3_dflash.py:639`. The retained kernel base's `csrc/layernorm.cpp:1263` accepts 2D per-outer-layer weights. Do not restore the older per-layer XPU loop without evidence; test distinct layer weights against the realized native library. |
| Draft convolution | `qwen3_dflash2.py:153` enables the fused grouped convolution only for `hidden_states.is_cuda`; XPU takes the PyTorch fallback at `:156`. This is a profiling candidate, not an established bottleneck or promised gain. |
| Eager and compilation | Existing eager operation is a viable initial execution choice. The XPU compile issue [#56787](https://github.com/vllm-project/vllm/issues/56787) documents a draft fake-kernel stride failure, but is not proof that this exact source still fails. Qualify execution modes explicitly; graphs are not needed to establish eager correctness. |
| New DFlash metadata shortcut | Commit `36f3972812` (#60692), `spec_decode/dflash/speculator.py:499`, skips full metadata building only for full graphs with backend support. A stateful KVarN builder must not advertise this capability until reserve/flush/commit work is preserved. Eager mode still builds metadata. |

[Upstream PR #53122](https://github.com/vllm-project/vllm/pull/53122) was still
open when inspected. Its description and September patch are not authoritative
for this release's actual context-projection implementation. The local frozen
source is the basis of the distinctions above. The pinned Hugging Face config
URL was not accessible through the web tool; checkpoint facts below are from
preserved local metadata and must be reverified against downloaded bytes before
qualification.

## Drafter choice and memory

The strongest existing starting candidate is
`syvai/Qwen3.8-27B-DFlash2-W4A16` at
`4d30ec736ffc6b8688dc2ae2b502d9b48bdec279`. Preserved identities are in
`build-dev/issue17-packaging-checkpoint/docs/updates/issue17-20260920-snapshot-audit/draft-identities.json`;
the actual config inspected is
`benchmark-results/issue17-20260920/syvai-config.json`.

It specifies five layers, hidden size 5120, vocabulary 248320, 32 query heads,
eight KV heads, head dimension 128, noncausal 2048-token sliding attention,
target layers `[5,19,33,47,61]`, candidate selector rank 256/top-k 16, and
block size eight (seven proposals plus anchor). INT4 weights are symmetric
G128 compressed-tensors; conv kernel projections and candidate/hidden selector
projections are excluded. Geometry matches the intended target family, but
acceptance with the exact RedHat INT4 target is unmeasured. The recorded
`model.safetensors` size is 1,280,633,960 bytes; this is checkpoint storage,
not resident GPU memory.

For this geometry, the raw BF16 draft KV data for a strictly bounded 2048-token
window is `5 * 2048 * 8 * 128 * 2(K/V) * 2(bytes) = 40 MiB` per request, or
160 MiB for four requests. This arithmetic excludes alignment, lookahead,
null blocks, temporary projection buffers, allocator grouping and workspace.
The real allocation can be larger and must be verified, especially under the
fork's pool-sizing code. It is nevertheless small enough to assess before
assuming compressed draft KV is required. A separate compatible noncausal
sliding backend must be chosen for BF16 draft attention; do not inherit the
target's KVarN backend accidentally. Verify target/draft backend selection,
cache dtype, block tables and mutation ownership separately.

If K4V2 draft KV yields a meaningful capacity or speed benefit, extend it
explicitly to D128/Q32/KV8 and noncausal sliding attention. The existing native
compact path is specialized for target/MTP D256 geometry. A 16-block BF16
resident-history window at block size 128 can retain the entire 2048-token
draft window uncompressed; a dtype label alone would then overstate compression.
Charge bounded draft-window pools for **each admitted slot**, independent of
the shared target token pool. Preserve null pages and lookahead.

The preserved alternatives are `0xMiami305/Qwen3.8-27B-DFlash2-W4A16-MLP-only`
at `6d2937789337b005a3a74a3f5991ac5896d85192` and
`gratex/Qwen3.8-27B-DFlash2-W4A16-g128-sym-GPTQ` at
`5564be8081de421f9d106087b82c33046ca22054`. Use them only to answer an actual
loading, acceptance or performance question; do not enlarge the qualification
matrix indiscriminately. An unquantized Inco draft is another compatibility
reference, but its larger weights require a measured memory/capacity check.

Keep serving recipes immutable by pinning target and draft revisions. Packaging
currently roots only the target in HF cache management
(`nix/modules/vllm-xpu.nix:840`); add the chosen external draft revision to the
root manifest and provenance before release. Record actual sharing of target
embedding/output-head storage and persistent/transient allocations rather than
estimating loaded memory from checkpoint size.

## Required KVarN/V2 integration

Existing fork guards in `vllm/platforms/xpu.py:67` and
`vllm/config/vllm.py:_get_v2_model_runner_unsupported_features` still reject
V2 KVarN and non-MTP speculation. Removing those guards alone is insufficient.
The per-commit audit owners should preserve these contracts when porting:

1. **Exact history and ownership.** Keep exact accepted/committed token lengths
   separate from V2 `seq_lens_cpu_upper_bound`. Give target and draft metadata
   builders independent mutable tails, reservations and pending flush state.
   Preserve MTP prefill metadata ownership and GDN rollback after rejection.
2. **Allocation and reset.** Port compact byte-record allocation, per-group
   attention/recurrent pools and zeroing to shared V2 initialization. Include
   draft groups in physical-cache registration/zeroing. Distinguish replacing
   a cache from destroying a model: invalidate cache references and allocator
   shadows without losing implementation registration.
3. **DFlash insertion ordering.** Frozen
   `vllm/v1/worker/gpu/spec_decode/dflash/speculator.py:493` precomputes and
   stores context KV before `:500` builds normal draft metadata. Compressed
   draft support needs explicit reserve → insert committed context → materialize
   queries → commit/reclaim phases; moving the old builder earlier does not
   establish this order. Preserve rejection masks and null block zero.
4. **RAM prefix restore.** V2 has generic
   `vllm/v1/worker/gpu/kv_connector.py:ActiveKVConnector`, but it does not
   automatically carry the local compact-attention plus recurrent-state
   serialization, cache-reference repair and MTP boundary fixes. Verify restored
   target history plus new draft reconstruction; draft context states must be
   available after target prefix skips. A cache hit cannot silently skip
   auxiliary hidden states needed by DFlash.
5. **Window semantics and capability flags.** Preserve noncausal bidirectional
   queries, 2048-token window eviction, partial pages, dummy runs, cancellation,
   slot reuse and concurrency. Keep graphs/adaptive verification disabled until
   their lifecycle assumptions are separately tested.

Useful existing tests in frozen upstream include
`tests/v1/spec_decode/test_dflash2.py`, `test_dflash_prepare_inputs.py`,
`test_dflash_causality.py`, `test_dflash_lookahead.py`, and
`tests/v1/attention/test_gdn_zero_draft_spec_rows.py`. They provide focused
coverage to extend, not end-to-end proof of the custom stack.

## First request after idle: evidence

The journal does contain a matching idle interval, without a process restart:

- API PID 9730 and engine PID 12210 started on October 17. Weight loading took
  about 30.73 seconds; HTTP application startup completed at 14:02:22 PDT.
- Serving activity ended around October 18 12:23 PDT. The next recorded chat
  HTTP response header was October 19 19:43:11 PDT, approximately 31 hours later,
  using the same processes. Continuation/flush logs appear at 19:43:25.
  This is compatible with the reported delay but is **not a TTFT measurement**:
  Uvicorn's HTTP 200 can precede body tokens, and sparse kernel logs do not mark
  every engine phase.
- No intervening model-loading/startup records appeared. The next stop was the
  user's maintenance stop at October 19 20:40:57. These observations weaken a
  model-reload explanation for this interval; they do not identify the cause.
- A subsequent transfer sample at 19:43:38 reported 267,182,114 bytes loaded
  from host cache in 0.0673 seconds. That recorded transfer alone does not
  explain a 10–15-second stall; lookup, allocation, device residency and other
  work are separate. Do not extrapolate one sample to every request.
- Actual systemd unit and `nix/modules/vllm-xpu.nix:644` use persistent
  `HOME`/`VLLM_CACHE_ROOT` under
  `/var/lib/vllm-xpu/chat/build/14ww78z5v8m0ps2qcpc8v19y83546mqf`.
  `ExecStartPre` removes **other package namespaces**, not the current one.
  There is no daily JIT-cache expiry in this module and no idle model-unload
  option in the selected profile. This is not proof against driver eviction
  or an external cleanup mechanism.
- The active eager setting disables torch.compile, graphs and JIT warmup.
  Runtime Triton/driver compilation is still possible. In frozen 0.32,
  `vllm/v1/worker/gpu_worker.py:1094` deliberately skips the JIT monitor when
  warmup is disabled; absence of JIT warnings is therefore not evidence that
  compilation did not occur. Simply turning on verbose monitor settings may
  not instrument this profile.
- Open WebUI routes through Bifrost. Its module is
  `/home/jasonbk/.config/nix/modules/homelab/services/open-webui/default.nix`.
  Nginx buffering/cache are disabled; model-list cache TTL is 300 seconds;
  search/retrieval query generation is enabled; titles remain enabled; other
  automatic tags/completion/follow-up tasks are disabled; default reasoning
  effort is low. These facts motivate separating visible-answer latency from
  first reasoning token and gateway/background-task time.
  `.../services/bifrost.nix:97` sets zero provider retries, 600-second request
  timeout and 300-second stream-idle timeout. Those timeouts are limits, not
  evidence that they cause the observed stall.

Read-only systemctl bus queries were sandbox-denied; unit-file reads and
`journalctl` succeeded. No privileged retry or service action was performed.
No matching OOM/GPU-reset record was found in the inspected October 17–19 kernel
journal, but no historical page-fault, swap, device-power or residency trace
exists to exclude those causes.

## Bounded instrumentation and benchmark plan

Do not add a keepalive to hide the symptom before diagnosing it. Do not drop
system caches or restart production to imitate a long-idle case. A restarted
process, a new package compilation namespace and an unchanged process after
31 hours are different conditions.

| Comparison | Controlled conditions and evidence |
| --- | --- |
| Request path | Identical short text request directly to loopback vLLM, through Bifrost, and through Open WebUI. Start with a fresh conversation and search/tools off, then repeat the normal UI path. Preserve model/reasoning/sampling settings and actual prompt-token counts. Record request ID and monotonic times at browser, gateway, API receipt, scheduler admission and first output. |
| Warm versus idle | Same process and package; immediate repeat, 5-minute idle, 30-minute idle, then a genuine overnight/day interval. Record PID/start time, actual previous inference time and competing activity. Avoid GPU telemetry probes or warmups immediately before the idle request if they could wake the device. Alternate which path receives the first request across separate idle intervals. |
| Process/cache coldness | Separately label fresh process with existing persistent caches and fresh package cache namespace. Retain readiness time and first successful generation separately. Both are separate from same-process idle. |
| Prefix state | Fresh small prompt, exact repeat, and small continuation of a cached conversation. Capture real prompt size, cache hit count, host lookup/load/store time and bytes, reconstruction/materialization time, GDN restore time and queue time. UI message size is not total tokenized prompt size. |
| Visible versus model output | Timestamp first response headers, first nonempty SSE event, first reasoning token, first user-visible content and completion. Header TTFB and browser spinner time are not interchangeable with model TTFT. Speculation bundles multiple tokens, so SSE-event gaps are not per-token latency. |
| Faults/device/JIT | During the request collect process RSS/swap/major faults, cgroup memory events/pressure, system swap-in/IO, GPU residency/power/frequency/reset counters, and compiler/cache-write events where available. Attribute host-cache restore and XPU memory migration separately. Validate collector overhead; do not enable an unsupported profiler and infer a limit from missing traces. |

First falsifying experiment: replay the same short, already-warmed prompt after
an untouched-process idle interval with first-token timestamps and process/fault
metadata, then repeat immediately. If direct API is fast while the UI is slow,
investigate gateway/task/rendering phases. If the direct API is slow, distinguish
queue/prefill/restore from a pre-kernel gap, runtime compilation, page migration
and device wake-up. Test only the mechanism supported by that evidence. Existing
logs alone do not select one.

For the release performance comparison, begin with these four arms on the same
target revision, supported image budget, target K4V2/RAM-cache policy, context
ceiling, scheduler slots and GPU-memory budget:

1. Released V1 + bundled MTP2: user-experience regression control.
2. Candidate V2 without speculation: runner/cache correctness and cost control.
3. Candidate V2 + bundled MTP: primary same-stack comparator; retain MTP2 and
   screen one proposal if already supported rather than weakening the baseline.
4. Candidate V2 + selected DFlash2: initially seven proposals as specified by
   the checkpoint; independently record the draft weight/cache formats and
   actual selected backend. Screen another supported proposal count only after
   correctness and a profile identify useful scope.

Use fixed prompts/seeds/sampling/output budgets, balanced arm order, at least
three measured repetitions and two fresh starts. Separate warmups. Primary
interactive anchors: C1 and C2, 4K/16K prompts, at least 512 emitted output
tokens, with vision support enabled. Add shorter prompts for TTFT, a decode
overlapping long prefill, ~65K/128K and near-limit input-plus-output checks,
and supported C4 stress. Include two-image conversations, old images beyond
the resident sink/window, tools, cancellation/reuse, RAM prefix restore,
accept/reject extremes, ragged scheduling and context-limit boundaries.

Report per-request TTFT, verified emitted-token decode rate, completed latency,
aggregate throughput, accepted/proposed tokens by position, verification-round
cost, preemptions and memory. Exclude prefill and the entire first bundled
stream chunk consistently from decode windows. Retain trajectories: different
speculative paths can change work even with fixed seeds. Do not call lossy
K4V2 mathematically lossless; combine matched-history numerical checks with
bounded output/quality evaluations.

Reuse and validate `scripts/kvarn_xpu_profile_smoke.py`,
`kvarn_xpu_profile_overhead.py`, `kvarn_xpu_profile.py`,
`kvarn_xpu_profile_compare.py` and `kvarn_provenance.py` before tuning. Obtain
correlated CPU/XPU phase attribution for target verification, quantized draft
QKV/context projection, grouped conv, selector/top-k, metadata synchronization,
cache materialization and transfer. Measured profiler overhead is separate from
profiler-off speed results. Candidate opportunities include per-layer context
projection work, XPU grouped-conv fallback, exact-history synchronization and
D128 attention; none is yet proven to dominate end-to-end time.

Finally, compare MTP and DFlash2 at the same VRAM budget: usable shared **target**
KV tokens after both models warm, exercised per-request combined context,
exercised concurrency, peak/resident memory and draft-window allocation. Never
add draft tokens to reported target context capacity or confuse four scheduler
slots with four simultaneous 262K conversations. A useful repeatable latency
gain and preserved supported capacity/quality are the release evidence; neither
acceptance rate alone nor an external GPU's tokens/second is a result for Brutus.

## Observed baseline screen and first idle probes

Read-only follow-up reviewed the parent's isolated stable-baseline run on
`127.0.0.1:18000`; this reviewer issued no inference requests or process actions.
Evidence is `../baseline-performance.json`, `../baseline-idle-probes.json`, and
raw artifacts under
`benchmark-results/release-v032-v2-20261019/baseline-{01,performance}/` plus
`idle-probes.json`. Both committed-summary candidates match their raw copies.
The manifest identifies the released `gdcba400` package, actual host serving
arguments, private persistent runtime-cache namespace and offline cached model.
These override details matter for cold-start comparisons.

The short HTTP screen is internally consistent. Reconstructing all 16 request
results from raw SSE reproduces token identities/chunk timings, prompt counts
and decode rates; harness, workload and server-manifest hashes match. Twelve
measured requests each emitted exactly 512 tokens with a length stop. Each
measured concurrency phase accounts for exactly six successful requests and
3072 generation tokens, with no counter resets or preemptions. C2 batches have
14.74–15.79 seconds of overlapping token delivery. The decode-rate formula
excludes the complete first chunk, and speculation counters are correctly
labelled phase-wide.

| Workload | Input tokens | C1 median TTFT | C1 median decode tok/s | C2 median TTFT | C2 median decode tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| English | 73 | 0.286 s | 29.55 | 0.465 s | 28.67 |
| Arithmetic explanation | 133 | 0.307 s | 35.26 | 0.394 s | 33.78 |

There are three measured repetitions per workload/concurrency, all within one
server start. With this sample size the reported nearest-rank p95 is simply
the maximum, not a stable tail estimate. C1/C2 accepted-to-proposed ratios are
69.23%/68.92%; these do not independently demonstrate performance. C1 warmup
TTFT was 2.78–4.90 seconds and C2 warmup TTFT 3.60–3.75 seconds. This establishes
a warmup/measurement difference without identifying compilation or any other
cause. The workload named `reasoning` is an arithmetic prompt with thinking
disabled, not a reasoning-mode benchmark.

Important scope limits: these are 73/133-token prompts, not the planned 4K/16K
anchors. Raw Prometheus deltas show **zero GPU and external prefix-cache hits**
in both measured phases, with all 618 input tokens recomputed per phase.
Offload stores occurred, but this screen does not exercise RAM-cache restore
or demonstrate its latency. C1 output hashes repeat exactly per workload;
C2 output hashes vary across repetitions despite fixed greedy settings.
This variation predates the V2/DFlash candidate and must be retained as baseline
evidence, not attributed to a new candidate by default. The run contains no
complete time-series memory/profile evidence or exercised long-context capacity.

The first idle probe began **148.8 seconds after the benchmark summary finished**,
not after five minutes or a day. On its 21-token prompt, headers arrived at
26 ms, first content at 354 ms and completion at 719 ms. Immediate repetition
gave 20 ms, 207 ms and 602 ms respectively; both emitted 11 tokens with no
reasoning tokens. PID/start-time values were unchanged; both processes had
zero major-fault deltas and zero recorded swap before/after. The first probe
added nine engine minor faults. These are cumulative host-process counters;
the large pre-existing major-fault totals do not establish faults during a
probe, and unchanged counters do not exclude device-level memory migration.

This pair did not reproduce the reported 10–15-second delay. Its 147 ms
first-content difference cannot isolate an idle mechanism: there is one pair,
the first probe prompt was not shown to have been warmed before idle, and
gateway/UI, device and compile-phase evidence is absent. The raw probe script
is `build-dev/release-v032-v2-20261019/idle_probe.py`; preserve its exact request
settings/timing definitions when comparing another arm. No long-idle
reproduction or fix is established.

Necessary adjustments to candidate matched measurements are bounded:

1. Keep this hashed short-prompt screen unchanged as one regression control;
   compare by workload and concurrency, preserving token trajectories and
   existing first-chunk accounting. Add the already-planned 4K/16K anchors
   separately rather than relabelling this screen.
2. Require observed prefix-hit/restore counters for the warm-conversation
   cell, and retain the existing raw queue/prefill/offload metrics. Repeated
   prompts alone do not establish a restored-cache workload.
3. Use a second fresh start and balanced baseline/candidate order before a
   speedup conclusion; preserve warmup rows, cache namespace and actual token
   counts. Keep memory/capacity and profiler evidence as separate gates.
4. For idle comparisons, warm the same probe before a known idle interval,
   record actual duration and first-content time, and repeat the pair. Keep
   same-process short idle distinct from the still-unreproduced day-idle
   symptom; no keepalive or configuration change is justified by this pair.

## Exact-package V2 control screen

The exact Brutus v0.32 package at `06edb20` passed the four text/reasoning/vision/
tool smoke cases and the20K RAM restore proof. Two restores reproduced the cold
output token IDs after local GPU-cache reset. First restore latency was5.08s,
then0.324s, with434,143,264 loaded bytes each. This newly started process was not
naturally idle for a day; the observation does not diagnose the reported issue.

The independent performance review reconstructed all32 baseline/control SSE
streams. Every measured response contains512 output tokens; each phase has six
successes and3072 generated tokens, without preemption or reset. C2 delivery
overlaps for16.50–16.61s. Per-workload median decode rates (stable V1 MTP2 versus
V2 without speculation) were English29.55→31.82tok/s and reasoning35.26→31.79
at C1; English28.67→30.72 and reasoning33.78→30.80 at C2. C1 reasoning has
identical512-token trajectories in both arms, so its slowdown is observed even
with the same continuation. Removing MTP is part of this control comparison.
English diverges after token182 at C1; baseline C2 also varies across repeats.

V2 restores384/618 measured prompt tokens from RAM, loading492,159,072 bytes
per phase, whereas stable computes all618 without hits. That is an observed
serving-cache difference, not isolated runner/prefill speed evidence. Candidate
also enables request logging and local qualification endpoints; repeat stable
with matched instrumentation before the final DFlash comparison. Keep natural
cache behavior in the primary production-profile comparison; label any separate
fresh-salt cold subset explicitly. No V2/DFlash speedup is established yet.

## DFlash attention and recurrent reservations

The retained native FlashAttention tests primarily cover causal Q24/KV4/D256.
The actual drafter requires noncausal Q32/KV8/D128 with eight queries and symmetric
2047/2047 windows. Brutus's default prefill preset includes that specialization;
the narrower packaging validation preset does not. The additional exact-package
probe compares the native operation with CPU FP32, rejects fallback and checks
bitwise replay after alternating allocations, at page sizes64/128 and history
lengths104/2048/2113/4097, including a four-request ragged batch.

Qwen GDN still reserves one recurrent state per speculative token. ReplaySSM is
not a configuration solution: Qwen3_5/GDN lacks the support interface, the dispatch
only initializes Mamba1/2, speculative Mamba2 replay requires FlashInfer/cache
mode none, Triton replay rejects V2, and all replay rejects KV connectors.
Forcing the flag would remove reservations without replacing GDN's state writes.
The seven-proposal/four-sequence memory requirement needs actual startup
qualification without silently reducing context or concurrency.
