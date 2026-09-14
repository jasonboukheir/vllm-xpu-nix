# Compact K4V2 on Brutus

Prepared 2026-09-12, before GPU execution. The operator subsequently handed
over Brutus and authorized the release goal. Execution evidence and live
checkpoints are in the private, ignored
`benchmark-results/kvarn-k4v2/20260913T051252Z/` directory. This plan defines
the acceptance workflow; completed release notes report the measured outcome.

The objective is an optional `kvarn_k4v2_g128_compact` mode with native Xe2
write, decode, cached-prefill reconstruction, and bundled Qwen MTP support.
The deliverable is a new coordinated `vllm-xpu-nix` release supporting this
option, with cleaned-up K4V2 commits in both source forks and measured memory,
quality, and performance comparisons against the working compact K4V4 release.
The user explicitly requested retaining both upstream bases and pursuing
performance as close to K4V4 parity as possible. Release publication and source
commit cleanup are in scope; host deployment is a separate handoff.

**Profiling prerequisite, added at the user's request on 2026-09-14 UTC:**
Before further performance changes, complete the
[K4V2 profiling milestone](kvarn-k4v2-profiling-plan.md). Unblock and validate
the profiling tools, capture the current candidate, and write an evidence-backed
critical-path and resource analysis that selects the next experiment. Allocate
time to resolving profiling failures rather than bypassing this milestone with
more speculative tuning. The release objective and acceptance gates below remain
unchanged. A fresh agent must read that milestone and the current run's
`CURRENT.md` and `status.json` before launching work; archived checkpoints are
historical evidence, not current execution instructions.

**Performance scope updated by the user on 2026-09-14 UTC:** Compare performance
with MTP disabled. Apply the retained parity target and formal performance floors
to matched MTP-off workloads. Keep MTP functionality, correctness, quality, and
lifecycle qualification. Preserve earlier MTP performance measurements as
historical evidence; they are not release performance gates or tuning targets.
This instruction supersedes earlier run-local queues that require an MTP2 speed
floor before continuing. Do not resume those queues unchanged.

1. **Freeze the experiment and complete the handoff.**

   Use the current coordinated release as the starting point: vLLM
   `3b6c822e8a4f56dc3904c7fce388a76dff4c7bdd`, kernels
   `9bb1e2d428b8a25b262c0cd6704614bdf6ee3ad8`, and the packaging revision
   recorded at execution time. Inspect local changes before copying sources.
   Preserve the actual upstream bases: vLLM
   `51da0ca66c8065619c79e35dff97aa99aeaf5644` and kernels
   `e7f20cbc18d741419a84aa57178e1f1cc0b453da`. Keep nixpkgs, Torch, Triton,
   oneAPI, and unrelated dependency locks pinned. This is a feature release
   on the existing bases, without an upstream refresh.
   Create isolated working copies in writable experiment storage, preserving
   required submodules. Keep immutable snapshots of each measured candidate.

   Freeze `RedHatAI/Qwen3.8-27B-INT4` revision
   `bf08f3dbd9a324e53956920aad378a1f1b6dd24a`, W4A16/BF16, TP1/PP1, eager V1,
   G128, 2,048 prefill tokens, prefix caching off, and the current FP16
   sink/recent-history policies. Start with MTP off; qualify MTP2 later.
   Retain four scheduler slots for the final serving profile, 262,144 maximum
   combined tokens per request, and the existing two-image limit.

   After the operator stops `vllm-xpu-chat`, verify actual GPU clients and
   free memory, including any other inference workers. Test through owned
   foreground workers on a free loopback port, normally 18000. Record process
   groups and clean up only processes started by this experiment.

   Before leaving the run unattended, verify that the intended command path
   can access the Nix daemon, compiler environment, model cache, GPU device,
   and output storage. This session currently writes only the packaging
   workspace and `/tmp`; use isolated source copies there. The earlier
   sandboxed systemd status query could not access the system bus, so service
   inspection and GPU access must be established during this preflight.
   Follow normal approval handling for any required access; do not weaken the
   sandbox or host permissions. Leave production activation to a later handoff.

2. **Define the storage contract and independent oracle before native work.**

   Add the compact preset and dtype registration. At D256/G128, one block/head
   contains 16,384 K payload bytes, 1,280 K metadata bytes, 8,192 V payload
   bytes, and 1,024 V metadata bytes: **26,880 bytes** total. Four KV heads
   therefore use a **107,520-byte page per layer**. The existing compact K4V4
   contract remains 35,072 bytes per block/head.

   Extend the independent benchmark oracle and nearby layout tests for K4V2.
   Cover every two-bit code, cross-byte boundaries, scale/zero offsets, fixed
   rounding behavior, constant tiles, arbitrary physical page order, and
   allocation accounting. Define the native V permutation explicitly and
   prove its inverse using an independently expressed mapping. Check packed
   bytes and reconstructed tensors separately so matching writer/reader bugs
   cannot cancel each other out.

   Reuse generic two-bit quantization as a reference implementation. Keep K
   precision, Sinkhorn iterations, scales, and FP16 windows fixed. Avoid
   reusing the padded `kvarn_k4v2_g128` preset as the capacity candidate.

3. **Implement native K4V2 and verify every consumer of the cache.**

   Extend the native balanced writer to quantize V to codes 0 through 3 and
   pack four values per byte. Adapt the DPAS fragment loader's V loads,
   unpacking, prefetch ranges, metadata offsets, and stride checks. Preserve
   K's four-bit loading and the established attention arithmetic.

   Extend native cached-prefill dequantization and the backend's layout and
   capability dispatch. Audit speculative verification and all cache-layout
   consumers, including Python/Triton fallbacks. Give the format an explicit
   ABI/capability identity so an older shared library fails clearly instead
   of interpreting K4V2 bytes as K4V4. Enable MTP admission only after its
   implementation and focused tests exist.

   Reuse the current native decode variant first. Any later scheduling or
   unpacking optimization gets its own correctness-passing candidate; the
   first implementation should make the bit-width change easy to attribute.
   Extend the service harness's dtype selection and provenance checks, which
   currently assume `auto` and compact K4V4 in several places. Record explicit
   `bfloat16` for unquantized controls: this checkpoint embeds an FP8 default,
   so `auto` alone does not establish an unquantized reference.

   Primary source areas:

   | Repository | Files or areas |
   | --- | --- |
   | vLLM | `model_executor/layers/quantization/kvarn/config.py`, cache dtype registration, `platforms/xpu.py`, `v1/attention/backends/kvarn_attn.py`, KVarN layout/dispatch helpers |
   | XPU kernels | `kvarn_balanced_writer_xe2.cpp`, `collective/kvarn_decode_mainloop.hpp`, `kvarn_decode_xe2.cpp`, `kvarn_dequant_xe2.cpp`, bindings/capability reporting, `benchmark/kvarn_utils.py` |
   | Packaging | Explicit candidate runtime composition and extensions to the existing correctness, replay, service, and comparison tools |

4. **Build narrowly, then pass kernel and lifecycle checks.**

   Reuse the pinned Nix toolchain and Brutus package overrides. Build the
   changed Xe2 attention components with bounded compiler parallelism; retain
   the release's Xe3p exclusion. The previous cold attention translation unit
   reached approximately 78 GiB host RSS, so inspect available RAM and avoid
   overlapping large builds. Use the existing virtualenv/Nix execution path.

   Test the exact candidate native library and record its resolved path and
   hash. A skipped GPU fixture is incomplete evidence. Reuse and extend
   `test_kvarn_compact_record.py`, `test_kvarn_dpas_layout.py`,
   `test_kvarn_decode_xpu.py`, `test_kvarn_native_xpu_dispatch.py`,
   `test_kvarn_mtp_xpu.py`, and the relevant configuration/cache allocator tests.

   Cover one and four requests, ragged lengths, partial and full tiles, sink
   and tail transitions, flush boundaries, recycled pages, mixed prefill and
   decode, multiple decode splits, and speculative accept/reject paths.
   Include dirty-allocation repeats, canaries, unchanged-input checks, finite
   outputs, and deterministic state ownership. Run the established K4V4
   regressions against the candidate package as well.

   Compare native K4V2 attention with attention over independently reconstructed
   K4V2 tensors. This proves implementation correctness separately from the
   model-quality effect of reducing V precision. Preserve existing applicable
   numerical tolerances and exact state invariants.

5. **Measure model quality before broad performance tuning.**

   Run matched BF16-KV, compact K4V4, and compact K4V2 controls using identical
   weights, prompts, tokenizer output, and frozen continuation tokens. Use
   `kvarn_forced_decode.py` and `kvarn_compare_logits.py` for persistent cache
   replay. Start at 4K/16K, then the longest context fitting the BF16 control
   with MTP off and images disabled. Do not change the weights to make a
   reference fit.

   Freeze the complete retained quality-threshold profile and its checksum
   before inspecting K4V2 scores. Preserve applicable thresholds; report top-1
   and top-5 agreement, selected-token and matched-logit errors, and coverage.
   The existing top-k capture cannot establish full-distribution KL divergence.
   A failed quality gate is a failed candidate for the current quality target,
   even when its kernel implementation is correct. Diagnose it without
   loosening the threshold after seeing the result.

   Add paired task checks for exact retrieval, instruction following, coding,
   tool-call parsing, and long generation, using retained fixtures where
   possible. Run two independent starts for the core comparisons. Numerical
   replay and coherent prose alone do not establish benchmark accuracy or
   justify calling the result lossless.

   At 128K and near 262K, perform full-model K4V4/K4V2 retrieval and lifecycle
   checks with room reserved for output tokens. Mark the absence of a same-GPU
   BF16 long-context control explicitly. Distinguish kernel coverage at 262K
   from an actual model request at that length. Broader reasoning evaluations
   can follow if the core gates pass and the run has time; partial eval sets
   remain partial in the report.

6. **Establish memory savings, service speed, and MTP behavior.**

   First compare at identical allocated attention-block counts, scheduler
   slots, MTP settings, and FP16 pool sizes. The exact expected saving is
   8,192 bytes per block/head, or 1.0625 GiB for 262,144 paged tokens across
   the 17 target-plus-MTP attention layers. Record actual allocator bytes,
   recurrent reservations, tail buffers, and device residency separately.

   Then compare at the same 0.96 memory-utilization budget. The last release
   reported 265,856 shared attention tokens; approximately 346,880 is the
   K4V2 projection if other reservations stay fixed. Explain any discrepancy
   using measured allocations. Keep the per-request ceiling at 262,144.
   Exercise occupancy above the K4V4 pool using concurrent requests and
   verify live occupied blocks; startup allocation alone is not this test.

   Benchmark 4K, 16K, approximately 65K, 128K, and near-limit contexts in that
   order. Use one and four active requests where aggregate cache capacity
   permits; four requests do not each receive a 262K cache. Compare performance
   with MTP off in both formats. Qualify MTP2 functionality separately. Warm
   each exact performance workload, use balanced
   A/B and B/A ordering with at least three measured repetitions per arm and
   two fresh starts for the core anchors. GPU work is serial, with compilation
   and profiling excluded from timed samples.

   Record TTFT, completed-call and request latency, per-request decode speed,
   aggregate throughput, MTP drafted/accepted tokens, preemptions, and log
   failures. Use actual output tokens and completed stream timing; MTP bundles
   do not establish individual inter-token latency. Run streaming, tool-call,
   two-image, four-request overlap, and prefill-during-decode service gates.

   Target parity within measurement variation, with 0-3% regression as the
   working tuning target for representative throughput/decode anchors. Keep
   the retained formal floor as an outer limit: throughput and decode ratios
   at least 0.95, latency ratio at most 1.10, against matched K4V4 controls.
   Passing that outer limit alone does not finish performance work. Investigate
   repeatable MTP-off regressions above 3%, including short contexts, before
   proposing the release. Report each core anchor as well as aggregate results
   so a favorable average cannot hide a material workload regression.

   Complete the required [profiling milestone](kvarn-k4v2-profiling-plan.md)
   to attribute MTP-off time to V unpacking, loads, register pressure, prefetch,
   flush/reconstruction, or host scheduling. Tune
   the measured limiting component, retaining the same quantization semantics.
   Repeat affected correctness tests and only the measurements invalidated by
   each change. Prefer the existing native scheduling policy unless evidence
   supports a narrow K4V2 specialization. Pursue improvements while there are
   concrete supported hypotheses; document any remaining tradeoff. Report
   sample variation and uncertainty, and leave an unresolved parity miss
   explicit rather than silently treating the 5% floor as the user's target.

7. **Checkpoint continuously and prepare the handback.**

   Use a fresh ignored `benchmark-results/kvarn-k4v2/<run-id>/` directory.
   Save the plan and threshold profile before measurements, then atomically
   update `status.json` and a short progress log after every stage/arm. Keep
   source snapshots or patches, exact commands, runtime/library identities,
   raw logs, token IDs, comparisons, failed attempts, and outstanding checks.
   Resume only results whose source and runtime identities still match.

   Each build/test/service job needs a recorded process group, a timeout
   appropriate to its workload, and cleanup. Derive expensive full-model test
   timeouts from the measured baseline; allow multi-hour near-limit prefills
   when warranted. On OOM or timeout, terminate the owned worker, verify
   memory recovery, diagnose, and retry only after a concrete change. A GPU
   device failure that persists after owned-process cleanup ends GPU work
   pending operator recovery; do not reset the device or reboot unattended.

   An overnight run targets native functionality, core correctness, model
   quality screening, and useful memory/performance evidence. Cold compilation
   or long-context evaluation may require a follow-up run. Budget remaining
   time toward finishing prerequisite gates before opening additional tuning
   experiments; never label unrun checks as passed.

   Qualification requires all applicable gates above, including live MTP2 and
   concurrency checks. A failed candidate starts another focused repair or
   tuning iteration; it does not complete the release objective. If progress
   needs operator recovery or exceeds the available run, preserve the exact
   unfinished stage and evidence. Clean up experiment workers at handback.
   Preserve the operator's maintenance state when the operator stopped chat.

8. **Clean up the source commits and validate the final package.**

   Fold implementation experiments, temporary fixes, and follow-up repairs
   into coherent K4V2 feature/fix commits in each fork. Include the associated
   tests and user-facing documentation with their owning change, and preserve
   sign-off and authorship. The cleanup concerns this feature's work; retain
   the established upstream bases and unrelated existing patch stack.

   Review the complete paired delta from `xpu-v1.8.0`, the native ABI, fallback
   dispatch, and the Nix patch/application and architecture configuration.
   Preserve recovery refs and record old-to-final commit mappings. Verify the
   actual origin push URLs and current remote tips before publication. Prefer
   cleaning unpublished feature commits; if an already-published development
   history must be rewritten, protect it with recorded backups and exact
   leases. Existing release references remain immutable.

   Freeze the clean final fork SHAs, rebuild the paired Nix candidate with
   Brutus's `package.nix` overrides, and record the exact loaded native
   libraries. Source-tree equivalence after commit cleanup can preserve earlier
   evidence only when build-input equivalence is also proven. Revision-based
   version changes can alter derivations; validate those differences and run
   affected checks plus final live service coverage on the package to release.
   Run the applicable Python, C++/SYCL, Nix formatting, unit, kernel cache
   identity, configuration, and GPU checks through the pinned working tools.

9. **Publish and verify the coordinated feature release.**

   A compatible new cache option calls for a minor stack version on these same
   bases: provisionally `xpu-v1.9.0`, subject to checking all local and remote
   release refs at execution time. Keep individual package versions tied to
   the existing upstream labels and final source revisions.

   Publish the clean source commits and new coordinated annotated tags and
   permanent `releases/<version>` branches at the exact qualified fork SHAs.
   Point both packaging source input refs at those tags and refresh only those
   source locks. Verify resolved SHAs and package equivalence to the qualified
   candidate; rebuild and revalidate any material difference before publishing
   the packaging release. Commit the final manifest and release notes, then
   publish and verify the corresponding packaging branch and annotated tag.
   Use explicit refspecs and per-repository atomic publication where supported.
   Recover partial publication from recorded intended SHAs; never move a
   published release reference to repair it.

   Document how to select `kvCacheDtype = "kvarn_k4v2_g128_compact"`, supported
   MTP/concurrency behavior, the qualified model/profile, exact retained bases,
   clean fork SHAs, runtime derivation, measured memory/capacity, quality,
   throughput/latency, and outstanding qualification limits. Update the option
   documentation and KVarN guide. Keep raw/private benchmark artifacts ignored.

   Completion means all three remote release branches and peeled annotated
   tags match the recorded qualified commits, the packaging manifest resolves
   the intended source pair, and the release notes accurately describe the
   evidence. Report the release links and the operator's current service state.
   Changing Brutus's deployed pin or activating K4V2 is a later requested action.

For unattended execution, start this plan as an explicit Codex goal after GPU
handoff. [OpenAI's goal documentation](https://learn.chatgpt.com/use-cases/follow-goals)
describes continuation across turns for work lasting multiple hours. Brutus
and the local Codex runner need to remain available; a goal does not remove
runtime, account, connectivity, or approval limits. Persisted artifacts make
an interrupted run resumable. Use the goal tool's actual status to determine
whether execution is active; this document does not control that status.

Suggested handoff instruction:

> The GPU is free. Set a goal to implement and qualify
> `docs/kvarn-k4v2-plan.md` on Brutus and publish the new coordinated release.
> Keep both current upstream bases and unrelated dependencies pinned. Clean up
> the K4V2 commits in both forks, use the existing harness, and tune performance
> as close to K4V4 parity as possible. First complete the profiling milestone in
> `docs/kvarn-k4v2-profiling-plan.md`: unblock collection, validate traces and
> counters, and explain the measured bottlenecks before further optimization.
> Continue through the checks and release
> verification, saving checkpoints. Leave host deployment for review.

The plan follows the existing [KVarN profiling workflow](kvarn-profiling.md) and
[xpu-v1.8.0 qualification](releases/xpu-v1.8.0.md). The method and published
accuracy claims originate in [upstream PR 46812](https://github.com/vllm-project/vllm/pull/46812)
and its [RFC](https://github.com/vllm-project/vllm/issues/46613); those results
are motivation for this experiment, not qualification of the Brutus candidate.
