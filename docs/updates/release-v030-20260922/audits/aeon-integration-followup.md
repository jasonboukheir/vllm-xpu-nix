# AEON integration follow-up: preserve the failure and separate its causes

Read-only source-owner reconciliation follow-up, 2026-09-23 UTC, by
`/root/r30_aeon_audit_followup`. The candidate and exact published baseline
both fail the scoped AEON within-wave duplicate-output gate, producing the
same two complete output sequences. This native-path limitation predates the
rebase; its precise cause and any additional candidate differences remain
unresolved. Do not publish a new whole-model invariance claim, waive a newly
introduced regression, or restore/drop a source patch on this evidence alone.
Issue #17 remains paused; this concerns the ordinary
eager V1 release and its existing optional AEON path.

Only this report was written. No source, refs, locks, service state or paused
artifacts were changed; no build, GPU test or service operation was run by
this reviewer. Both source worktrees were clean when inspected. Read the
source AGENTS files, the explicitly invoked update skill and integration
failure rule in `references/workflow.md:591`, v03/v04/v05/k04/k05 audits,
and parent reconciliation.

## Frozen evidence

Candidate source pair:

- vLLM `466b9d5abe84f13c29b38c686e4d77fc09e09aa4`.
- kernels `193cee080d654ff67c4f8bfaf924201bc88b9cf5`.
- Installed runtime
  `/nix/store/zhv84h5afvyx99fkw0l56rpj8rldhivl-v030-brutus-qualification-env`;
  process package
  `/nix/store/h3phd3ijav9zp34kzbvhyicxqsbanr94-python3.12-vllm-xpu-0.30.0+unstable.2026.09.23.g466b9d5`.

Raw directory: `benchmark-results/release-v030-20260922/aeon-api-02/`.
Independently recomputed SHA-256:

| Artifact | SHA-256 |
| --- | --- |
| `manifest.json` | `a6b9f5a9ead194725a9860cf142bae00dee3d7b2a8df9f27f73144c8ace51c70` |
| `service-gate.json` | `7ec50ed3d5707e171e877f1636c890e4217dae85e8bafe9fbbd05b80cbaa391d` |
| `service.log` | `74869ec4b43cae271e00ab783e82046dcfc8be47c94d2831c2a3e316fdee2e00` |

The actual process has the canonical AEON model ID and pinned revision
`6b0622f4354481d5d04577d48ba0db844efc1330`, BF16 compute/W4A16 weights,
K4V4 compact cache, eager V1, language-only, no MTP or prefix caching,
65,536 context, four scheduler slots and 2,048 batched tokens. Both scoped
projection and RMSNorm selectors log true, with `profile_eligible=true`.
This is distinct from the ineligible offline-path harness attempt `aeon-api-01`.

All eight isolated/repeated completions and the first three concurrent
fixture results have 384 output IDs and hash
`28febb9f278a6428b35685bccfde9d974b4d7346a59e30f9ab6db964e5021504`.
The fourth fixture, `dialogue-127-probe-3`, has hash
`9859bfb2d797bd3560ce1995e566a1f55e76a3b3537d2a8f34470265e712886f`.
The first mismatch is zero-based output index **127**, token **3555 versus
1156**. The prior 127 output tokens match exactly. All prompts contain the
same 127 IDs. The gate explicitly requests temperature zero; no random
sampling explanation is established by the model-default temperature warning.
Metrics observed four simultaneous running requests and no quality findings.

The fourth **client fixture** is not proven engine row or physical state slot
3: those mappings and per-step batch admission were not captured. Nor does
output index 127 prove an immediate 128/256 boundary fault. With a 127-token
prompt, output index 127 is predicted at logical sequence length 254. Earlier
hidden-state/logit differences may precede the first unequal argmax. The
shutdown EngineDeadError follows the harness-triggered SIGINT and does not
explain the already recorded duplicate mismatch.

The gate aborts before cancellation/replacement and mixed scheduling; these
phases remain untested for this AEON invocation. The parent reports 705
installed native/backend tests, 14 supplemental GDN oracle cases, and the
ordinary RedHat K4V2/MTP2/vision API/lifecycle/long-context gates passed.
Those successes narrow the investigation but do not replace this failed gate.

The parent subsequently completed the same gate on the published
`xpu-v1.10.0` runtime
`/nix/store/a7jkfx4zc58nsicqvkif0pdqbgh5v162-mtp-draft-cache-comparison-env`,
process package
`/nix/store/v8jzs75kb7w0wy0542a99kl6njxsd1kp-python3.12-vllm-xpu-0.28.0+unstable.2026.09.15.g2fdd9d7`.
This reviewer independently read and hashed those completed artifacts in
`benchmark-results/release-v030-20260922/aeon-baseline-01/`:

| Artifact | SHA-256 |
| --- | --- |
| `manifest.json` | `2d2d683bd4297695d1fa20377bf704d7327592f573b3a0e3bd1922c94d89fc71` |
| `service-gate.json` | `e891d61afb30c58e02e88f27f32e258155cae7d547a5b8f13f8e76f7cc1dc091` |
| `service.log` | `ef6af3b4768e81f68a1ee7a9cbafa63eb2f30e307b4aa6331516d30b88b9b42f` |

Both selectors and eligibility are true. Fixture hash
`b74e4406f95e34e3c9b164f64e0232b06c9718b37150d18330eebc81f449dc80`
and gate-script hash
`7f48d9fa12cd714b3ed67e753db1177e7a39d5e93e6f42ced83d0f3b305e94d1`
match the candidate; four-way overlap was observed. All eight baseline
isolated/replay outputs exactly equal the candidate's isolated sequence.
Baseline concurrent fixture 0 also matches; fixtures 1–3 each equal the
candidate's alternate **entire 384-token sequence**, first differing at index
127. Thus candidate 3:1 and baseline 1:3 are distributions of the same two
observed trajectories, not a controlled error-rate comparison. Arrival order
and engine row/state placement were not held constant. The published stack's
failure establishes an inherited native-path limitation for this workload;
the output equality is stronger evidence than simply observing two failures,
but still does not identify its internal origin or prove universal parity.

## Contract and source trace

**v04 has a narrower, independently testable contract than whole-model
token invariance.** Its functional residual RMSNorm performs the FP32 sum
and a fixed row-local reduction, returns separate normalized and rounded
residual tensors, preserves inputs, and must give exact equal outputs for
the same row across batch placements/sizes. The scoped Gemma caller uses
that helper with FP32 `weight + 1`. Direct XPU tests of that contract have
passed in this candidate. They do not establish equality of every upstream
projection, GDN state, attention output and model logit across arbitrary
schedules.

Relevant old release (`2fdd9d71...`/`2c39287...`) to candidate differences:

1. `batch_invariant.py:1051` restores the native residual branch only when
   the platform is **not XPU**. The XPU fixed-tree implementation is unchanged.
   `layernorm.py` and the Gemma caller are unchanged. The request-stable
   dispatch diff is typing plus using `part_residual` to allocate its padded
   view; with the validated identical residual shape/dtype/device this does
   not change the active functional callback path. Blaming v04 from the
   current model result would be unsupported.
2. `gdn_attn.py:247` replaces the removed CPU computed-token field with
   `(m.seq_lens - m.naive_query_lens()).cpu()` for the eligible profile only.
   This is the same mathematical expression as
   `CommonAttentionMetadata.compute_num_computed_tokens()`. The result becomes
   an immutable tuple at metadata construction, together with request query
   boundaries and prefill phases. `platforms/xpu.py:359` validates agreement
   across GDN groups and supplies canonical request slices. A caller ordering
   or row-mapping error is still testable, but no incorrect formula was found.
3. `request_stable_linear.py:353,418` fast-paths ordinary B1 and packed B2–B4
   decodes without using the absolute position. Canonical 64-row lane
   placement is used for prefill/multi-row requests. Thus stale positions
   could first affect a prefill/join operation, then amplify into a later
   output mismatch; a decode-only position-boundary explanation is weaker.
4. The native GDN production difference is k04's reduced barrier form:
   restore the upstream U group barrier and replace the unconditional added
   per-chunk fence by a fresh-state O2 barrier inside `!has_prev_state`.
   k05's mixed decode/prefill split and v03's adapter are unchanged, apart
   from an adapter comment. Candidate-to-released-barrier-only is a clean
   causal control; the existing direct tests are useful but cannot exclude a
   model-geometry/input/stream-sensitive race by themselves.
5. KVarN now snapshots exact device lengths instead of a cached CPU field;
   its lifecycle cleanup also changed. Native split policy, resident-window
   policy and numerical decode implementation otherwise retain the released
   path. At sequence lengths in this short fixture the K64 work-unit rule
   collapses B1/B2/B3/B4 splits to one until at least the relevant threshold;
   there is no source evidence of a split-count transition at logical 254.
   The log confirms one split on both bound and generic paths. Pool/row
   aliasing, differing earlier admission paths or native arithmetic still
   require captured state to distinguish.

The August acceptance used different source objects
`5acf37c1...`/`8f52482...`, the native residual normalization branch, and
**`KVARN_NATIVE_XPU=0`**, verified in `final-engine-first.log`'s actual
launch line as well as `run-service.sh`. The current factory proves native
Xe2 variant 18 is selected. That historical full-model pass establishes a
behavioral requirement, but it does not demonstrate that today's native
attention configuration ever passed it. Do not re-enable a retired selector
and silently describe the result as qualification of the release path.

## Smallest discriminating sequence

1. **Published-baseline gate completed, failing as recorded above.**
   Preserve its identities, selectors, fixture/harness hashes and actual
   overlap. For subsequent controlled workloads, if the published release
   passes and the candidate repeatably
   fails, this is a candidate regression to resolve before release. If both
   fail, the native configuration already lacks whole-model invariance;
   that alone does not prove identical causes, bounded numerical error or
   absence of additional candidate regressions. Compare the isolated and
   duplicate output patterns and perform matched-history investigation.
   A single pass versus single failure is insufficient to exclude a race.
2. **Use the prepared released-barrier runtime with otherwise identical
   candidate package/configuration.** Repeat the fixture under current and
   released barriers with controlled fresh starts. Verify that the loaded
   GDN DSO is the only relevant component difference. If released barriers
   reproducibly repair the failure, fold restoration into k04, preserve the
   prior tip and rerun dependent GDN/model checks. If both forms fail, do not
   call either safe or unsafe based only on this test. The five-form direct
   GDN qualification remains independently required by k04's audit.
3. **Capture the first unequal internal result under identical histories.**
   Reuse the persistent forced-token/logit mechanism in
   `scripts/kvarn_forced_decode.py`, extending the experiment to identical
   B1/B4 requests and a deliberately staggered join. Force the retained
   isolated sequence after recording the unmodified logits; do not compare
   post-divergence free-running trajectories as identical-history accuracy.
   Start with the first 130 outputs, retaining raw logit top candidates and
   margins. Different logits before step 127 would disprove an immediate
   step-127 cause; equal logits with unequal token IDs points to sampling or
   result mapping. Retain the original free-running failure independently.
4. **Metadata-specific probe, before speculative repairs:** capture scheduler
   request ID, scheduled/computed counts, engine row, query starts, exact
   device lengths, prefill phase, recurrent slot and attention block IDs at
   prefill/join and around the first unequal logit. Assert the generated GDN
   start tuple equals `seq_len - query_len` and the model positions for each
   mapped request, not merely another copy of the same formula. Record
   packed projection path, mixed-split predicate, native split count and
   resident/packed block state. Reordering fixture IDs alone does not reveal
   physical placement. A narrowly scoped trace around the first unequal
   layer can then distinguish GDN recurrence from full attention/projection.

A cheap native supplement is four identical T127 prefills in permuted,
poisoned state slots, followed by a carried decode chain, compared against
four B1 calls with BF16 and FP32 recurrent state. Existing ragged batch
separation covers `(127,4095,4095,127)`, while the mixed split test is one
decode plus one T127 prefill with FP32 SSM state. Extend those fixtures rather
than inventing a second native implementation; include three decodes plus one
prefill and BF16 state if trace shows that actual schedule. Compare output,
z, convolution and SSM state, with independent reference accuracy as well as
exact repeatability. A stable but wrong computation is not a passing result.

## Disposition and remaining qualification

Keep v03/v04/v05/k05 requirements and provisional k04 adaptation pending
discrimination; no source disposition change is justified yet. Retain the
failed invariant as failed. The published native baseline now establishes
the same observable limitation on this workload. Explicitly distinguish that
inherited optional whole-model limitation from v04's passed row-local
functional contract and from any new rebase regression. This report does not
authorize relaxing the release scope or dropping the optional selector.

Run cancellation/reuse and mixed scheduling even if exact token equality is
an inherited failure: use a diagnostic continuation that records phase
failures without aborting the remaining independent phases. Do not rename
that diagnostic aggregate as an invariance pass. Preserve actual selector
evidence and confirm RedHat's selector remains inactive. Any repair must
return to its owning patch, then receive the runbook's map/lock update and
affected final-package checks. No DFlash speedup or performance ceiling is
inferred from these results.

The baseline completion was incorporated before finalizing this report.
Cancellation/reuse and mixed-phase outcomes remain pending parent capture.
