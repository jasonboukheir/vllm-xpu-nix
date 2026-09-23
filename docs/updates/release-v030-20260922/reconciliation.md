# Parent reconciliation work record

The later user-selected release acceptance is recorded in `release-acceptance.json`.
Actual Brutus operation passed. Extra controlled optimization comparisons are
not release gates for this scope and were not run; no new speedup is claimed.
The independent reports retain their original recommendations as history.

**Source audit/reconciliation gate passed:** all 27 distinct target-specific
reports have been reviewed, hashed and mapped. Inventory coverage matches the
16 vLLM and 11 kernel commits, including released fixes absent from main.
`commit-map.json` is the per-original-commit record, including report hashes.
Replay, final source review and verified source-main publication are complete.
See source-review.md for final identities, adaptations and validation.

## vLLM dispositions reviewed

| Original rows | Residual / planned owner |
| --- | --- |
| v01, v06 | DROP together: exact net-zero inspection workaround/revert; retain fixed Torch and cold lifecycle qualification. |
| v02 | KEEP separate GDN request-index normalization; correct the outdated padding comment and add discriminating suffix-padding coverage. |
| v03 | KEEP scheduler arithmetic-grid and guarded mixed GDN dispatch, paired with k05; preserve upstream checkpoint handling and original exclusions. |
| v04 | ADAPT separate functional fixed-tree XPU residual RMSNorm. Restore upstream non-XPU dispatch and native CUDA tests; add effective XPU coverage. Preserve the existing scoped Gemma caller. |
| v09 | KEEP independent guarded explicit FP16/BF16 cache writer translation; fold v15's independent causal reference tests here, forbidding native fallback. |
| v05, v07, v08, v10–v16 | One coherent KVarN feature commit. Preserve the final released behavior rather than historical intermediate restrictions. v15's reference-test subset belongs to v09 as above. |

The intended vLLM stack therefore has five explained commits. Every original
SHA remains in recovery history; preserve original attribution/sign-off in
the surviving commits and map the two v15 subsets explicitly.

## Required KVarN port details

- Keep the existing eager V1 guard, supported format/draft combinations,
  concurrent MTP admission, vision and narrow AEON request-stability behavior.
  Retain native plan classes named `V2`; that internal name is unrelated to
  Model Runner V2. No new backend/execution-mode support is requested.
- Keep `block_table_cpu` and actual physical rows. Drop removed common CPU
  length fields. The existing builder's `cam.seq_lens.tolist()` fallback is
  exact; never replace it with `seq_lens_cpu_upper_bound`. Preserve its exact
  per-builder list for flushing/materialization. Update the stale cached-copy
  comment and retain the exposed synchronization cost as unmeasured in this release.
- v16's surviving production change is out-of-place CPU upper-bound addition.
  Adapt its existing tests to owner/overlapping-view/separate/missing bounds,
  retained snapshots, two draft steps and deliberately optimistic bounds.
- Keep v11's narrow exclusion from target mutable builders. V1 already builds
  dedicated draft metadata before its first forward and later draft steps.
  Physical zeroing must also enumerate applicable draft groups without
  creating another mutable owner; add a behavioral regression for omitted
  draft storage and null/pool-qualified zeroing.
- Preserve HiSparse role, placement, layout and host-capacity fields and early
  dispatches. Extend `num_blocks_of` for independent device pools while keeping
  host semantics. Respect `has_layer_views` raw backing in allocation. Use
  field-preserving worker-group projection, maintaining empty group positions.
- Carry every pool-qualified copy, zero, scheduler and lifecycle consumer;
  retain ordinary/shared-pool and HiSparse behavior. Equal usable attention
  capacity remains the supported target/draft allocation policy.
- Fold v08/v13 into one reservation helper for admission, auto-fit and
  overrides: recurrent costs use the actual Mamba spec times `max_num_seqs`,
  attention override scales attention only, and each pool's null is charged
  once. Whole compatible buckets and spec-local mixed-format history policy
  survive. v0.30 has no `max_num_active_seqs` or later snapshot lookahead fields.
- Preserve 26,880-byte D256/G128 K4V2 records and paired `value_bits=2` ABI;
  K4V4 remains 35,072 bytes. Keep compact 16/8 history policy, split boundaries,
  bounded serial materialization, exact extents and current-stream lifetime.
- Preserve whole-model reset before replacement-model loading. Verify shutdown
  cleanup and any reachable retained-model cache replacement. Cache-only
  invalidation must preserve live implementation registration; do not apply
  whole-model reset at the wrong boundary. Do not claim an unobserved leak.
- Bounded cleanup: remove definition-only `_KVARN_NATIVE_RECORD_BYTES`, replace
  `_pack_dpas_v4`'s test with `_pack_dpas_v(q, 4)`, remove the production-dead
  `use_trusted_qlen1_inline_plan` wrapper branch and its synthetic-only test
  after rechecking callers. Preserve active native/reference/CUDA/preset paths.

## Kernel dispositions reviewed

- k01 KEEP: deterministic oneDNN W4A16 attributes, unchanged upstream and
  dependency implementation, focused numerical/replay gates retained.
- k02 KEEP: early disabled-policy host recursion pruning survives
  upstream's different enabled-device extern repair. A controlled cold compile/RSS comparison remains unrun; retention follows
  the user-selected operational acceptance scope. It prunes eight policies in the narrow
  packaging profile and none in the actual all-policy Brutus profile. Preserve
  upstream's broader b16 routing. No measured build-speed claim yet.
- k03 KEEP: the supported untiled non-speculative convolution guard and
  poison/replay regression remain necessary; the upstream tiled implementation
  is unchanged. Do not substitute a new tiled implementation during this port.
- k04 ADAPT: retain the fresh-state O2 producer/consumer workgroup barrier
  inside `if (!has_prev_state)` before its output-dimension loop. Upstream
  barriers already order U and next-chunk S. The installed regression suite and 14 supplemental independent-reference
  cases passed. Comparative performance controls were prepared but not run.
- k05 KEEP: the trailing optional `split_mixed_non_spec=False` schema and
  cached-decode arithmetic path must ship with v03. Preserve the target's
  upstream checkpoint behavior. The k04 test insertion dependency is textual.
- k06 KEEP production and tests: selected v0.1.15 lacks the immediately following
  #600/`da16a559` ragged traversal fix. Correct the paused target's different
  disposition. Preserve previous accepted capacity and inactive-state canaries;
  this remains separate from v02 padding and v03/k05 mixed arithmetic.
- k07 ADAPT test-only: retain the five eager native attention replay cases,
  reject `_fallback_varlen_attn` directly, and check the independent FP32
  paged-attention reference before exact replay. Avoid warn-once log inference.
- k08 KEEP: the native KVarN feature owns k09 and the reconciled k10/k11
  refinements. Preserve variant 18, D256/G128/Q24/KV4 guards, all public schemas,
  current-stream scratch lifetime and shared ordinary-attention corrections.
  Remove only the definition-only `fill_k_fragment_by_coordinate` helper;
  `load_k_dpas_quantized` and `load_v_dpas_quantized` remain used by materialization.
  Correct obsolete GRF/ID20 comments but retain library `MIXED_GRF_SIZE` and
  the public legacy scratch ABI without evidence supporting their removal.
- k09 KEEP/fold k08: compact K4V2 records, width-aware readers/writers and
  trailing `value_bits=4` defaults remain paired with v10's explicit value 2.
  Historical coordinate coverage is evidence for ownership, not GPU qualification.
- k10 KEEP/fold k08: independent reanalysis confirms a 1.03032884x
  historical short-cell gain and 216 exact matching output comparisons. Broad
  repeatability failures and the controlled 3.67% K4V4 regression remain open.
  Fresh-pair ABBA was not run. Retention follows source review, installed
  correctness and the user-selected operational acceptance; no new speed claim.
- k11 KEEP/fold k08: preserve resident FP16 block loads, independent K/V
  64-byte alignment guards and scalar fallback. Recomputed eight historical
  timing cells and 32 saved-output hashes support the 39–51% K4V2 span gain;
  the short K4V4 +12.3% anomaly remains explicit. Installed independent correctness passed; fresh controlled ABBA was not run
  and is outside the user-selected release acceptance.
  k11 is independently retainable if the k10 reducer gate later rejects k10.

The intended kernel stack has eight explained commits: k01 through k07
individually, followed by the coherent k08 feature. k02/k10 are retained under the later user-selected Brutus operational
acceptance. Their extra controlled comparisons are unmeasured, not passed;
this decision makes no fresh performance or build-resource benefit claim.

## Validation interpretation

Audit CPU extraction checks and rehashed historical evidence support source
dispositions. They are not installed-pair qualification. The installed-pair results and later acceptance scope are recorded in
`qualification-status.json` and `release-acceptance.json`. Preserve historical
semantic/NaN-padding failures and concurrent free-running trajectory differences.
Hardware-unavailable CUDA checks must remain marked unavailable. No DFlash
experiment, new BF16 serving milestone or performance-ceiling conclusion belongs
to this release workflow.
