# v05 — KVarN cache, independent pools, MTP and lifecycle

**Disposition: ADAPT.** Retain the KVarN requirement and its paired native ABI.
The frozen target does not implement KVarN or independent attention/recurrent
device pools. Preserve upstream's new allocator/checkpoint behavior while
reapplying the local pool, metadata ownership and cache lifecycle extensions.
There is no evidence-backed wholesale deletion of this feature in this release.
The concrete simplification opportunity is to use the new spec-aware backend
block-size interface instead of consulting global target configuration when a
per-layer spec is available.

## Frozen inputs and scope

| Item | Immutable identity |
| --- | --- |
| Assigned commit / final old vLLM tip | `466b9d5abe84f13c29b38c686e4d77fc09e09aa4` |
| Parent | `984e6c3edda05c4f2dca37703534cb0e69cfa0fa` |
| Subject | `feat(xpu): add KVarN cache with compact K4V2 and bundled MTP` |
| Stable patch ID | `ffe68651f548654af1e078df09d79b6389c9aa3f` |
| Old vLLM upstream | `ced6857afa0ea7b2e3f0846a62e1394e90f15607` (v0.30.0) |
| New vLLM upstream | `db9527a46873454610df6dbedf79a36d6bf1a7f6` (v0.31.0) |
| Old kernel upstream | `1c7cbeee1cd0c1481d48f5031f679a47f3f0ef45` |
| Old kernel tip / paired feature | `193cee080d654ff67c4f8bfaf924201bc88b9cf5` |
| Selected kernel source | `ddf336d86e3c8602888572a3502f951abd51df12` (release/0.1.15.4) |

Read the update skill's `references/commit-audit.md`, `docs/fork-delta.md`,
this run's `inventory.json`, sibling vLLM `AGENTS.md`, and the published
`docs/releases/xpu-v1.11.0.md`. The complete five/eight-commit inventory and
individual patch identities are in `inventory.json`. No source, Git state,
services or GPU state was changed during this audit. Only this report was
written. All implementation observations below refer to the frozen objects,
not the current working tree.

## Requirement and final old behavior

This is a consolidated feature commit, 75 files and roughly 20,767 inserted
lines, including tests, optional configurations and reference implementations.
It is not one isolated correctness workaround. Its required production behavior
is compact K4V2 attention for the target and bundled MTP draft, with bounded
uncompressed tails, native packing/readers, exact accepted-history boundaries,
independent attention/recurrent capacities and pool-qualified block ownership.
Full-cache materialization would defeat the feature's memory benefit.

The previously qualified configuration is Intel Arc Pro B70 with
`RedHatAI/Qwen3.8-27B-INT4` revision
`bf08f3dbd9a324e53956920aad378a1f1b6dd24a`, W4A16 weights, BF16 compute,
eager Model Runner V1, `kvarn_k4v2_g128_compact` target/draft, MTP2, four
scheduler slots, 2048 batched prefill tokens, maximum context 262144, prefix
caching disabled, and Qwen image support. The recorded usable attention capacity
was 346880 shared tokens. Four scheduler slots do not mean four full contexts.
These are historical qualification facts, not v0.31 measurements.

The commit also preserves natural-layout/reference configurations, compact
K4V4 and mixed BF16 draft choices, deterministic oneDNN policy, and optional
scoped request-stability machinery. The RedHat profile does not activate the
AEON-specific projection/RMSNorm policy. Removing these alternatives is a
deliberate support reduction, not a demonstrated upstream replacement.

## Upstream coverage and minimum residual change

### 1. Cache format and native operators — KEEP

`git grep` for `kvarn` and `k4v2` in target vLLM `vllm/` and `tests/`
returned no matches. The same search in selected kernel source `csrc/` and
`vllm_xpu_kernels/` returned no matches. Inspection of the target cache enum,
backend registry and allocation paths confirms that this is an absent feature,
not an alternate spelling of an upstream implementation. Other quantized cache
formats do not establish the KVarN record ABI or equivalent quality/capacity.

Retain the local enum range 100–105, dtype/config plumbing, registry/backend,
packed-record spec customization, quantization helpers, native dispatch,
bounded materialization and native/reference regression tests. Preserve the
native record layouts and argument ordering with kernel `193cee0`'s successor.
Do not delete a vLLM call while retaining a stranded native operator, or replace
the quantized cache with BF16 to make the release build.

### 2. Independent pools and allocation accounting — ADAPT, not obsolete

At the target, `KVCacheConfig` still has one device `num_blocks` and special
HiSparse host fields, with no `KVCachePoolSpec`. `get_kv_cache_config_from_groups`
still overlays normal cache groups into one device allocation;
`worker/utils.py::allocate_kv_cache` asserts every normal cache tensor has the
same `size` and creates one backing buffer. `KVCacheCoordinator.__init__`
constructs one ordinary device `BlockPool`. Thus upstream grouping improvements
and HiSparse's separate host storage do not provide the local requirement.

The residual remains local `KVCachePoolSpec`, pool IDs on blocks/tensor
placements, per-pool demand/admission/watermarks, `MultiBlockPool`, independent
allocation and group-qualified zeroing/copies/freeing. Keep natural KVarN page
geometry and grouping whole compatible layer buckets; shared-pool page
equalization would waste the memory this feature is intended to save. Retain
the recurrent reservation for every scheduler slot, null-page costs and
per-pool multi-worker reconciliation. The legacy minimum `num_blocks` is not a
valid allocation bound for every pool.

Important target changes to retain during reconciliation:

- `77cca24dcc14830fe5119ee18e97c4292ec779f8` changes HiSparse GPU capacity
  accounting: `_pool_bytes_per_block` and `_max_memory_usage_bytes_from_groups`
  filter host-resident groups. The independent-pool branch must coexist with
  those target filters and HiSparse's specialized allocator; do not restore
  the old unfiltered shared-pool implementation while resolving conflicts.
- `29468dde8b515031dc6d4d9d06bf0a2fa0442098` adds
  `KVCacheSpec.block_stride_alignment`; `_get_kv_cache_bytes_per_block` rounds
  up to the LCM of layer alignment requirements. The local
  `_plan_kv_cache_tensors` already calls that helper per physical pool: use
  the **target** helper so aligned strides remain correct. Preserve alignment
  fields when merging/replacing specs. This is not a reason to reintroduce a
  shared allocation for KVarN.
- `927c87b3472a39da79a95a8df12d0a2b21e2d32e` moves DCP ownership onto
  `KVCacheSpec.dcp_sharded`, including replicated draft specs. The target
  removes `dcp_world_size_for_kv_cache_spec`; `SingleTypeKVCacheManager`
  now derives DCP/PCP size from the spec. Do not resurrect the removed helper
  when editing the adjacent `block_pool=` argument. Retain the target's
  grouping, layout selection and spec merge fields.
- `a5738bfe4cba03510c2a31c4ef4d6b8d8dda33eb` adds TP replication metadata
  and `kv_cache_groups_tp_replicas`. Keep the target's final assignment to
  `kv_cache_config.kv_tp_replicas` after local per-pool reconciliation.
- Keep the target `_project_kv_cache_groups_to_worker` fast path that returns
  an unchanged fully owned group, and its existing
  `is_eagle_group=group.is_eagle_group and bool(worker_layer_names)` handling
  for partial projections. An empty worker projection must not claim a draft
  group.

Generic CPU/V2 worker changes in the local feature are needed because the
scheduler's zero/copy payload changed globally. The XPU KVarN V2 guard does
not make those consumers dead code: non-KVarN workloads use the same payload
schema. Keep their compatible handling unless the parent redesigns the schema
and proves all consumers together.

### 3. MTP annotation and metadata ownership — upstream improvement, retain residual

`0bce411a073e0da2c9bded9bb6ced1cbee51c564` replaces the old
DeepSeek-only fallback with `_uses_trailing_mtp_layers`, adds
`_groups_partition_layers_exactly`, and calls `_annotate_eagle_groups` on the
hybrid path with the new trailing-layer fallback. Use that target annotation
unchanged, including its partition guard. Do not copy back the old
`use_deepseek_v4_fallback` keyword when modifying grouping for independent pools.

This covers the upstream **scheduler draft-group annotation** defect. It does
not replace these distinct local guarantees:

- `GPUModelRunner.initialize_attn_backend` excludes draft-owned KVarN layers
  from target KVarN metadata builders using `_draft_attn_layer_names`.
  KVarN builders own mutable tail/flush state; `is_eagle_group` does not
  isolate those builder instances.
- `_attn_group_iterator` includes `drafter.draft_attn_groups` when constructing
  zeroing metadata, so draft-only allocations remain initialized.
- `SpecDecodeBaseProposer._update_positions_dependent_metadata` replaces
  `seq_lens_cpu_upper_bound` with a new tensor rather than incrementing an
  aliased target/prior-step tensor in place. The target still has the old
  in-place operation.
- CPU block-table mirror propagation and KVarN's exact
  `committed = seq_len - query_len` handling remain required. A CPU upper
  bound containing rejected speculative tokens cannot decide flush eligibility.

The target annotator explicitly documents that trailing caches spanning
multiple groups still require generalization. Do not infer universal MTP
ownership support from the upstream commit. Existing local restrictions on
bundled Qwen MTP1/2 and the supported draft cache combinations remain justified.

### 4. Scheduler/checkpoint refactor — ADAPT the pool wrapper around upstream

`2e3e9900308a8cb13d8fe3234d7fec98b8b5eced` adds `prefill_end` to
coordinator and single-type allocation estimates. The target manager computes
`max(request.num_prompt_tokens, request.num_tokens - 1)` and passes it to
both full-sequence and per-step allocation checks. Under dense retention the
coordinator resets it to zero; under sparse retention Mamba reserves an internal
prefill checkpoint only for the relevant final chunk. This prevents incorrect
reservation and prompt-boundary eviction.

The local replacement `get_num_blocks_to_allocate_by_pool` must gain the same
`prefill_end=0` parameter and forward it to non-cross-attention managers;
the integer-return compatibility wrapper must forward it too. Preserve
`if self.retention_interval != 0: prefill_end = 0` in the actual per-pool
implementation. Keep the target manager's two keyword arguments when changing
its calls to the pool-aware method. A clean textual replay that omits these
steps can either raise `TypeError` or quietly lose upstream's reservation fix.

Also keep these upstream changes instead of replacing Mamba logic with the
old implementation:

- `f30a195bbb15b920d9c2c40e6a3466d8961ab101`: cap predicted physical Mamba
  demand with `min(num_new_blocks, physical_block_cap)` rather than replacing
  an already smaller estimate with the cap.
- `7a7283a0a25cd44fef4c7f326ad8a65f1244fd96` and
  `2e3e9900308a8cb13d8fe3234d7fec98b8b5eced`: retain prompt-end checkpoints
  under sparse retention.
- `d5051abaf19a1cce19159756a79d27ccb0b2fccf`: restore MTP prompt-tail hits.
- `49ee12d742b4a5524d2a3a3ee6c3fb77e4913073`: rename fine-grained Mamba
  prefix cache settings to shared-prefix checkpoints.
- `1dc2d854c12099f1f08233c886368a8c7baf430f`: allow zero-new-token slot
  allocation while adopting a nonzero computed prefix for bounded replay.

These are accepted upstream functionality, not KVarN hunk replacements. The
production KVarN configuration disables prefix caching; keep the fail-closed
guard in `KVCacheManager` against independent pools plus prefix cache/events,
and the KV connector compatibility rejection. Prefix checkpoint fixes do not
make cache events, connectors or prefix lookup pool-qualified.

### 5. Spec-aware block-size API — ADAPT and simplify

`4fb767ff0a0e0aae9dd2720c47245d0202136dfd` extends
`AttentionBackend.get_supported_kernel_block_sizes(kv_cache_spec=None)` and
passes specs from `Attention._largest_kernel_block_within`. The local KVarN
override accepts no arguments and infers tile size from global target config.

Minimum change: accept the optional `KVCacheSpec`; when it carries a KVarN
quant mode, select the preset's group directly from that mode. Retain the
no-argument/global fallback for callers which still omit the spec. This
reuses upstream's explicit per-layer input and avoids inferring a draft layer's
geometry from a target-only setting. Keep `get_preferred_block_size` behavior
for no-spec initialization. A positional-spec call must not raise `TypeError`.
Do not broaden supported layouts while making this interface adjustment.

### 6. Runtime state and stability — KEEP

The target has no KVarN allocator generation, per-builder tail state, native
bindings, flush staging ring or reset logic. Retain profiling-cache versus
process-state resets, cache-owner identity checks, bounded scratch allocation,
native eligibility predicates, exact flush boundaries and discarded-tail
overwrites. Generic upstream graph/metadata changes do not substitute for them.

`e4340e41c9a32b2247cc2442b35aa91b081c424f` adds XPU batch invariance,
but `XPUPlatform.check_and_update_config` explicitly rejects quantized model
weights and cache dtypes outside `auto`, `float16`, `bfloat16`. It does not
cover the W4A16/K4V2 scoped stability requirement. Keep the local scoped policy
and functional RMSNorm dependency pending its dedicated v03 audit. Historical
whole-model AEON invariance failures remain a limit, not a passed guarantee.

## Performance and Model Runner V2

The target improves MRV2 Mamba metadata reuse in
`dd3bc9c72faf410043b453e18e58553a1ad4d388`: `gpu/attn_utils.py`
caches reusable metadata for matching **MambaSpec** and builder type. It
does not operate on KVarN attention metadata or establish safe shared mutable
tail ownership. `dcfc17e0b1cb4a4bb4eeca9a75c82b82d776f25e` enables XPU
graphs by default, but the local compact KVarN backend still requires eager
V1, exact host-side lifecycle decisions and its dedicated native path.

Therefore retain both the platform and worker-local V2 guards for this release.
Do not enable DFlash or graphs merely because target upstream gained more
support. V2/DFlash is a separate port/qualification effort, including metadata
ownership, native ABI, draft memory, graph capture and quality/performance
comparisons. Switching to unquantized KV to enable V2 changes supported long
context capacity and is not a free optimization. No new performance win is
claimed here. The spec-aware geometry simplification is maintainability work.

## Dependencies and validation

The feature depends on native kernel `193cee080d654ff67c4f8bfaf924201bc88b9cf5`
and its full set of KVarN bindings. GDN metadata/scheduling changes in
`cfbd127232d3121e185eac2d24a9a50ea8141aa0` and
`6efe5e7cc59de64ae8d3ec67bbe77379be34d2c9` must be reconciled with their
own audits and paired GDN native behavior. Scoped request stability calls the
functional helper from `d909b25f0119419481bc378c3c0a3c656f9b76b0`.
The allowed BF16 draft configuration relies on explicit unquantized dtype
support `984e6c3edda05c4f2dca37703534cb0e69cfa0fa` or an equivalent retained
implementation. There are no later local commits after the assigned commit.

Observed during this audit: frozen-object source/diff/history inspection,
absence searches, interface and call-path comparison. No runtime tests or GPU
benchmarks ran. The prior release recorded successful native/reference tests,
MTP/vision/concurrency/long-context/cancellation API checks and 346880 usable
tokens, with explicit AEON and unused-BF16-NaN limitations; those results cannot
qualify a new base.

Required checks after parent adaptation:

1. Run existing independent-pool tests, particularly overlapping local block
   IDs, exhausted-pool admission without mutation, free/reuse, limiting-pool
   usage and group-qualified recurrent zeroing. Run worker ownership,
   `KVBlockZeroer`, copy and warmup suites, including legacy single-pool CPU
   and V2 consumers.
2. Run `test_kvarn_capacity_override_preserves_all_recurrent_slots`, hybrid
   independent sizing, context auto-fit, natural page alignment, and
   compatible-layer grouping cases in `tests/v1/core/test_kv_cache_utils.py`.
   Preserve target HiSparse and packed-stride tests and new MTP annotation
   cases. Run upstream partial-prefix-cache/Mamba checkpoint tests to detect
   lost `prefill_end` or physical-cap logic even though production disables
   prefix caching.
3. Extend the existing KVarN/config test files to pass a quantized spec to the
   backend block-size method explicitly (including a preset different from
   global target config), and check no-argument behavior. This distinguishes the new API compatibility
   from merely preserving the old production setting.
4. Run exact-history and ownership regressions: proposer CPU bounds must not
   mutate target tensors; draft builders must own only draft layers; MTP
   lifecycle flush must wait for committed page fill; profiling/reset/reuse
   must not retain prior-model cache owners.
5. Run installed native KVarN suites, including
   `test_dpas_mtp_verify_masks_future_and_overwrites_rejected_tail` around
   lengths 383/384/385, queries 2/3, K4V2/K4V4 and long history, plus bounded
   materialization and native dispatch checks. Verify paired schemas load.
6. Qualify the actual Brutus model through health/model checks, chat/reasoning,
   images/tools, active MTP acceptance, concurrent mixed prefills, cancellation
   across a flush and fresh reuse. Check capacity, long contexts and DRM
   errors against the recorded baseline. Performance claims require matched
   measurements; successful health/API checks alone do not show a speedup.

Parent replay guidance: preserve the feature as one coherent surviving change;
fold these compatibility adaptations into it. No supported feature hunk is
recommended for unconditional deletion on the evidence above. Preserve new
upstream implementations in overlap areas and reapply only the residual pool,
format and lifecycle behavior described here.

## Integration completed in isolated worktree

At parent request, the audit owner subsequently integrated the feature in
`build-dev/release-v031-20261010/vllm`, on parent
`1e1d1471b77014e3a54c7da59b3cf42487ceeaf4`. Final feature commit:
**`ef92dc5600af64103c01f97e3300b81c3a1083ee`**. Worktree was clean at
handoff. Feature diff: **75 files, 20813 insertions, 371 deletions**.
Authorship, co-author and sign-off were preserved; the message now describes
v0.31 adaptations and current validation rather than inherited v0.30 claims.
The paired native dependency is
`dbb684f4c9b38a8b3bbf52bda0c490ee47049221`, supplied by the parent.

Implemented the per-pool `prefill_end` forwarding and retention rule;
spec-aware KVarN block sizes and six preset regression cases; new upstream
global-invariance branches alongside scoped Gemma RMSNorm behavior; and the
audited upstream allocator/grouping metadata. Updated local test mocks for
`num_prefill_lookahead_tokens`, TP size and `_find_non_ssm_backends`. One existing
prefix-split mock also needed the retained GDN scheduler's
`mamba_prefill_alignment` and `needs_mamba_cache_alignment` fields.

**Additional demonstrated deletion during integration:** upstream
`8b1d188046034b42024d22ee52fa6c3948b4acbd` separates request-level HMA
load failures from single-group block failures. Its scheduler no longer reads
`null_block_id` in the old failure loop. The old local `BlockPool` import,
single-pool assertion and null-ID lookup at that location were dropped; they
would restore unused code. The earlier report's no-unconditional-deletion
conclusion concerned feature behavior; this obsolete compatibility hunk is
now explicitly accounted for.

Static validation passed:

- `/nix/store/z2078an0s17i6phm99np502fzj5iybd8-ruff-0.16.2/bin/ruff check`
  and `format --check` on all 73 changed Python files.
- Python in-memory `compile()` for the same 73 files.
- `git diff --check` and clean worktree after commit.

Source CPU checks used the previous release's dependency closure at
`/nix/store/zhv84h5afvyx99fkw0l56rpj8rldhivl-v030-brutus-qualification-env/bin/python`
(Torch 2.13.0+xpu), with `PYTHONPATH=.` pointing to the isolated worktree,
`VLLM_TARGET_DEVICE=cpu`, empty `CUDA_VISIBLE_DEVICES`/`ZE_AFFINITY_MASK`,
`HF_HUB_OFFLINE=1` and `PYTHONDONTWRITEBYTECODE=1`.
`vllm.__file__` was confirmed inside the v0.31 worktree. No native/GPU
qualification is implied by these CPU checks.

| Source CPU checks | Observed result |
| --- | --- |
| Independent pools plus upstream checkpoint-reservation matrix | 65 passed, 57 deselected |
| Full `test_kvarn.py`, independent pools, partial-prefix suite, pool ownership, block zeroer | 338 passed, 14 CUDA skips; one stale scheduler mock failed and passed after repair |
| Cache utils selected by `kvarn or independent or eagle or packed or alignment or hisparse or mtp or replicas or trailing_layer` | 86 passed, 127 deselected after mock repairs |
| Scoped request-stability, proposer, warmup selection | 104 passed; three unrelated model-runner tests selected too broadly could not set up without `dist_init` |
| Exact target/draft owner, lifecycle cleanup nodes and full platform beta guards | 78 passed |
| KVarN config defaults/rejection and repaired prefix split | 4 passed (three config cases and one prefix split) |

Counts overlap; they are not an aggregate unique-test total. Initial ordinary
pytest startup could not import `tests/conftest.py` because the old closure
lacks `tblib`. Runs therefore used `--noconftest -p no:cacheprovider`.
The cache-utils logging cases used the **unchanged**
`temporary_enable_log_propagate` and `caplog_vllm` fixture definitions, extracted
by Python AST from the current `tests/conftest.py` into an in-memory pytest
plugin. No assertions were removed or mocked out. Broader distributed runner
tests remain for an environment providing their real fixtures.

Concrete source CPU rerun list for the new dependency closure:

```text
tests/quantization/test_kvarn.py
tests/v1/core/test_independent_kv_cache_pools.py
tests/v1/core/prefix_cache/test_partial_prefix_cache_hits.py
tests/v1/worker/test_kv_cache_pool_ownership.py
tests/v1/worker/test_kv_block_zeroer.py
tests/model_executor/layers/test_request_stable_linear.py
tests/v1/spec_decode/test_llm_base_proposer.py
tests/v1/worker/test_gpu_warmup.py
tests/platforms/test_xpu_kvarn_beta.py
tests/v1/worker/test_gpu_model_runner.py::test_target_kvarn_builder_does_not_own_draft_cache
tests/v1/worker/test_gpu_model_runner.py::test_kvarn_cleanup_releases_target_and_draft_cache_owners
tests/test_config.py::test_kvarn_uses_v1_model_runner_by_default
tests/test_config.py::test_explicit_v2_model_runner_rejects_kvarn
```

Also run `tests/v1/core/test_kv_cache_utils.py` with the selection above,
including upstream logging fixtures. The native/GPU tests and real host
qualification listed in the audit remain pending with the parent. Do not
mistake passing source tests against old Torch/Triton dependencies for the
new packaged runtime loading or serving successfully.
