# v01 — keep exact live GDN request-index extents

**Disposition: KEEP `3ecc9a1d4dcf6ce530bc61a45af9bdabf7e450d6` unchanged.**
The frozen v0.32 XPU adapter is AST-identical to the unpatched v0.31 adapter.
Upstream improves pure speculative metadata construction but still exposes
padded request arrays through other producer/staging paths. The retained
native kernel requires exact live request extents. Keep the five guarded
slices and their existing regression test, including V1 fallback support.

Auditor: `/root/audit_v01`, 2026-10-20 UTC. Read
`.agents/skills/vllm-xpu-update/references/commit-audit.md` and
`/home/jasonbk/Projects/vllm/AGENTS.md`. No source, Git, service or GPU state
was changed; this report is the only written artifact.

## Frozen inputs

| Input | Identity |
| --- | --- |
| Assignment | v01, `fix(xpu): strip graph padding from GDN index tensors` |
| Commit | `3ecc9a1d4dcf6ce530bc61a45af9bdabf7e450d6` |
| Parent / old upstream | `db9527a46873454610df6dbedf79a36d6bf1a7f6` |
| Recomputed stable patch ID | `bf72801d5ce616733d6a1102ec4605bcc310276e` |
| Old complete vLLM tip | `dcba4007ef6af8ba143d2794c194c36b58397339` |
| Selected vLLM | v0.32.0, `5df63a7203cdb9328d824a27b682e77c7b9928be` |
| Kernels old/new base | `ddf336d86e3c8602888572a3502f951abd51df12` |
| Kernels complete old tip | `dbb684f4c9b38a8b3bbf52bda0c490ee47049221` |
| Inventory | `../inventory.json`, run `release-v032-v2-20261019` |

Verified complete ordered histories and parents against the frozen inventory.
vLLM has eight local commits: `3ecc9a1d4d`, `490bf7e422`, `41ab2f1882`,
`1e1d1471b7`, `ef92dc5600`, `81dc387cf9`, `a76fbb1ce0`, `dcba4007ef`.
Kernels has seven: `dcbbeac2f2`, `7a1af66939`, `2322b83090`, `7bc80d1569`,
`c75a699930`, `2e7f76c799`, `dbb684f4c9`. Full immutable SHAs, parents and
patch IDs remain in the inventory. Prior rationale was read in
`docs/updates/release-v031-20261010/audits/v01-gdn-padding.md` and checked
against the selected 0.32 source rather than assumed to remain true.

## Requirement and final old behavior

`vllm/_xpu_ops.py::_gdn_attention_core_xpu_impl` is the boundary between
GDN metadata and `torch.ops._xpu_C.gdn_attention`. Request padding must not
be interpreted as additional live requests, and rollback columns must not
be confused with request rows or token counts.

| Tensor | Exact native-boundary extent |
| --- | --- |
| `non_spec_query_start_loc` | `num_prefills + num_decodes + 1` |
| `non_spec_state_indices_tensor` | `num_prefills + num_decodes` |
| `spec_query_start_loc` | `num_spec_decodes + 1` |
| `spec_state_indices_tensor` | `num_spec_decodes` rows; all rollback columns |
| `num_accepted_tokens` | `num_spec_decodes` |

The commit retains views of the live prefix with `None` guards. It does not
change projected/output tensor extents, token-index tensors, source values,
or `num_actual_tokens`. Its non-spec `.contiguous()` predates this change.
All five slice statements and the five-case adapter regression are unchanged
in the complete old stack. v02 adds the independent mixed-dispatch predicate
after the slicing; the later KVarN and host-offload changes do not supersede it.

The supported baseline is Qwen3.8 hybrid GDN on Brutus's Intel Arc Pro B70,
including eager V1, KVarN/K4V2, MTP and retained prefix caching. New V2/DFlash2
qualification must preserve this native contract whenever it uses the XPU
GDN path. This is a correctness/compatibility safeguard, not a throughput
claim or proof that every eager request has padded metadata.

## Upstream coverage at the frozen target

The old-to-new `vllm/_xpu_ops.py` diff adds normalization wrappers and changes
FlashAttention handling but makes **no change** to
`_gdn_attention_core_xpu_impl` (target lines 178–270). A stdlib AST comparison
confirms equality of the complete function between old upstream and target.

Relevant upstream commits and alternate paths:

- `87a4bf664f4b2344eced5a10add977b750d4b7d5`, **Slice pure spec-decode rows
  instead of a host-mask gather** (#58763), changes
  `GDNAttentionMetadataBuilder.build` and `update_block_table` in
  `vllm/v1/attention/backends/gdn_attn.py`. Pure-spec state rows and accepted
  counts now use prefix slices rather than a host-mask gather. This covers
  the immediate pure-spec construction, but not the full boundary contract.
  Later in `build`, `batch_size = m.num_reqs`; `_stage_spec_decode` exposes
  state/accepted arrays of that padded count and query arrays of count + 1,
  filling inactive suffix entries. `_stage_decode` similarly exposes padded
  non-spec arrays. Thus producer slicing does not remove the adapter need.
- Non-spec `build` still assigns `block_table_tensor[:, 0]` and
  `query_start_loc` without trimming after subtracting trailing zero-query
  requests from `num_prefills`. The mixed-spec branch allocates non-spec
  query/state arrays using the full complement of the spec mask, which can
  include suffix padding excluded from the live prefill count. These are
  additional reasons not to drop normalization merely because pure-spec
  construction improved.
- `0eac15270710ddb7f5296d7244fcd0496c2daa0f`, **Route every
  speculation-capable row through the speculative path so recurrent state
  stays consistent** (#56531), keeps zero-draft rows on the speculative path
  to apply accepted-token offsets. This fixes recurrent-state routing, not
  padded index sizes. Preserve it and cover zero-draft rows in the final
  speculation qualification.
- `b538d807ffe0976af7e5d107ccfe77e676b83b70`, **Keep GDN prefill checkpoint
  metadata local to each cache group** (#59536), and
  `3e2a7e74a556623ddee3a327299d7bf04055501e`, **Enable KDA prefill checkpoints
  for GLM-5.3-Flash** (#56960), add/checkpoint or regather other metadata.
  They do not normalize the XPU native call's five tensor extents.
- `548597367e66dcc90e24576c08e0e25f88c9ee6c`, **Add AITER FlyDSL GDN
  prefill backend** (#57560), changes the ROCm prefill path. It does not
  replace XPU dispatch. At target,
  `qwen_gdn_linear_attn.py` still assigns `self.forward_xpu` for XPU at
  lines 487–488; `forward_xpu` calls
  `torch.ops.vllm.gdn_attention_core_xpu` at line 1086.
- V2 `vllm/v1/worker/gpu/model_states/mamba_hybrid.py` still supplies
  `GDNAttentionMetadataBuilder` through
  `MambaHybridAttnMetadata.get_extra_attn_kwargs`; its incoming accepted and
  draft arrays are sliced to caller `num_reqs`, not a proof of each native
  subgroup's live extent. Moving to V2 does not supply an alternate XPU GDN
  implementation that makes this requirement disappear.

At the unchanged kernel base,
`csrc/xpu/gdn_attn/gdn_attn_interface.cpp::causal_conv1d_spec` still has exact
checks for query length (line 69), state rows (line 93) and accepted length
(line 116). `causal_conv1d_non_spec` likewise requires query length
`num_prefills + num_decodes + 1` and state length
`num_prefills + num_decodes`. The wrapper `gdn_attention` does not normalize
these request arrays. The full kernel local delta does not relax these
checks. Its token-capacity relaxation, already in the selected old/new base,
does not change request cardinality.

## Minimum residual and dependencies

Keep the existing 14-line adapter addition and existing regression test.
No new operator, schema, cache layout or cross-repository API is introduced.
There is no mandatory companion fork patch for this normalization, and no
later repair needs to fold into v01.

`490bf7e42287a65f1d7dfa83ab8100e36d8f006b` (v02) and
`7bc80d1569bedeb8f4adceb43524fbc1b4d07516` (k04) separately own the
`split_mixed_non_spec` keyword and its paired native implementation. Keep
their disposition independent; moving code beside the slices must preserve
those slices. KVarN's features in v05–v08 and k07 do not justify dropping the
native-boundary behavior. Preserve original authorship and commit trailers.

No blocker was found for replaying v01. Any future producer redesign could
retire these slices only after proving exact live extents for every supported
producer, including V1 fallback and graph staging. Padding must remain a
suffix: these slices do not repair interior inactive rows, invalid counts or
an unrelated `has_initial_state` mismatch.

## Validation observed and remaining

Read-only checks performed:

- Verified both complete frozen histories and parents; recomputed v01's
  stable patch ID with `git show --format= ... | git patch-id --stable`.
- Inspected old upstream, old final, frozen target, upstream changes to the
  adapter and metadata builder, V2 metadata plumbing, XPU dispatch, and the
  native kernel base/final interface delta.
- Ran stdlib AST checks with the candidate checkout's `.venv/bin/python -B`:

  ```text
  PASS: frozen 0.32 GDN adapter AST identical to unpatched 0.31 adapter
  PASS: assigned five-case tensor-boundary regression unchanged at old final tip
  PASS: all five original None-guarded slices preserved at old final tip
  ```

The prior v0.31 audit recorded a CPU-Torch sensitivity check of the existing
five-case regression: unpatched upstream failed the four present-metadata
cases and passed all-None; old patched final passed all five. That is prior
recorded evidence, not a new 0.32 runtime result. The local historical Nix
qualification environment was unavailable, so no new Torch/native/GPU test
was run for this bounded audit. The AST equality shows the unpatched adapter
behavior itself has not changed; it does not qualify a complete 0.32 build.

Required parent-owned validation:

1. Run `tests/v1/attention/test_gdn_metadata_builder.py`, including
   `test_xpu_gdn_adapter_trims_only_request_padding`, in the rebuilt package
   environment. Keep the storage-sharing, unchanged data tensor identity,
   all rollback columns, all-None and zero-live-count assertions.
2. Native XPU comparison of padded vs unpadded metadata: non-spec, mixed and
   speculative batches; shortened queries, zero-draft rows, accepted-token
   rollback, inactive suffix sentinels and state reuse. Run capture/replay
   if the selected serving profile enables graphs.
3. Qualify the actual V2/DFlash2 + KVarN model/configuration and retain a
   V1/MTP smoke gate. Passing this boundary test alone does not prove V2
   cache ownership, speculative rollback or model correctness.

There is no new performance optimization to benchmark in this view-only
fix. Full paired release correctness/performance gates remain required.
The parent must record the replay SHA in the reconciliation map.
