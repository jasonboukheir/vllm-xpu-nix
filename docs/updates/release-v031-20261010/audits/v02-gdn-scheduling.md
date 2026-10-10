# v02 — Qwen GDN prefill grid and mixed scheduling

**Disposition: KEEP**, paired with kernel `9a17a2c8617be2b0471db79fb301912601ee945c`.
The selected upstream pair does not supply either retained arithmetic property.
Retain the production behavior and adapt the multimodal test fixture to the new
upstream `ModelConfig.supports_multimodal_inputs` interface. No production hunk
is demonstrated redundant. This recommendation is source-audit evidence, not
qualification of the new binaries or a whole-model determinism claim.

## Frozen inputs and instructions

| Input | Identity |
| --- | --- |
| Owner / run | `/root/v02_gdn_scheduling` / `release-v031-20261010` |
| Assigned commit | `6efe5e7cc59de64ae8d3ec67bbe77379be34d2c9`, `fix(xpu): stabilize Qwen GDN prefill and mixed scheduling` |
| Parent | `cfbd127232d3121e185eac2d24a9a50ea8141aa0` |
| Verified stable patch ID | `b1f78180963d8928e113c17f5418afc7a63517c7` |
| vLLM old base / old final | `ced6857afa0ea7b2e3f0846a62e1394e90f15607` / `466b9d5abe84f13c29b38c686e4d77fc09e09aa4` |
| vLLM target | v0.31.0, `db9527a46873454610df6dbedf79a36d6bf1a7f6`, fetched as `refs/upstream-releases/v0.31.0` |
| Kernels old base / old final | `1c7cbeee1cd0c1481d48f5031f679a47f3f0ef45` / `193cee080d654ff67c4f8bfaf924201bc88b9cf5` |
| Kernels selected source | release/0.1.15.4, `ddf336d86e3c8602888572a3502f951abd51df12`; published-wheel/source mapping is owned by the parent |

Read the local commit-audit runbook and `../vllm/AGENTS.md`. Verified the ordered
old inventories with `git log --reverse old_base..old_final` against
`docs/fork-delta.md`: vLLM `cfbd127232`, `6efe5e7cc5`, `d909b25f01`,
`984e6c3edd`, `466b9d5abe`; kernels `98bdbfd`, `e1bec3b`, `956b61f`,
`3b5deb0`, `9a17a2c`, `30fda95`, `8b6ec9d`, `193cee0`.

Prior rationale was read from packaging commit
`8c72895340deafef77998a579335d078800626ab`, artifact
`docs/updates/release-v030-20260922/audits/v03-7de1e1c5ec5f.md`, plus the
published `docs/releases/xpu-v1.11.0.md`. That prior report gives the same
stable patch ID and the historical regression evidence. Its historical raw
numerical logs were not independently rerun or rehashed in this audit.

## Requirement and contribution in the final old stack

On native Xe2 Qwen GDN, changing a prefill's chunk boundary changes arithmetic,
and adding a prefill to a cached one-token decode otherwise sends that decode
through a different arithmetic implementation. The patch retains two scoped
properties independent of the attention KV format:

1. Non-final XPU Qwen GDN prefills end on an absolute 64-token grid. For prompt
   lengths `[127, 4095, 4095, 127]` and an 8192-token budget, the partial third
   prompt receives 3968 instead of 3970 tokens; its final 127-token tail remains
   permitted. This applies only to cache mode `none`, no speculative tokens,
   no multimodal support and no encoder-decoder model. A budget smaller than
   `64 + max_num_seqs - 1`, or a positive long-prefill threshold below 64,
   fails closed to prevent starvation. Cache checkpoints cannot bypass this
   independent arithmetic grid.
2. `_gdn_attention_core_xpu_impl` opts into native `split_mixed_non_spec` only
   for mixed ordinary batches with one token per decode, contiguous decode-first
   packing, exact non-spec token totals, no speculative decodes, no token
   indirection and no speculative masks. The paired native implementation then
   preserves the standalone cached-decode route.

The distinct `QWEN_GDN_ATTN` identity scopes the grid and is also consumed by
the later KVarN request-stability code. Subsequent local commits neither revert
nor replace these behaviors: AST comparisons verified the adapter and
`Scheduler._mamba_block_aligned_split` are identical between this commit and
the old final tip. Later KVarN changes add CPU metadata and pool ownership.

The published Brutus configuration uses Intel Arc Pro B70, RedHatAI/Qwen3.8-27B-INT4,
BF16 compute, eager V1, K4V2 target/draft, MTP2 and vision. Consequently its
production profile does **not** activate the no-spec/non-multimodal scheduler
grid. This remains a retained supported configuration and an input to the
optional request-stability policy, not a newly proven requirement for every
Brutus request. The mixed adapter predicate can operate on eligible ordinary
batches even in a model configured for speculation; it excludes batches with
actual speculative metadata.

## Upstream coverage and alternate implementations

All target-source conclusions below refer to the frozen SHAs above.

| Evidence | Coverage conclusion |
| --- | --- |
| `vllm/_xpu_ops.py`, `_gdn_attention_core_xpu_impl` | Old upstream and v0.31 implementations are AST-identical. The native call still ends with `reorder_input`; no split predicate or keyword exists. |
| `vllm/v1/core/sched/scheduler.py`, initialization and `_mamba_block_aligned_split` | Upstream still enables splitting only for `has_mamba_layers and mamba_cache_mode == "align"`, and uses `cache_config.block_size`. Mode `none` receives no arithmetic-grid constraint or starvation guard. |
| `vllm/v1/attention/backends/gdn_attn.py`, `GDNAttentionBackend`; `registry.py`; `gdn/base.py`, `mamba_type` | Generic GDN identity remains; no Qwen-specific backend or `get_required_prefill_chunk_size` contract exists. |
| `vllm/platforms/xpu.py`, `update_block_size_for_backend` | Generic GDN still rounds cache block size to 64. Cache sizing does not cause the scheduler to align mode-`none` prefills. |
| `qwen_gdn_linear_attn.py`, initialization and `forward_xpu` | XPU still selects its native custom-op path. Generic CUDA/Triton mixed splitting is bypassed, as it was on the old base. |
| Kernels `gdn_attn_interface.cpp`, `gated_delta_rule_non_spec` | For Xe2, any `num_prefills > 0` selects `chunk_gated_delta_rule_xe2` for the entire non-spec group; decode-only uses `gdn::gated_delta_rule`. The source target therefore retains the failure mechanism addressed by the split. |
| Kernels `gdn_attn_interface.cpp`, `gdn_attention`, and `torch_bindings.cpp` | The legacy fused entry point still chains unsplit non-spec conv/delta ops; its schema lacks the optional split flag. |

Relevant upstream changes inspected, beyond patch-ID/conflict checks:

- `79468c20ef23227d4e051f807e10fd54fb24e24f` (#51565) changes
  `GDNAttentionMetadataBuilder.build` so a one-token first chunk with no prior
  state is a prefill, and excludes trailing padding from prefill counts.
  This fixes fresh-state classification, not arithmetic switching for cached
  decodes mixed with prefills. Preserve it and exercise a one-token fresh
  prefill joining cached decode in qualification.
- `dd3bc9c72faf410043b453e18e58553a1ad4d388` (#58762) reuses GDN metadata
  across MRV2 KV groups, but `supports_update_block_table` requires
  `device.type == "cuda"`. It neither replaces XPU dispatch nor offers an XPU
  simplification of these changes.
- `975dca5bb5db302077674cfa9afe851ee700ad73` (#57140) scatters generic mixed
  speculative outputs into the caller buffer. `forward_xpu` does not take
  that generic path; this is also a different batch class than the ordinary
  no-spec split being retained.
- `c64b15cde583c7dd3cbb906b30164131a030d6b4` (#57951) removes the
  long-prefill cap for a lone request; `6697a7dd30ceebbe69a9d37e4012d11d40f79685`
  (#58459) adds adaptive fair-share thresholding. Neither aligns arbitrary
  shared-budget endpoints to 64. Keep these upstream improvements; they do
  not justify removing the minimum-budget/threshold guards for the supported
  multi-request grid profile.
- `c8824a1e42bbd674751f808d657d58024c68fb0f` (#57913) moves multimodal
  support detection from `MultiModalRegistry` to `ModelConfig`. The production
  `supports_mm_inputs` local still exists, but `_make_xpu_gdn_scheduler` must
  mock the new model-config property instead of the removed registry method.
- `49ee12d742b4a5524d2a3a3ee6c3fb77e4913073` renames fine-grained prefix
  cache controls to shared-prefix checkpoint controls. Preserve the target
  names and checkpoint semantics when replaying; they remain separate from
  mode-`none` arithmetic alignment.
- `e4340e41c9a32b2247cc2442b35aa91b081c424f` (#55881) adds XPU batch
  invariance, but its platform validation rejects quantized model weights and
  quantized KV caches. It does not change this native GDN path and is not an
  equivalent replacement for the W4A16/KVarN supported configuration.
- Kernel `f4b729ae7fb3eb22c564606ec8b4d184c7c8bc4a` (#600) fixes ragged
  speculative traversal. In `gdn_attn_interface.cpp` the selected old-to-new
  delta only relaxes the speculative-capacity assertion; the mixed non-spec
  dispatch and schema are unchanged. The other three selected kernel commits
  (`d09702a637cfea2a0c02909af82bd789d26ea319`,
  `885b120ffa082312b7d93c2be7a1e3b3dd79672f`,
  `ddf336d86e3c8602888572a3502f951abd51df12`) do not cover this requirement.

## Minimum residual and paired dependencies

Keep the existing narrowly scoped grid, backend contract and mixed-dispatch
predicate. The enum, Qwen model override, XPU block-size recognition and
`derive_mamba_conv_split` connector support are necessary consumers of the
distinct identity. Removing them piecemeal would strand supported paths.
Only test-context adaptation is demonstrated necessary; no new generic
framework or broader grid activation is warranted by the upstream changes.

- Hard paired dependency: kernel `9a17a2c8617be2b0471db79fb301912601ee945c`
  supplies trailing `bool split_mixed_non_spec=False` in native declaration,
  schema and implementation. Keep both sides together and preserve the default
  false ABI for existing callers. New upstream kernels alone reject the Python
  keyword.
- Adapter metadata normalization `cfbd127232d3121e185eac2d24a9a50ea8141aa0`
  touches the same function but is independently required/reviewed.
- KVarN `466b9d5abe84f13c29b38c686e4d77fc09e09aa4` consumes
  `QWEN_GDN_ATTN` in `XPUPlatform.set_additional_forward_context`; removing the
  identity requires a coordinated replacement of that consumer.
- Kernel convolution workaround `956b61f95a0b38e73eeebe9cbfeff1be782a9283`
  and ordering fix `3b5deb0773be875a328a1831c5e4a08233460d32` are distinct
  correctness requirements. This audit does not imply they are superseded.
- No later local repair/revert needs folding into this original commit.

## Validation observed and proposed

Observed in this audit: immutable inventory/parent/patch-ID checks; whole
old-to-target source delta inspection for the touched paths; target native
schema and alternate call-path inspection; successful standard-library AST
checks through `../vllm/.venv/bin/python` showing (a) upstream adapter
old-to-new identity, (b) local adapter assigned-to-final identity, and
(c) local scheduler split assigned-to-final identity. These are source checks,
not runtime regression tests. No source worktree, ref, lock, service or GPU
was changed. No new pytest or model inference was run.

Historical evidence read from the prior audit: original mixed arithmetic had
3383 output mismatches with maximum absolute difference 0.001953125; the
split control passed. Noncanonical `3970+125` reported 99121 output and
274112 SSM-state mismatches; `3968+127` was the canonical control. The prior
release separately passed operational Brutus mixed/MTP/vision checks while
not establishing full-model exact invariance or a performance comparison.
These historical results must not be relabeled as v0.31 results.

Required on the reconciled candidate:

1. Run existing scheduler grid, backend selection, native adapter predicate,
   prefix-cache and NIXL HMA tests. Verify 3968/127 boundaries, final tails,
   67-token budget progress with three decodes, threshold/budget rejection,
   and unchanged generic/non-XPU/speculative/multimodal behavior. Adapt the
   multimodal fixture and preserve upstream shared-prefix checkpoint names.
2. Run upstream GDN metadata first-chunk/padding cases together with the adapter
   cases, including cached decode plus fresh one-token prefill. Check the
   actually loaded native operator schema and exclude indexed/speculative
   batches from the ordinary split.
3. On paired Xe2 binaries, compare standalone cached decode against mixed
   execution for output, z, convolution state and SSM state. Exercise fresh and
   dirty allocation, canonical boundary carry, slot reuse and joining/leaving
   batches. For upstream-only comparison, call supported native split conv/delta
   ops directly; an unsupported-keyword error is not numerical evidence.
4. Qualify the host's actual eager V1 K4V2/MTP2/vision profile independently of
   the optional grid policy. Measure matched no-spec mixed latency/throughput
   if making performance claims: alignment and extra native launches can cost
   throughput. No speedup or parity has been established here.

Unresolved gates are new-pair runtime/numerical qualification and the parent's
published kernel-wheel/source verification, not demonstrated upstream coverage.
Keep these requirements covered until those gates are resolved.
