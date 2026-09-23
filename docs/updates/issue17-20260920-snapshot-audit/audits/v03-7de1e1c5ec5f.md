# v03 — Qwen GDN prefill arithmetic and ordinary mixed batches

Disposition: **KEEP**. Both correctness requirements remain unmet at the frozen upstream pair. Retain the Qwen backend identity, narrowly scoped 64-token scheduling grid, and ordinary mixed-batch opt-in, together with companion k05's operator implementation. Reconcile scheduler/test context with newer upstream checkpoint handling; no demonstrated upstream coverage justifies deleting a production hunk. This decision preserves existing behavior and does not qualify new V2 execution.

## Frozen inputs and scope

| Input | Identity |
| --- | --- |
| Audit owner | `/root/audit_v03` |
| Assigned commit | `7de1e1c5ec5f3005b0e46bb0ec323f5e5864876c` — `fix(xpu): stabilize Qwen GDN prefill and mixed scheduling` |
| Parent | `cf0b834cd66c54156393e2f943382b1015360f84` |
| Stable patch ID, verified | `b1f78180963d8928e113c17f5418afc7a63517c7` |
| vLLM repository | `/home/jasonbk/Projects/vllm-xpu-nix/build-dev/issue17/vllm` |
| Old upstream base / main | `51da0ca66c8065619c79e35dff97aa99aeaf5644` / `3b6c822e8a4f56dc3904c7fce388a76dff4c7bdd` |
| Old released vLLM | `2fdd9d71f6895379b0940a8f755b4c9095840558` (`xpu-v1.10.0`) |
| New vLLM upstream snapshot | `e378275a8f20eac9e92212e4a1fee84ec5a31dc7` |
| Kernel repository | `/home/jasonbk/Projects/vllm-xpu-nix/build-dev/issue17/vllm-xpu-kernels` |
| Old kernel base / main | `e7f20cbc18d741419a84aa57178e1f1cc0b453da` / `9bb1e2d428b8a25b262c0cd6704614bdf6ee3ad8` |
| Old released kernels / new target | `2c39287deec5e59cc2656e6d4a6dce51a567c411` / `da16a5595c105605bcd44b558c7933b934a05f85` |

Read the run manifest, complete v01–v16/k01–k11 inventory, issue snapshot, pair-compatibility record, both source `AGENTS.md` files, update skill and commit-audit reference. The user permits these untagged upstream snapshots. This audit writes only this report; no source/ref/build/GPU/service/publication operation was performed.

## Requirement in the final old stack

The original target is native XPU Qwen GDN, BF16 arithmetic on Xe2, including Brutus/Arc Pro B70. Recurrent convolution/SSM state is separate from KVarN attention KV storage. Two independent batching changes can otherwise alter a request's arithmetic:

1. **Non-final prefill endpoints:** with four ragged requests `[127, 4095, 4095, 127]` and an 8192-token scheduler budget, the third request would receive 3970 tokens. The native prefill processes 64-token chunks. v03 gives Qwen its own `QWEN_GDN_ATTN` backend and `get_required_prefill_chunk_size() == 64`, then schedules that partial chunk at absolute endpoint 3968. A final 127-token tail is retained. Cache-boundary stops remain separate from arithmetic alignment, so internal checkpoint availability cannot bypass the arithmetic grid. A minimum budget of `64 + max_num_seqs - 1` prevents one-token decodes from indefinitely starving the next aligned prefill; nonzero thresholds below 64 fail closed.
2. **Cached decode mixed with prefill:** the ordinary native decode path must continue producing the same output and updated recurrent state when another request prefills. v03 sets `split_mixed_non_spec` only with both groups present, one token per decode, an exact non-spec token total, zero speculative decodes, and no token-index remapping or speculative mask. k05 then runs decode and prefill groups separately through their existing native arithmetic paths.

The scheduler grid is deliberately limited to XPU, Qwen backend, `mamba_cache_mode="none"`, no speculation, no multimodal inputs and no encoder-decoder model. Existing cache-alignment mode retains its own rules. The mixed-dispatch guard is distinct: it qualifies each forward's ordinary non-spec metadata. **Neither change establishes cross-partition equivalence for MTP/DFlash2 or enables a 64-token constraint for vision.** Do not extend those profiles incidentally during replay.

The final release retains both behaviors. Comparing v03 to the released tip shows no change in its adapter, Qwen identity or connector implementation. v05 `fcab69494f1719e91e416aace17ef6ad501fe900` adds immutable CPU metadata and independent-pool zeroing but does not supersede either fix. It also consumes `MambaAttentionBackendEnum.QWEN_GDN_ATTN` in `XPUPlatform.set_additional_forward_context` to select Qwen layers for request-stable dispatch. Thus deleting the identity would break an additional downstream consumer unless that consumer were changed with equivalent evidence.

## Upstream coverage and alternate implementations

All line references in this section are to the frozen upstream targets above.

| Inspected implementation | Finding |
| --- | --- |
| `vllm/v1/core/sched/scheduler.py:344–382`, `Scheduler.__init__` | `need_mamba_block_aligned_split` is enabled only by Mamba cache mode `align`; no XPU arithmetic requirement is queried. A 64-token cache block alone does not constrain scheduling in mode `none`. |
| `scheduler.py:422–535`, `_mamba_block_aligned_split` | Upstream clips for cache checkpoints and uses `cache_config.block_size`; it has no independent arithmetic alignment or associated minimum-budget validation. |
| `vllm/platforms/xpu.py:394–426`, `update_block_size_for_backend` | Upstream already rounds GDN cache block size to a multiple of 64, but only recognizes `GDN_ATTN`. It does not replace the scheduler fix, and must recognize the retained Qwen identity. |
| `gdn/base.py:51`, `backends/registry.py:217`, `backends/gdn_attn.py` | Qwen still inherits generic `GDN_ATTN`. Frozen-tree searches found no `QWEN_GDN_ATTN`, `get_required_prefill_chunk_size`, `mamba_prefill_alignment` or `split_mixed_non_spec`. |
| `qwen_gdn_linear_attn.py:405–406,977–1010`; `_xpu_ops.py`, `_gdn_attention_core_xpu_impl` | XPU selects `forward_xpu` and calls the same native adapter. Old-base-to-target `_xpu_ops.py` changes are docstrings; the split opt-in is absent. |
| Kernel `gdn_attn_interface.cpp:763–908`, `gdn_attention` | The legacy fused entry point still forwards all non-spec rows together through `causal_conv1d_non_spec` then `gated_delta_rule_non_spec`. Its schema/signature lacks the extra split argument. |
| Kernel `gdn_attn_interface.cpp:695–738`, `gated_delta_rule_non_spec` | On Xe2, the presence of any prefill sends the whole non-spec group through `chunk_gated_delta_rule_xe2`; decode-only uses `gdn::gated_delta_rule`. Upstream has not normalized those arithmetic paths or split ordinary mixed requests. |

Specific alternate solutions and changes were checked, rather than deciding from patch similarity:

- `fa27d4e9cf3c8d8a5a143f38c346b27c02b2c2e3` (#44700) already splits ordinary mixed Qwen requests in the generic implementation. An ancestry check confirms it predates the old vLLM base. The target `_forward_core` at `qwen_gdn_linear_attn.py:1405–1505` uses `split_non_spec` and a separate recurrent decode update, but the XPU `forward_xpu` bypasses that function. It therefore does not cover native XPU.
- `975dca5bb5db302077674cfa9afe851ee700ad73` scatters generic mixed speculative outputs into the caller buffer; `fbe8a157fbdd4b2f813ae6395397232de97d5c5d` enables the ROCm AITER layout path; `f6326f53bda46898a331c2d24500332c285d9a2b` enables FlashInfer prefill on SM12x; `d05da62e9ccdf8e342b15bf6785d83224cc165af` corrects XPU logging. None changes the native XPU branch or its numerical partitioning.
- `263c4ff95fadb62d80f315171b17d4f623494e56` fixes EAGLE cache checkpoints at resume positions. Its scheduler additions preserve fine-grained prefix-cache/junction stops; they do not turn on arithmetic alignment with caching disabled. Preserve those upstream changes when replaying v03, and update test stubs for the new checkpoint fields instead of restoring the old scheduler wholesale.
- The only old-base-to-target kernel change under `csrc/xpu/gdn_attn` is `da16a5595c105605bcd44b558c7933b934a05f85` (#600). It traverses actual ragged speculative query ranges and preserves rollback capacity, relaxing the speculative-token capacity check. It does not change ordinary mixed dispatch or Xe2 prefill arithmetic. Kernel split entry points already existed at the old base and still select chunk arithmetic based on `num_prefills > 0`.
- V2 remains on the same path: `MambaHybridModelState.prepare_attn`, `vllm/v1/worker/gpu/model_states/mamba_hybrid.py:232–333`, builds the selected backend metadata; it does not replace GDN execution. `model_runner.py:1225–1232,2338–2350` sorts requests through `sort_batch_req_ids`, placing ordinary one-token decode rows ahead of longer prefills when speculation is off. This supports the existing contiguous decode-first contract, but runtime qualification is still required.

## Minimum residual and dependencies

Retain v03's production behavior and focused regressions, with mechanical integration into the frozen scheduler/backend APIs. Keep the generic GDN backend unchanged and the Qwen-specific identity so unrelated recurrent models are not constrained. Preserve connector support in `derive_mamba_conv_split`, registry/dispatch coverage and the XPU block-size recognition; those are necessary consequences of introducing a distinct backend, not obsolete feature work.

- **Hard paired dependency:** k05 `9e2ac704b8627e2fd8d37c42f769e615cfc1f122` supplies `bool split_mixed_non_spec=False` in `csrc/xpu/torch_bindings.cpp`, the matching `ops.h` declaration and Xe2 split implementation. Retaining the Python keyword against upstream kernels alone fails the schema contract. Dropping either half independently loses the requirement. Keep its default false and preserve the speculative/indexed path.
- **Related correctness prerequisites:** k03 `32f42475774852d7174fe6cd9f8201878603381f` disables unstable tiled causal convolution; k04 `09925af1f9cb439803a63aa8c8e6704cb2932082` fences prefill global-state writes. Their owning audits decide the residual native repairs. v03's alignment/splitting does not prove those defects gone, and those fixes alone do not enforce request scheduling or decode-path selection.
- **Independent adapter repair:** v02 `cf0b834cd66c54156393e2f943382b1015360f84` narrows request metadata before the same native call. Preserve that behavior when adding the split predicate. v03 does not supersede padding repair.
- **Downstream consumer:** v05's request-stable KVarN dispatch uses the Qwen enum and CPU metadata. Keep its V2 adaptation mapped to v05; do not fold KVarN allocation/metadata work into v03 merely because paths overlap.
- k06 `be4d79b04ab65a32e6502a3ca0e10943751607d4` concerns shortened speculative rollback, outside v03's split predicate and scheduler profile. Its upstream overlap does not justify dropping v03.

There is no later local repair/revert to fold into v03. A coordinated GDN correctness commit may combine related changes if the parent retains explicit old-to-new mappings and authorship, but the two requirements and cross-repo schema dependency must remain reviewable. Keep independently useful RMSNorm/W4A16 repairs separate.

## Observed checks and remaining validation

**Completed in this audit:** verified commit parent and stable patch ID; inspected complete local diff and final-release follow-up changes; inspected target call paths, schema and scheduler; checked generic-split ancestry; compared old/new GDN kernel sources; read the historical logs below. No new pytest, GPU experiment or candidate build was run. One supplementary `git show` was inadvertently issued in the packaging repository and returned `bad object`; it was rerun successfully in the vLLM repository and is not a source/test failure.

**Historical behavior evidence, not new-pair qualification:**

| Artifact under packaging `benchmark-results/` | Observed result |
| --- | --- |
| `kvarn/20260830T164500Z/direct-gdn-mixed-route/pytest.log` | `test_qwen38_gdn_cached_decode_is_exact_in_mixed_prefill` failed: 3383 output mismatches, maximum absolute difference `0.001953125`. |
| `kvarn/20260830T213744Z/focused-gdn-mixed-route/pytest.log` | Same focused test passed with the split route: 1 passed. |
| `kvarn/20260830T213744Z/focused-gdn-controls/pytest.log` | Canonical state carry and fresh-state batch separability passed: 2 passed. Noncanonical `3970+125` diagnostics still reported 99121 output mismatches (max `0.001953125`) and 274112 SSM-state mismatches (max `0.016520142555236816`); canonical `3968+127` was required to match one-shot T=4095 exactly. |
| `kvarn-k4v2/20260913T051252Z/split-policy-cpu-tests-01.log` | 84 passed, including the GDN adapter test named in warnings; XPU device-count/Sysman warnings make this CPU/mock dispatch evidence, not a native numerical result. |

The August logs predate the consolidated September commits; they establish failure mechanism and historical fix behavior, not exact final-candidate binary identity. Log SHA-256 values, in table order: `94c5ad06de18e58043febec4056eb0e5909e6fb9190b19515dc62c8c3664809a`, `a22c43ddc520b51f98de12bed8f134094205638acb6fb0cf00cf42baa74c4368`, `4bf78f6ef97b8fda8a2249cd89cadd69161c9edbbecd22a696bc570a855f14c1`, `0b03557c8fcea09d359461f1d7d1ddc6e497a5ef5e07a5a5f6d9672eb6ddf568`.

**Required before qualification:**

1. Run existing scheduler/backend/connector tests against the replayed source, especially `tests/v1/core/test_mamba_align_chunk_split.py`, the affected scheduler/prefix-cache tests, `test_qwen_gdn_uses_distinct_backend`, GDN adapter split predicate and block-size tests, and NIXL HMA conv-transfer coverage. Preserve endpoint 3968 for the ragged 8192-budget case, final tail 127, 67-token minimum-budget progress with three decodes, rejection of undersized budgets/thresholds, and unchanged generic/non-XPU/speculative/multimodal profiles. The arithmetic-grid assertions should fail on upstream alone and pass with v03.
2. On the realized paired XPU binaries, verify the loaded `gdn_attention` schema includes the optional flag and run kernel `test_qwen38_gdn_cached_decode_is_exact_in_mixed_prefill`, `test_qwen38_gdn_state_carry_is_exact_on_canonical_boundaries`, fresh-state replay/batch-separability and async-chain tests. Compare output, z, convolution state and SSM state with standalone controls, including dirty allocations and reused state slots. Use equivalent direct split-op controls on upstream if the flag is unavailable; schema rejection is not a numerical comparison.
3. Exercise V2 eager no-spec Qwen with ragged requests, non-final/final prefill boundaries and a cached decode joining/leaving prefill. Validate real decode-first metadata and matched-history outputs/state. Separately qualify required K4V2 MTP2, syvai DFlash2 and vision behavior; v03 intentionally does not solve their partitioning or rollback requirements.
4. Measure matched no-spec mixed-batch throughput/latency and scheduled-token counts after replay. Splitting adds launches and alignment can leave budget unused; preserve correctness while quantifying those costs. No new performance benefit, full-model bitwise determinism or performance parity is claimed by this audit.

Remaining uncertainty is runtime compatibility and numerical/performance qualification on the new pair, not demonstrated upstream coverage. Keep the requirement represented until those checks resolve it.
