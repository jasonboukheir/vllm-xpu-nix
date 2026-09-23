# Rechecked implementation map

> Paused at user request. These are historical September source findings and future work, not authorization to implement. Read CURRENT.md/RESUME.md; recheck applicability against the future selected pair.

Source inspection against the frozen pair in `manifest.json`; no GPU results.

| Issue requirement | Frozen upstream coverage and residual |
| --- | --- |
| Stacked RMSNorm | Kernels `7e85c90` (#579) is included, and vLLM `48f663c37f8c77803d9281778d49ec0727abcacf` (#56431) uses per-layer XPU normalization in `Qwen3DFlashModel._normalize_context_k`. Do not backport either again. Test distinct per-layer weights against the actual loaded kernel; profile before removing the fallback. |
| Quantized context projection | Still missing: `_build_context_kv_buffers` slices `a.qkv_proj.weight`; `_project_context_kv` uses `F.linear` on the fused buffer. Quantization-aware projection or a narrowly scoped, measured dequantized context buffer is required. |
| Quantization mapping | qkv/o/gate-up/down and context `fc` receive quantization configuration. The causal wrapper inherits Qwen3's mapping, but `models/utils.py:get_draft_quant_config` never applies `configure_quant_config` to the draft class. Verify inherited mapping and actual XPU INT4 dispatch; quant-config construction alone is insufficient. |
| V2 target/draft ownership | `init_attn_backend` supports active-layer filtering. Target V2 initialization still needs released KVarN draft ownership and independent mutable state reconciled with the separate speculator builders. Filtering alone is insufficient: V2 autoregressive/MTP prefill currently reuses target metadata, so its dedicated draft owner must also build prefill metadata. See the v11 audit. |
| Committed history | Upstream V2 device metadata is not proof that the local CPU-tail state can use its CPU upper bounds. Implement exact committed history and flush/rollback semantics; disable adaptive verification until mismatched query lengths are correctly supported. |
| Cache allocation | Upstream now has HiSparse host/device allocation support. It is not equivalent to KVarN attention/recurrent per-group pools. Reconcile the two paths rather than overwriting upstream allocation changes during replay. |
| DFlash insertion order | The speculator still computes and inserts context K/V separately from ordinary attention forward. Add reservation/insertion/commit/reclaim semantics for compressed caches, including dummy runs and rejected proposals. |
| Draft geometry/window | Local released native KVarN paths are D256/Q24/KV4-oriented. syvai needs D128/Q32/KV8, a 2048-token noncausal sliding window and bounded draft allocation. Existing target geometry cannot be reinterpreted. |
| GDN rollback | Kernel `da16a5595c105605bcd44b558c7933b934a05f85` fixes ragged speculative traversal. Individual k03–k06/v02–v03 audits must establish which local correctness fixes remain. |
| Host configuration | The consumer flake is now `~/.config/nix/modules/flake/nixos/server`, not the root config flake. The active unit still points to the released runtime and was already inactive. Activation remains separate. |

Update this map after audit reconciliation/replay and whenever tests disprove
an assumption. Each residual implementation requires focused source tests,
then the issue's integrated direct-K4V2 qualification and profiling gates.

## Lifecycle constraints confirmed during audit

- V2 DFlash's `_prepare_dflash_inputs_kernel` already masks evicted/null
  physical block 0 and rejected context rows to `PAD_SLOT_ID`. Preserve that
  upstream behavior. The KVarN builder must not allocate resident slots or
  sinks for those null block-table entries when adapting its old `bid >= 0`
  loops. Test dummy/profile behavior separately from live writes.
- The released K4V2 16/8 resident-history policy is not a sliding window.
  Blindly applying its 16-block residency to a 2048-token draft window with
  128-token blocks can retain the whole visible window uncompressed. Define
  and measure a draft-specific bounded policy, and verify actual packed
  writes/reads rather than treating a configured dtype as compression proof.
- The k09 audit verified generic D128 natural-layout source machinery
  (13,824 bytes per G128/K4V2 record), but factory variant 18 forces the
  native D256 DPAS contract. A natural route needs explicit compatible
  dispatch. Its suffix mask and ordinary prefill attention wrapper do not
  establish DFlash's multi-query noncausal sliding semantics. Compare those
  semantics to the checkpoint/reference before choosing the implementation.
- For the first correctness prototype, derive exact lengths from live device
  metadata if necessary; neither V2 CPU sequence upper bounds nor stale
  aliased shadows establish committed history. Track synchronization costs
  for the later profiler diagnosis, without changing lifecycle correctness
  to remove a synchronization before measurement.
- DFlash context writes occur before normal draft metadata construction.
  Reservation must cover incoming committed context plus query writes;
  flushing is legal only once the relevant accepted context is resident.
  Moving the existing builder earlier without separating these phases is
  insufficient.
- After separating owners, `_init_kv_zero_meta` still needs all physical
  caches, including the dedicated draft groups. Keep ordinary-backend MTP
  metadata reuse intact unless its callers are migrated explicitly.
  `ExtractHiddenStatesSpeculator` also consumes draft entries in target
  metadata, so an unconditional filter for every backend is inappropriate.
- Distinguish whole-model reset from retained-model cache replacement.
  The latter must invalidate KVarN cache references, allocator mirrors and
  builder shadows while preserving implementation registration. Upstream
  `clear_layer_kv_caches` alone does not clear the local KVarN state.
- Reserve the bounded draft window for concurrent runner slots independently
  of the target's growing shared capacity. One full-target-request capacity
  unit may serve several shorter requests, each with a draft window. Reusing
  the released allocator's equal attention-token capacity across every pool,
  or multiplying every attention spec by the target override alone, does not
  meet that contract. Use the sliding spec's in-flight/alignment semantics in
  allocation, auto-fit and admission, charging each pool's null block once.
- Upstream removed the legacy exact CPU shadows, but the retained V1 proposer
  still increments a borrowed CPU upper bound in place. Keep only its
  out-of-place bound update from v16. V2 already builds independent CPU
  bounds; add ownership/history regressions there without duplicating the old
  increment implementation or confusing independence with exactness.

## Quantized projection reference inspected during A

Upstream PR #53122 is still open at head
`647fb1d0cb524734bdf5e73ea9d924df3204729c`, with no merge commit. Its saved
diff and API metadata are in `benchmark-results/issue17-20260920/`.
The PR description is stale: the actual diff adds `_dequant_kv_slice` with
packed INT4/INT8 and other format handling, plus draft mapping configuration;
it does more than the description's dtype-rejection guard.

The relevant idea is to unpack only K/V output rows once at `load_weights`
time, before `process_weights_after_loading` changes their checkpoint layout.
Preserve INT4 QKV/MLP/fc execution; a deliberately scoped context buffer is
permitted by issue17, while full-model dequantization is not. For syvai's five
layers, D128 and eight KV heads, the retained BF16 fused context weight buffer
alone is 100 MiB (`5 * 2 * 8 * 128 * 5120 * 2` bytes). Actual persistent and
transient allocation still need measurement. Do not copy the patch wholesale:
its broad `except Exception: pass` around mapping can hide an incompatible
draft, and packed shape/bit/group/symmetry checks must match the pinned syvai
format and the XPU loader. Its dummy-weight/init/reload path must also be
safe after XPU repacking. Reuse meaningful numerical tests for pre/post-load
layout, different group scales and K/V rows; validate model projection outputs.
