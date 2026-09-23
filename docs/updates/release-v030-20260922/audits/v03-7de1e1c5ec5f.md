# v03 — Preserve Qwen GDN prefill and mixed-batch arithmetic on v0.30

**Disposition: KEEP**, paired with k05. The released upstream pair does not
cover either correctness requirement. Retain the scoped arithmetic grid,
Qwen backend identity and guarded native mixed-batch split, integrating them
with the target scheduler's checkpoint handling. No production hunk is
demonstrated redundant. This is a source-audit recommendation, not runtime
qualification.

## Assignment and immutable inputs

| Input | Identity |
| --- | --- |
| Owner / run | `/root/r30_v03` / `release-v030-20260922` |
| Assigned commit | `7de1e1c5ec5f3005b0e46bb0ec323f5e5864876c` — `fix(xpu): stabilize Qwen GDN prefill and mixed scheduling` |
| Parent / verified stable patch ID | `cf0b834cd66c54156393e2f943382b1015360f84` / `b1f78180963d8928e113c17f5418afc7a63517c7` |
| vLLM old base / main | `51da0ca66c8065619c79e35dff97aa99aeaf5644` / `3b6c822e8a4f56dc3904c7fce388a76dff4c7bdd` |
| vLLM released tip / target v0.30.0 | `2fdd9d71f6895379b0940a8f755b4c9095840558` / `ced6857afa0ea7b2e3f0846a62e1394e90f15607` |
| Kernels old base / main | `e7f20cbc18d741419a84aa57178e1f1cc0b453da` / `9bb1e2d428b8a25b262c0cd6704614bdf6ee3ad8` |
| Kernels released tip / target v0.1.15 | `2c39287deec5e59cc2656e6d4a6dce51a567c411` / `1c7cbeee1cd0c1481d48f5031f679a47f3f0ef45` |

Inspected the frozen clones under `build-dev/release-v030-20260922/`, manifest,
27-row inventory, commit-audit instructions, both source `AGENTS.md` files,
and [prior individual audit](../../issue17-20260920-snapshot-audit/audits/v03-7de1e1c5ec5f.md).
The prior report's verified SHA-256 is
`617f56edf915b2e06e9e7cd401a99f9328cd4259fcf7017b35f0d9689dbb3d74`.
This report revalidates its findings against the release targets, with scope
limited to existing eager V1 K4V2 target/draft MTP2 and vision. Issue #17 remains
paused; its V2/DFlash qualification requirements are not part of this audit.

## Requirement and final old behavior

Native Xe2 Qwen GDN changes numerical arithmetic with prefill partitioning or
when another request's prefill joins a cached one-token decode. The two repairs
are independent of the attention KV format:

- Keep non-final prefill endpoints on the 64-token arithmetic grid. For ragged
  prompts `[127, 4095, 4095, 127]` under an 8192-token budget, schedule 3968
  instead of 3970 tokens for the partial prefill, preserving the final 127-token
  tail. The grid applies only on XPU, Qwen GDN, cache mode `none`, no speculation,
  no multimodal input support and no encoder-decoder model. Minimum budget
  `64 + max_num_seqs - 1` prevents starvation; nonzero prefill thresholds below
  64 fail closed. Cache checkpoints must not bypass the independent grid.
- Set `split_mixed_non_spec` only for contiguous ordinary decode-first batches
  with both groups present, one token per decode, exact non-spec token total,
  zero speculative decodes, no token indirection and no speculative mask.
  Paired k05 then runs the existing decode and prefill arithmetic separately.

Neither guard extends the arithmetic-grid claim to MTP or vision. Those existing
profiles must continue to pass regressions with the guards unchanged.
Comparing v03 with the old released vLLM tip shows its adapter and model/connector
identity unchanged. Subsequent GDN metadata and scheduler edits add v05's
immutable CPU metadata and independent-pool zeroing, not a replacement for
either repair. v05 additionally consumes `QWEN_GDN_ATTN` for request-stable
dispatch in `XPUPlatform.set_additional_forward_context`.

## Fresh release-target evidence

All line references here are to the exact release targets above.

| Evidence | Coverage conclusion |
| --- | --- |
| vLLM `vllm/_xpu_ops.py:137–222`, `_gdn_attention_core_xpu_impl` | Native call remains unsplit; no optional split flag. Old base and v0.30 have identical file blob `c1f2bb6969b9d4bfe22989c9fcc258d8bcfd3adb`. |
| `vllm/v1/attention/backends/gdn_attn.py` | Old base and v0.30 have identical blob `23daff54300befa32acc09bab34cf0d3975a4e05`; Qwen-specific identity/arithmetic requirement remains absent. Target searches also find no `QWEN_GDN_ATTN`, `get_required_prefill_chunk_size`, `mamba_prefill_alignment` or `split_mixed_non_spec`. |
| `vllm/v1/core/sched/scheduler.py:340–365,413–526` | Alignment is enabled only for cache mode `align`, with `cache_config.block_size`; no arithmetic grid/minimum-budget requirement exists for mode `none`. |
| `vllm/platforms/xpu.py:395–428`; `gdn/base.py:49–51` | XPU rounds GDN cache block size to 64 but only recognizes generic `GDN_ATTN`; Qwen inherits that generic identity. Cache sizing alone does not constrain the scheduler in mode `none`. |
| `qwen_gdn_linear_attn.py:402–403,976–1010,1407–1463` | XPU selects `forward_xpu`, bypassing the generic implementation's `split_non_spec` path. The generic solution therefore does not cover this native branch. |
| Kernels `csrc/xpu/gdn_attn/gdn_attn_interface.cpp:689–733,757–904` | Any nonzero prefill count selects chunk delta-rule arithmetic for the whole non-spec group; the fused entry point still passes that group through convolution/delta without request-type splitting. |
| Kernel `csrc/xpu/torch_bindings.cpp:246–259` | `gdn_attention` schema still ends with `reorder_input`, with no split flag. |

The entire kernel `csrc/xpu/gdn_attn` tree is identical between the old base and
v0.1.15: tree object `fa1496f9edb4f687f71b1a63d05ea24a9810a108`.
Thus the prior numerical failure mechanism has no new native implementation to
reassess for v03. In particular, **v0.1.15 does not contain**
`da16a5595c105605bcd44b558c7933b934a05f85` (#600); its prior snapshot-only
ragged-rollback coverage must not be carried into the separate k06 decision.

Checked alternative upstream solutions and old-base-to-release changes:

- Generic mixed splitting `fa27d4e9cf3c8d8a5a143f38c346b27c02b2c2e3` predates
  the old base (ancestry verified) and is still bypassed by native XPU.
- `f6326f53bda46898a331c2d24500332c285d9a2b` (#55715) is the only old-base-to-
  target change in the GDN model directory: it enables CUDA SM12x FlashInfer
  prefill and does not change XPU execution.
- Scheduler fix `263c4ff95fadb62d80f315171b17d4f623494e56` (#53945) handles
  EAGLE resume checkpoints. Preserve that upstream logic when applying the
  independent arithmetic/cache-alignment distinction; it is not a replacement
  for scheduling mode `none` on an XPU arithmetic grid.

## Minimum residual, paired dependencies and folds

Keep v03's focused production/test changes, preserving generic backend behavior
and all original exclusions. Introducing a distinct backend requires the enum,
model override, XPU block-size recognition and connector
`derive_mamba_conv_split` support; deleting those ancillary hunks would strand
consumers. Reconcile only changed scheduler/test context, retaining new upstream
checkpoint fields rather than restoring an old scheduler implementation.

- **Hard paired requirement:** k05 `9e2ac704b8627e2fd8d37c42f769e615cfc1f122`
  supplies the optional `bool split_mixed_non_spec=False` schema, declaration and
  native implementation. Python and kernels must ship together; upstream-only
  kernels reject the Python keyword. Preserve default false and excluded paths.
- v02 `cf0b834cd66c54156393e2f943382b1015360f84` trims request metadata at the
  same adapter; retain it independently when adding the predicate.
- v05 `fcab69494f1719e91e416aace17ef6ad501fe900` consumes the Qwen identity;
  its metadata/cache ownership adaptations remain mapped to v05.
- k03 `32f42475774852d7174fe6cd9f8201878603381f` and k04
  `09925af1f9cb439803a63aa8c8e6704cb2932082` address distinct convolution and
  barrier defects. This audit does not supersede their residual decisions.
- No later local repair or revert belongs folded into v03. If the parent groups
  related GDN fixes into a coherent commit, preserve every original mapping and
  authorship; keep unrelated RMSNorm/W4A16 requirements distinct.

## Observed verification and remaining gates

Completed read-only parent/patch-ID verification, old/final/target source
comparisons, target call-path/schema inspection, tree/blob equality and ancestry
checks. Both frozen source worktrees were clean when inspected. Only this audit
file was written; no source/ref/lock changes, build, GPU, service operation or
new pytest run occurred.

Re-read retained historical logs and verified their hashes against the prior
report: the original mixed route failed with 3383 output mismatches (maximum
absolute difference `0.001953125`), its split replacement passed, and canonical
state carry/fresh-state controls passed. The noncanonical `3970+125` diagnostic
reported 99121 output and 274112 SSM-state mismatches; `3968+127` was the canonical
control. The retained September CPU/mock suite had 84 passes. Exact paths and
hashes are in the linked prior audit and still match. These results establish
the historical requirement, not qualification of the new source pair.

Required before release:

1. Run affected scheduler/backend/connector and adapter tests, especially
   `tests/v1/core/test_mamba_align_chunk_split.py`, backend selection, adapter
   split predicates, prefix-cache and NIXL HMA cases. Check 3968/127 endpoints,
   67-token progress with three decodes, undersized-budget rejection, and
   unchanged generic/non-XPU/speculative/multimodal profiles.
2. On the realized paired binaries, verify the loaded operator schema and run
   cached-decode standalone-versus-mixed output/z/conv-state/SSM-state equality,
   canonical-boundary state carry, fresh/dirty allocation, slot reuse and async
   replay checks. Use direct split-op controls for upstream numerical comparison;
   an unsupported-keyword failure is not such a comparison.
3. Qualify existing eager V1 K4V2 target/draft MTP2 and vision, including ragged
   requests and joining/leaving mixed batches, without expanding the guards.
   Measure matched no-spec mixed latency/throughput and scheduled-token counts:
   extra launches and alignment can have costs. No new speedup, full-model
   bitwise determinism or performance parity is claimed here.

Remaining uncertainty is new-pair runtime/numerical/performance qualification,
not demonstrated upstream coverage. Keep the requirement covered pending those
gates.
