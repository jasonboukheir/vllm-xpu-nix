# K03: retain untiled GDN convolution prefill pending new-toolchain controls

**Disposition: KEEP.** Retain the small dispatch guard and its focused replay
regression. The selected upstream source has no correction or replacement for
the previously failing tiled non-speculative prefill implementation. The new
compiler/runtime makes a controlled retirement experiment worthwhile, but does
not establish that the failure is fixed.

## Frozen inputs

| Input | Value |
| --- | --- |
| Assigned commit | `956b61f95a0b38e73eeebe9cbfeff1be782a9283` |
| Parent | `e1bec3b3e9d0c02cc0e8ce9354d5d2908d51a87f` |
| Subject | `fix(gdn): disable unstable tiled causal-convolution prefill` |
| Recomputed stable patch ID | `f7fd14e5d76964b1426c46ae21f46913f227194d` |
| Kernel old base / full tip | `1c7cbeee1cd0c1481d48f5031f679a47f3f0ef45` / `193cee080d654ff67c4f8bfaf924201bc88b9cf5` |
| Kernel selected target | `ddf336d86e3c8602888572a3502f951abd51df12`, release branch for wheel `0.1.15.4` |
| vLLM old base / full tip | `ced6857afa0ea7b2e3f0846a62e1394e90f15607` / `466b9d5abe84f13c29b38c686e4d77fc09e09aa4` |
| vLLM selected target | `db9527a46873454610df6dbedf79a36d6bf1a7f6`, `v0.31.0` |

Read the assigned commit-audit procedure, kernel `AGENTS.md` (including the
frozen target version), current fork rationale, complete [inventory](../inventory.json),
and prior K03 audit preserved at packaging commit
`8c72895340deafef77998a579335d078800626ab`, path
`docs/updates/release-v030-20260922/audits/k03-32f424757748.md`.
This audit wrote only this report; it changed no source, refs, services or GPU state.

## Requirement and effective old-stack behavior

Fresh Qwen GDN prefills on Arc Pro B70 must not corrupt projected Q/K/V or
depend on recycled convolution cache contents. Original historical commit
`1437161b97dfd7a6799ce2b2da5ab53fdcf36a6f` records corrupted Q/K/V and repeated
`!` output with prefill lengths at least eight on oneAPI 2026.0, even after
metadata experiment `54d39759cdbd3ea16f9d4736af173cdfa5ba4c90`.
That is recorded failure evidence, not independent proof of the claimed
compiler root cause.

K03 sets `enable_tiled_prefill=false` in `causal_conv1d_non_spec`, selecting
`gdn::chunk_causal_conv1d_xe2` with fused L2 normalization whenever the XE2
prefill branch runs. It leaves decode-only and non-XE2 dispatch unchanged.
The original patch is nine added/one removed production lines plus a 100-line
regression. The guard and regression survive at the full old tip; no later
commit reverses them. This supports the published RedHatAI/Qwen3.8-27B-INT4,
BF16, K16/V48/D128, TP1, eager V1, K4V2 target/draft, MTP2 Brutus workload.

The test `tests/gdn_attn/test_causal_conv1d_replay.py::test_qwen38_fresh_prefill_replay_is_exact`
uses 162 tokens, width four, `reorder_input=True`, fresh state and recycled
cache values 0, 3.25 and -1.75. It compares all returned intermediates, z and
selected final state exactly. It detects instability and dirty-state dependence;
by itself it cannot distinguish stable correct output from stable wrong output.

## Upstream coverage and alternate routes

The four-commit kernel delta contains `f4b729ae7fb3eb22c564606ec8b4d184c7c8bc4a`
(#600 ragged speculative traversal), `d09702a637cfea2a0c02909af82bd789d26ea319`
(Torch 2.14/toolchain update), paged-decode configuration and top-k interface
updates. Only #600 changes GDN code, and its convolution changes are confined
to `causal_conv1d_spec` and `causal_conv1d_spec_kernel`: compact speculative
query traversal and rollback capacity. It does not change this non-spec
prefill dispatch or either XE2 prefill convolution implementation.

At the selected kernel target:

- `csrc/xpu/gdn_attn/gdn_attn_interface.cpp:435` enters XE2 prefill when
  `num_prefills > 0`; line 466 still selects tiling at
  `non_spec_token >= gdn::conv1d_tile_size`. Lines 470 and 495 dispatch tiled
  and untiled implementations respectively. The tile size is eight in
  `xe_2/chunk_causal_conv1d_tiled_xe2.hpp:10`.
- The tiled header has identical old-base and target blob
  `21a73ccc3aa46da79a84b54c1cef58225848e5a6`. Its item-zero metadata broadcast,
  SLM load staging and group synchronization remain the same; for example
  metadata is written at lines 166-194 and consumed after the barrier at 196.
- The untiled header also has identical old-base and target blob
  `ccb6006d8e5ebf02c3ee33fbafd093a3c086e0fc`. Its local input is zero-initialized
  and prior history is loaded only when `has_init_conv_states` is true
  (`chunk_causal_conv1d_xe2.hpp:199-239`). This remains the available alternate
  implementation preserving the requirement.

At vLLM v0.31, `QwenGatedDeltaNetAttention` still selects `forward_xpu`
(`qwen_gdn_linear_attn.py:406`), calls `gdn_attention_core_xpu` at line 999,
and `_gdn_attention_core_xpu_impl` calls `_xpu_C.gdn_attention`
(`vllm/_xpu_ops.py:193`). The latter implementation is unchanged apart from
line offsets. Upstream `d05da62e9ccdf8e342b15bf6785d83224cc165af` corrects
XPU logging, while `68363d64ec476bbbf5025b30ced0f6a2425b1291` factors output
normalization; neither replaces convolution. New AITER and FlashInfer behavior
applies to other platform routes. No inspected redesign bypasses this native
XE2 prefill for the retained Brutus model.

Torch 2.14/oneAPI 2026.1 is a real intervening variable. A compiler fix could
make the historical workaround unnecessary even with identical headers.
No new-toolchain comparison was run in this read-only audit, so neither a
continued compiler defect nor its repair is asserted.

## Minimal retained change and dependencies

Replay the existing guard and focused test unchanged. This introduces no
operator, schema or ABI change and needs no prerequisite from the adjacent
build-policy commit. Keeping the upstream tiled implementation in place and
gating its dispatch keeps the maintenance change small; deleting the whole
implementation would enlarge the delta without meeting another requirement.

Kernel `3b5deb0773be875a328a1831c5e4a08233460d32` orders the later GDN delta
stage and is independently necessary unless separately covered. Kernel
`9a17a2c8617be2b0471db79fb301912601ee945c` and vLLM
`6efe5e7cc59de64ae8d3ec67bbe77379be34d2c9` preserve mixed-batch arithmetic;
their prefill group still traverses this guard. Kernel
`30fda95e35666948deb3c9bd7fd72aaff9c2ad66` concerns speculative rollback and
can be reconciled with #600 independently. KVarN and later test-only work do
not supersede K03. No follow-up fix needs folding into this commit.

## Validation and optional retirement experiment

Observed here: immutable patch-ID check, old-base/target implementation blob
comparison, effective-old-tip inspection, complete relevant upstream diff,
and companion dispatch review. Historical
`benchmark-results/releases/xpu-v1.8.0/kernel-regressions.sh` explicitly includes
the replay file; its log records 34 passed, 17 skipped and 99 deselected.
Published v1.11 release notes record successful native and actual-model
qualification with the guard. These are prior-stack results, not a v0.31
or oneAPI 2026.1 qualification. No runtime test ran in this audit.

The parent can make a minimal matched control after replay by changing only
`constexpr bool enable_tiled_prefill = false;` to `true` in
`csrc/xpu/gdn_attn/gdn_attn_interface.cpp`, keeping all other fork patches,
build flags, toolchain and inputs identical. This restores upstream dispatch,
including tiling for the 162-token fixture. Verify that XE2 is enabled and that
each test process imports its intended freshly rebuilt native extension; an
old cached library would invalidate the comparison. This is a source switch,
not a currently implemented environment option.

Exact existing focused tests, in priority order, in both isolated builds:

```bash
.venv/bin/python -m pytest -q \
  tests/gdn_attn/test_causal_conv1d_replay.py \
  tests/gdn_attn/test_gdn_attn.py::test_gdn_attention_gqa_ratio3_prefill

.venv/bin/python -m pytest -q \
  tests/gdn_attn/test_gdn_attn.py::test_qwen38_gdn_fresh_state_replay_is_exact \
  tests/gdn_attn/test_gdn_attn.py::test_qwen38_gdn_fresh_state_is_batch_separable \
  tests/gdn_attn/test_gdn_attn.py::test_qwen38_gdn_state_carry_is_exact_on_canonical_boundaries
```

The first command has five cases: one production-geometry fresh replay and
four ratio-three independent-reference cases (16-token cached prefill, both
layouts and FP16/BF16). The second exercises 63/64/65/104/107/162-token fresh
replay, ragged batch separation and canonical carried state. Fresh replay has
an independent-reference check only above 64 tokens, so its short cases alone
are insufficient evidence of accuracy. Keep delta-stage and mixed-batch fixes
constant to avoid attributing their failures to convolution.

For a direct independent convolution oracle, existing
`tests/gdn_attn/test_causal_conv1d.py::test_causal_conv1d` compares every returned
intermediate, z and convolution state using `ref_causal_conv1d` and
`_assert_conv_intermediates`; the latter handles chunk padding and fused L2
normalization. Select prefill/mix-mode cases rather than pure speculative
`test_causal_conv1d_mtp`, which does not exercise the changed dispatch.
Its default geometry is K16/V32, so extend a temporary diagnostic invocation to
K16/V48 and 7/8/9/162 plus 63/64/65 tokens, fresh/dirty state and both layouts
before retiring protection for the production geometry. Reuse that oracle
rather than comparing only two potentially incorrect native outputs.

If the tiled control fails while untiled passes, keep K03. If both pass,
require repeated dispatch-proven reference/replay checks and actual Brutus
model requests before dropping the guard; absence of one historical failure
in one invocation is weak evidence. Measure matched warm prefill/TTFT at the
host's normal 2048-token batch limit and representative short prompts before
claiming a performance benefit. Cross-implementation numerical checks should
use justified tolerance; exact repeatability is required within each route.
Any control result and final parent disposition should be appended to this
report or linked from the release reconciliation map.
