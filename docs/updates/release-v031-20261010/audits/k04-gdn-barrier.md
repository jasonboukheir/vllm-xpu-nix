# K04: keep the fresh-state GDN output barrier

**Disposition: KEEP.** The existing patch already is the minimum residual:
one conditional workgroup barrier publishing O2 before fresh-state output
computation, plus its regression coverage. The selected upstream kernel source
has no replacement synchronization or changed dispatch that makes it obsolete.
Fresh native qualification is required after the Torch/oneAPI upgrade; this
read-only audit is not a GPU pass or a performance measurement.

## Frozen inputs

- Repository: `../vllm-xpu-kernels`; instructions read: its `AGENTS.md`, packaging
  `.agents/skills/vllm-xpu-update/references/commit-audit.md`, `docs/fork-delta.md`
  and this run's [inventory](../inventory.json).
- Assigned commit: `3b5deb0773be875a328a1831c5e4a08233460d32`, subject
  `fix(gdn): order fresh-state prefill output reads`; parent
  `956b61f95a0b38e73eeebe9cbfeff1be782a9283`; recomputed stable patch ID
  `be4c0310f7f161dda8916d4d75bfd15db795f471` matches the inventory.
- Kernel old base: `1c7cbeee1cd0c1481d48f5031f679a47f3f0ef45` (v0.1.15);
  full old tip: `193cee080d654ff67c4f8bfaf924201bc88b9cf5`; selected source
  `ddf336d86e3c8602888572a3502f951abd51df12` from upstream
  `release/0.1.15.4`, accompanying published wheel 0.1.15.4. There is no
  corresponding source tag or wheel-embedded source SHA; see inventory
  provenance. This audit qualifies the frozen source comparison only.
- Companion vLLM old base: `ced6857afa0ea7b2e3f0846a62e1394e90f15607`;
  old tip: `466b9d5abe84f13c29b38c686e4d77fc09e09aa4`; target v0.31.0:
  `db9527a46873454610df6dbedf79a36d6bf1a7f6`.
- The inventory enumerates all five vLLM and eight kernel local commits.
  Relevant dependencies are identified below; no later local commit supersedes
  this barrier. Its header blob is identical at assigned commit and old tip.

## Requirement and smallest surviving change

On Xe2/Intel Arc Pro B70, Qwen GDN prefills must not consume another subgroup's
unfinished O2 writes. The failure class is nondeterministic fresh-state output
or state caused by unsynchronized cross-subgroup communication. Supported
regressions use BF16, K16/V48 heads, D128, width-four convolution, TP1 and
64-token chunks, matching the Qwen3.8-27B geometry. This matters with the existing
Brutus RedHatAI/Qwen3.8-27B-INT4 eager V1/K4V2/MTP2/vision profile independently
of attention-cache quantization.

`csrc/xpu/gdn_attn/xe_2/chunk_gated_delta_rule_kernels_xe2.hpp` gains only:

```cpp
if (!has_prev_state) {
  // Publish O2 across subgroups before the fresh-state output GEMM.
  sycl::group_barrier(item.get_group());
  // existing output dv loop follows
```

The historical v0.30 audit of predecessor
`09925af1f9cb439803a63aa8c8e6704cb2932082` already removed redundant U-barrier
substitution and unconditional per-chunk synchronization. Do not reintroduce
those older forms. Current K04 is that consolidated conditional fix, with a
trailing-newline cleanup and tests; there is no further demonstrated source
reduction.

The patch adds four test behaviors in `tests/gdn_attn/test_gdn_attn.py`:
`test_qwen38_gdn_fresh_state_replay_is_exact` (six lengths and three poison
patterns), `test_qwen38_gdn_fresh_state_is_batch_separable` (two ragged cases),
`test_qwen38_gdn_state_carry_is_exact_on_canonical_boundaries`, and
`test_qwen38_gdn_t107_async_chain_is_exact`. The last queues 24 W4A16/GDN/output
projection chains on a non-default stream with allocator churn. Later local
changes add 182 lines to this test file without changing the barrier header.

## Exact upstream coverage and alternative paths

The old-base and target header both resolve to blob
`b457f3d06213cf141dfed99d0ed1a6a4b387dc7f`; the assigned commit and final old
tip both resolve to `2db48c2fed78a6bcc92356d1431261704848c2af`. The complete
old-base-to-target kernel interval has four commits:

| Upstream SHA | Change | Coverage of this requirement |
| --- | --- | --- |
| `f4b729ae7fb3eb22c564606ec8b4d184c7c8bc4a` | Ragged speculative token traversal, #600 | None: changes `causal_conv1d_spec_kernel`, `gated_delta_rule_spec_kernel`, speculative size validation and tests; does not change the non-spec Xe2 chunk output kernel. |
| `d09702a637cfea2a0c02909af82bd789d26ea319` | Torch 2.14/toolchain/build updates, #525 | No synchronization substitute; increases the need to retest compiled behavior. |
| `885b120ffa082312b7d93c2be7a1e3b3dd79672f` | Paged decode configuration, #602 | Separate attention backend. |
| `ddf336d86e3c8602888572a3502f951abd51df12` | `topk_softplus_sqrt` interface, #603 | Separate MoE operator. |

Line references below are to the frozen upstream header:

- `chunk_fwd_o_kernel` starts at 885. `chunk_gemm_policy_fwd_o` uses the
  64x64x32/4x2 policy; the output grid at 1471 assigns one request/value-head
  per workgroup, comprising eight subgroups. Chunk and output-dimension loops
  stay within that workgroup.
- O2 is stored to the global A workspace by
  `copy(copy_O2_c, tCrO2_c, tCgO2_c)` at 1087.
- The carried-state branch at 1108 has an existing workgroup barrier at 1160
  before `gemm_TTS(O2_tensor, U_tensor_T, ...)` at 1163. It orders updated U
  and O2 for that branch.
- Fresh state skips that entire branch. State-update GEMMs and stores at
  1183–1212 are computation, not a cross-subgroup synchronization point.
  The fresh-state branch at 1214 reaches the O2 consumer at 1223 without a
  barrier following the O2 producer. K04 supplies this missing boundary.
- Earlier barriers at 985 and 1002 precede the current O2 store. The next
  iteration's 985 barrier protects previous-chunk state/SLM handoff, but occurs
  after this iteration's output reads and cannot fix the fresh-state dependency.
- `has_prev_state = (chunk_id != 0) || initial_state` at 968 is workgroup
  uniform: all work-items operate the same request and serial chunk iteration.
  The retained conditional barrier therefore uses converged control flow.

This reasoning follows the specified converged-control-flow and acquire/release
ordering behavior of [SYCL group barriers](https://github.khronos.org/SYCL_Reference/iface/group-functions.html#sycl-group-barrier).
It is a source-level dependency analysis; a non-reproducing race in one run or
a newer compiler is not evidence that the dependency disappeared.

The companion release still reaches this native implementation:
`QwenGatedDeltaNetAttention.forward_xpu` in
`vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:972` calls
`torch.ops.vllm.gdn_attention_core_xpu` at 999; `vllm/_xpu_ops.py:138`
defines `_gdn_attention_core_xpu_impl`, calling `_xpu_C.gdn_attention` at 193.
The target native `gated_delta_rule_non_spec` in
`csrc/xpu/gdn_attn/gdn_attn_interface.cpp:633` selects
`chunk_gated_delta_rule_xe2` at 702 whenever Xe2 is enabled and `num_prefills > 0`.

The vLLM output-norm refactor `68363d64ec476bbbf5025b30ced0f6a2425b1291`
changes the post-GDN projection helper, not this dispatch. The mixed-output
scatter change `975dca5bb5db302077674cfa9afe851ee700ad73` affects the generic
core path; XPU continues its dedicated native call. The XPU logging change
`d05da62e9ccdf8e342b15bf6785d83224cc165af` and FlashInfer upgrade
`44bf011f447b15ae762dbbf4d9dba2752be1ca8b` supply no alternate XPU producer/
consumer ordering. Thus the visible CUDA/ROCm/GDN refactors do not retire K04.

## Dependencies and folds

No native schema, Python signature or ABI changes are introduced by K04, and
no companion edit is required specifically to retain it. Keep it as one
independently useful correctness fix.

- Kernel `956b61f95a0b38e73eeebe9cbfeff1be782a9283` stabilizes the preceding
  convolution stage; it is a separate requirement. Keep convolution dispatch
  constant if comparing barrier variants.
- Kernel `9a17a2c8617be2b0471db79fb301912601ee945c` and vLLM
  `6efe5e7cc59de64ae8d3ec67bbe77379be34d2c9` preserve mixed-decode arithmetic
  and canonical scheduling. The canonical state-carry test also validates
  their workload assumptions; splitting mixed batches does not remove fresh
  prefill's synchronization requirement.
- Kernel `98bdbfde3e75f164bc5040574990db843bf888de` supplies deterministic
  W4A16 projections needed for interpreting the asynchronous projection-chain
  test. A projection mismatch must not be misattributed to the GDN barrier.
- Kernel `30fda95e35666948deb3c9bd7fd72aaff9c2ad66` may be superseded by the
  new upstream ragged-spec fix, but that decision is independent of this fix.
  KVarN commits on either side do not change this header or retire GDN.

## Observed checks and required qualification

Observed in this audit: immutable patch-ID and header-blob comparisons;
complete kernel release delta review; target producer/consumer and dispatch
inspection; test-content inspection; prior release records read from packaging
archive `8c72895340deafef77998a579335d078800626ab`. No source, ref, service,
lock, GPU or build changes were made; only this report was written.

The prior [xpu-v1.11.0 release record](../../../releases/xpu-v1.11.0.md) and
archived `qualification-status.json` record 705 installed native/backend cases
and 14 supplemental GDN-reference passes, plus actual Brutus MTP/vision/mixed
prefill/cancellation checks. These are historical aggregate results on the
previous toolchain; they do not isolate the barrier or establish a v0.31 pass.
The prior audit's proposed comparative barrier/performance controls were not
run under that release's selected operational acceptance; no new speed claim
can be carried forward from barrier counts.

Required on the rebuilt pair, with loaded DSO/toolchain provenance recorded:

1. Run all four retained test behaviors, covering fresh poison, chunk-boundary
   lengths 63/64/65/104/107/162, ragged B4 versus isolated B1, canonical
   3968+127 state carry and the 24-call asynchronous T107 allocator chain.
   Include BF16 and FP32 recurrent-state reference coverage. The replay test
   skips its independent-reference comparison for lengths <=64; retain/add
   focused independent-reference cases for those lengths so a stable wrong
   result does not pass.
2. Complete actual Brutus model load/health and meaningful text, MTP, vision,
   mixed prefill, concurrent request and state-slot reuse checks. Torch
   2.14/oneAPI 2026.1 are a new compiled environment despite the identical
   upstream header.
3. If proposing removal, compare the refreshed source with/without only this
   barrier under identical convolution/matmul settings, tensors and allocation
   pressure. A passing unpatched sample cannot prove absence of a race; require
   an alternative synchronization or algorithmic argument as well. Inspect
   generated lowering if supposedly equivalent barriers disagree.
4. If making a performance claim, collect matched warm ABBA short/long fresh
   and carried-state prefill timings, kernel and end-to-end median/p95 plus
   variability. Retention is for correctness; this audit neither measures a
   synchronization cost nor claims performance non-regression.

The release parent owns native qualification and final reconciliation. Until
that evidence is available, KEEP preserves the demonstrated missing ordering
requirement with the smallest existing code change.
