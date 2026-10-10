# v03: KEEP functional XPU residual RMSNorm

Owner: `/root/v03_rmsnorm`. Read-only source audit on 2026-10-10; this report is
the only file written. No source, refs, services, GPU state or external issues
were changed. **KEEP** the helper and its regression tests. New upstream global
XPU batch invariance covers an overlapping use case, but does not cover the
scoped quantized-model caller or the direct helper's functional contract.

## Frozen inputs

| Input | SHA / identity |
| --- | --- |
| Assigned commit | `d909b25f0119419481bc378c3c0a3c656f9b76b0` — `fix(xpu): preserve functional batch-invariant residual RMSNorm` |
| Parent | `6efe5e7cc59de64ae8d3ec67bbe77379be34d2c9` |
| Verified stable patch ID | `2db40de80fa6583e1083c2f41979f06b51c3bd08` |
| vLLM old base / tip | `ced6857afa0ea7b2e3f0846a62e1394e90f15607` / `466b9d5abe84f13c29b38c686e4d77fc09e09aa4` |
| vLLM target | v0.31.0, `db9527a46873454610df6dbedf79a36d6bf1a7f6` |
| Kernels old base / tip | `1c7cbeee1cd0c1481d48f5031f679a47f3f0ef45` / `193cee080d654ff67c4f8bfaf924201bc88b9cf5` |
| Kernels target | release/0.1.15.4, `ddf336d86e3c8602888572a3502f951abd51df12` |

Read the run's full `inventory.json`, `docs/fork-delta.md`, commit-audit runbook,
and `../vllm/AGENTS.md`. Historical evidence was read directly from packaging
commit `8c72895340deafef77998a579335d078800626ab`, particularly the prior v04 audit
and `aeon-integration-followup.md`. Historical test results are not new tests.

## Requirement and effective contribution

`rms_norm_batch_invariant` on XPU must leave input, residual and weight unchanged,
return two separate outputs, normalize the unrounded FP32 input-plus-residual
sum, and return that sum rounded to the activation dtype as the residual.
Its fixed 1024-column reduction blocks keep a row's results independent of
unrelated rows, including the native kernel's 256-row launch transition.
It supports FP16/BF16 activations, absent weights, FP32 Gemma weights, strided
inputs/weights and higher-rank shapes. Non-XPU residual dispatch remains native.
This is a correctness/reproducibility requirement, with no measured speed claim.

The two files modified by the assigned commit are byte-identical between that
commit and the frozen full old tip (`git diff --exit-code` passed). Later
`466b9d5abe84f13c29b38c686e4d77fc09e09aa4` adds its important consumer:
`GemmaRMSNorm.forward_native` selects the helper through the request-stability
callback, and `forward_xpu` bypasses native fusion when scoped request slices
are present. The policy in `determinism/request_stable_linear.py` selects the
frozen AEON W4A16/BF16 model, compact K4V4/K4V2, eager V1, no speculation/prefix
caching, language-only and restricted parallelism. It is inactive for Brutus's
current RedHat MTP/vision profile. Inactivity there does not demonstrate that
the supported optional caller is obsolete.

## Upstream coverage and alternate paths

The relevant upstream addition is
`e4340e41c9a32b2247cc2442b35aa91b081c424f`, **[Feat][XPU] VLLM_BATCH_INVARIANT
support for Dense/MoE models (#55881)**. Inspected its complete changes and the
selected target implementations:

| Target implementation | What it establishes |
| --- | --- |
| `model_executor/layers/layernorm.py:117-130` and `177-190` | Under global `VLLM_BATCH_INVARIANT`, residual RMSNorm/GemmaRMSNorm call the functional native IR provider directly. This bypasses the generic residual helper for those ordinary layer calls. Preserve these new upstream branches during replay. |
| `model_executor/determinism/batch_invariant.py:enable_batch_invariant_mode` | Registers deterministic `aten::mean.dim` on XPU, making native IR's mean fixed-order under the global mode. Previously that registration was CUDA-only. This is a real alternative implementation, not merely matching patch text. |
| `ir/ops/layernorm.py:44-62` | Functional native provider adds in FP32, then calls `.mean`; absent global initialization it has no fixed-order mean guarantee. It also casts normalized values to `weight.dtype` before multiplication, unlike the retained helper's FP32 multiply for every weight dtype. It therefore is not an exact unconditional substitute. |
| `platforms/xpu.py:305-324` | Global XPU batch invariance explicitly rejects weight quantization and all cache dtype names except `auto`, `float16`, `bfloat16`. Enabling it for the scoped compressed-tensors/KVarN profile is unsupported. It also changes attention selection and other operators; enabling it just for this helper is not a minimal replacement. |
| `batch_invariant.py:rms_norm_batch_invariant`, residual branch at approximately 995 | Still invokes `_custom_ops.fused_add_rms_norm(input, residual, weight, eps)` and returns the mutated arguments. The old-to-new helper delta is docstring formatting only. Direct calls and the scoped caller remain uncovered. |
| `tests/v1/determinism/test_xpu_batch_invariant_ut.py:test_residual_norm_preserves_batch_invariance` | Adds discriminating RMS/Gemma layer cases, including 257-row and heterogeneous-column cases, under global mode. It does not cover the direct helper with global mode disabled, non-mutation, or all optional/FP32/strided weight cases. These tests supplement rather than replace the fork tests. |
| Kernels `csrc/layernorm.cpp:763-829` | Still selects `(num_tokens < 256) ? 1024 : 256` for residual launch size. Both mutation and activation-dtype residual rounding remain. The file is byte-identical between kernel old base and target: blob `fd504487cfff6257859966e1a1084ad963f2a559`. `csrc/torch_bindings.cpp` is also unchanged. |

The upstream layer path now obsoletes the helper as the generic **global-mode
layer** fix. It does not obsolete its remaining independent contract and caller.
No intervening local repair or companion norm change supersedes them.

## Minimal residual, dependencies and possible further reduction

Retain this small standalone correctness commit with its XPU-only residual
branch, fixed-tree kernel and two added tests. Keep the non-XPU branch and
upstream global-mode layer branches intact. The scoped Gemma caller remains
owned by vLLM `466b9d5abe84f13c29b38c686e4d77fc09e09aa4`; reconcile its
`GemmaRMSNorm.forward_xpu` changes around upstream's new global-mode branch.
The two selectors cannot normally overlap because upstream rejects that
quantized configuration, but replay must preserve both supported dispatches.

There is no kernel API/schema/ABI dependency and no source dependency on the
GDN commit despite their parent relationship. Kernel deterministic W4A16 does
not provide normalization determinism. No follow-up needs folding into v03.

A concrete smaller candidate, if desired after qualification, is to form the
merged tensor in FP32 and call the **existing no-residual** Triton helper on it,
then cast normalized and residual outputs back to the original dtype. That
could eliminate the duplicate fused kernel while retaining the fixed reduction
and FP32 weight arithmetic. It adds FP32 temporary tensors/launches and has not
been compared for exact results, storage ownership or latency. This audit does
not claim that source reduction preserves performance or recommend replacing
the tested implementation without those checks. Simply calling native IR or
turning global invariance on does not cover the residual requirement.

## Validation

Observed here: stable patch ID matched inventory; exact old/final/target call
paths inspected; relevant upstream commit and tests inspected; assigned-file
old-tip diff and kernel layernorm/binding old-to-target diff both empty. No GPU
tests or timings ran. Prior xpu-v1.11.0 records report direct functional RMSNorm
tests passing, while optional AEON whole-model concurrent token invariance
failed on both candidate and published baseline. Do not represent that broader
invariant as passing or as repaired by this commit.

Required on the parent's built candidate:

1. Run the retained `test_xpu_residual_rms_norm_is_functional_and_batch_invariant`
   cases (48 parameter combinations, each exercising rows 1, 4, 255, 256, 257,
   300), with global invariance disabled so the independent contract is tested.
   Require exact unchanged inputs/non-aliasing, FP32-add reference agreement and
   exact same-row outputs across batch sizes. Record real execution, not skips.
2. Run `test_non_xpu_residual_rms_norm_preserves_native_dispatch`; retain CUDA
   upstream tests and mark actual CUDA unavailability honestly.
3. In a separate process initialized for global invariance, execute upstream's
   XPU residual layer cases and quantized-config rejection tests. Do not enable
   this global flag in the production W4A16/KVarN serving profile.
4. Check that replay preserves the scoped Gemma callback and that the current
   RedHat profile keeps its selector inactive. Then run the parent's actual
   Brutus model/server/API qualification. That serves a different requirement
   from optional AEON whole-model invariance.
5. Any replacement composition needs warmed decode/prefill comparisons at the
   same widths/dtypes and explicit allocation/aliasing checks; no speedup or
   performance non-regression claim is established by this audit.
