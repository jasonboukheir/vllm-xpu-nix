# K01 — deterministic oneDNN W4A16 matmul

**Disposition: KEEP.** Retain the three-line deterministic attribute addition and
its focused regression file as one logical commit. Upstream retains precisely
the same oneDNN implementation and dependency revision. New XPU batch-invariance
support in vLLM does not intercept this custom INT4 operation. This is a
reproducibility requirement, with no claimed speed improvement.

## Frozen assignment and scope

- Assigned commit: `98bdbfde3e75f164bc5040574990db843bf888de`,
  `fix(onednn): make W4A16 matmul deterministic`.
- Parent / old kernel base: `1c7cbeee1cd0c1481d48f5031f679a47f3f0ef45`.
- Stable patch ID, independently recomputed:
  `7ea50ac3b0ceb331108743cc12894af0bd2fce23`.
- Old kernel tip: `193cee080d654ff67c4f8bfaf924201bc88b9cf5`.
- Kernel target: released wheel version `0.1.15.4`, frozen source
  `ddf336d86e3c8602888572a3502f951abd51df12` from its release branch. The
  inventory records the absence of an upstream source tag or wheel-embedded SHA.
- Companion vLLM old base: `ced6857afa0ea7b2e3f0846a62e1394e90f15607`;
  old tip: `466b9d5abe84f13c29b38c686e4d77fc09e09aa4`;
  new v0.31.0 base: `db9527a46873454610df6dbedf79a36d6bf1a7f6`.

Read the commit-audit contract, kernel `AGENTS.md`, current fork-delta document,
and the full five-vLLM/eight-kernel inventory in [inventory.json](../inventory.json).
Prior evidence was read directly from packaging commit
`8c72895340deafef77998a579335d078800626ab`, file
`docs/updates/release-v030-20260922/audits/k01-8530de9d4298.md`; that archive is
not present as ordinary files in the current checkout. All source inspection used
the frozen Git objects. No source, refs, services, dependencies, or GPU state were
modified by this audit.

## Requirement and effective old-stack behavior

Brutus's Intel Arc Pro B70 runs `RedHatAI/Qwen3.8-27B-INT4`, revision
`bf08f3dbd9a324e53956920aad378a1f1b6dd24a`, with compressed-tensors W4A16,
BF16 compute, and group-128 symmetric INT4 projections. The retained policy
requests repeatable accumulation for a fixed projection shape. The source adds
`pattr.set_deterministic(true)` in
`oneDNN::dnnl_matmul_w4a16_int4`, avoiding floating-point global K-parallel
strategies whose accumulation order can vary between identical executions.
The setting also affects supported FP16/asymmetric callers of the same op.

The accompanying `tests/test_int4_gemm_determinism.py` supplies nine BF16
fixed-shape replay cases, including M=1 decode and M=107/162 prefill shapes.
It interleaves different weights/scales with the same cached primitive and adds
allocator pressure. Six additional cases compare M=4 with four M=1 calls,
using `atol=rtol=0.015625` and recording exactness separately. These checks do
not establish whole-model determinism, bitwise equality across different M,
or independent mathematical-reference accuracy.

Both changed files have an empty diff from this commit through the frozen old
tip. No later local commit repairs, reverts, or supersedes their contribution.
The v1.11.0 release record documents successful real-profile serving but explicitly
does not claim whole-model batch invariance or a controlled speed comparison.

## Upstream coverage and alternate implementations

The complete kernel upstream delta has four commits:

| SHA | Change | Relevance to K01 |
| --- | --- | --- |
| `f4b729ae7fb3eb22c564606ec8b4d184c7c8bc4a` | Ragged speculative GDN traversal, #600 | Independent of oneDNN matmul. |
| `d09702a637cfea2a0c02909af82bd789d26ea319` | Torch 2.14 and toolchain bump, #525 | Changes the environment requiring qualification; does not change W4A16 source or oneDNN pin. |
| `885b120ffa082312b7d93c2be7a1e3b3dd79672f` | Paged-decode policy, #602 | Independent of oneDNN matmul. |
| `ddf336d86e3c8602888572a3502f951abd51df12` | `topk_softplus_sqrt` interface, #603 | Independent of oneDNN matmul. |

Exact target evidence:

1. `csrc/xpu/onednn` resolves to tree
   `6907657b9b9a8ad655e5ddc0fee565ea636596aa` at **both** kernel bases.
   Its full old-to-new diff is empty. At target
   `csrc/xpu/onednn/int4_gemm_w4a16.h:60`, the callback sets scratchpad,
   scales, zero points, and math mode but has no deterministic setting.
   `csrc/xpu/onednn/onednn_matmul.cpp:257` still invokes that implementation
   directly through the existing `int4_gemm_w4a16` operator.
2. The primitive cache in `csrc/xpu/onednn/onednn_ext.h:800` constructs the
   key from strides, M/N/K, batch dimension, bias and group sizes. On a miss,
   line 814 constructs `primitive_attr`, invokes the callback, then creates the
   oneDNN primitive. There is no bridge from Torch's deterministic controls in
   this directory and no equivalent fixed policy in the cache layer.
3. Kernel `CMakeLists.txt:58` still pins oneDNN
   `0e2a5bfeef1bfbffc3137464606540233086ce9b` (rls-v3.13), and
   `cmake/Modules/FindoneDNN.cmake` is unchanged. Packaging
   `nix/mk-kernels.nix` currently pins that same source. This custom static
   oneDNN build is not replaced by Torch 2.14's internal backend merely by
   upgrading Torch. Target metadata requires Torch 2.14.0+xpu; the Dockerfile
   changes oneAPI base image from 2026.0.0 to 2026.1.2 and XE3 helper compilation
   changes to C++20. Those can change generated code/runtime behavior, so old
   GPU success does not qualify the new build.
4. At companion target, the XPU mixed-precision candidate list remains
   `XPUW4A8IntLinearKernel`, `XPUwNa16LinearKernel` in
   `vllm/model_executor/kernels/linear/__init__.py:517`.
   `mixed_precision/xpu.py:100` still calls
   `torch.ops._xpu_C.int4_gemm_w4a16` for BF16/FP16 W4A16. This entire file has
   an empty old-to-new upstream diff. INC's oneDNN path also calls this op in
   `quantization/inc/schemes/inc_wna16_linear.py:341`.
5. vLLM `e4340e41c9a32b2247cc2442b35aa91b081c424f`, XPU batch invariance
   #55881, adds relevant mean/reduction, attention-selection and other support,
   but `determinism/batch_invariant.py:1091` registers overrides for `aten::mm`,
   `aten::addmm`, `aten::matmul`, and `aten::linear`. It does not override
   `_xpu_C::int4_gemm_w4a16`, and the custom C++ wrapper does not call those
   Aten operations. The source comment about XPU `bmm` does not apply to this
   quantized custom operation. Enabling the new general mode cannot substitute
   for this attribute.
6. W4A8 dynamically quantizes activations and does not preserve the W4A16
   workload's numerical contract. INC ARK is an optional alternate quantization
   path, not a replacement for this compressed-tensors dispatch. The separate
   `TritonW4A16LinearKernel` explicitly accepts only CUDA/ROCm in
   `mixed_precision/triton_w4a16.py:318` and is absent from the XPU candidate
   list. None is demonstrated equivalent coverage on Brutus.
7. Upstream `tests/test_int4_gemm_onednn.py` provides FP16 reference comparisons
   for symmetric/asymmetric quantization and activation ordering. Its only
   activation dtype parameter is FP16; it lacks this BF16 replay requirement.

## Minimal residual and dependencies

Keep exactly the existing three source lines and focused test file. There is
no prerequisite/follow-up local commit, schema change, weight-layout change,
or companion API change to fold. GDN, residual RMSNorm and KVarN changes are
independent requirements. The old commit's authorship and sign-off can be
preserved unchanged during replay.

Do not introduce a mutable per-request deterministic flag as an incidental
performance experiment: the current primitive cache does not include that policy
in its key. Such a feature would require cache separation and transition tests.
Likewise, silently falling back to nondeterministic execution on primitive
creation failure would retire the retained requirement without equivalent
coverage. Source maintenance cost is already one attribute call.

## Observed validation, required checks, and uncertainty

Performed: patch-ID recomputation; full frozen upstream commit/diff inspection;
old/new oneDNN tree identity; assigned-file no-follow-up verification; dependency,
dispatch, alternative-kernel, cache-key and test inspection. No build or GPU
tests were run by this read-only agent, and no candidate pass is claimed.

The parent reports a **pre-existing deployed-service** W4A16 failure, `could not
create a primitive`, before this release work. This audit did not reproduce or
attribute that failure. It is not evidence that determinism is obsolete, nor
proof that retaining it is compatible with the new toolchain. Diagnose this
early in qualification before using service-level failures as a reason to change
the patch:

1. In an isolated candidate process, first create the actual production BF16,
   symmetric group-128 primitives for decode and prefill. Record shape/strides,
   dtypes, zero-point layout, actual loaded extension/oneDNN identity, Torch,
   compiler and Intel driver/runtime versions. Capture oneDNN diagnostic output
   on failure. Separate incompatible runtime/library selection, unsupported
   descriptors and rejected deterministic implementations using that evidence.
2. Run all nine replay and six separability cases from
   `tests/test_int4_gemm_determinism.py` on the candidate/B70. Add/run an
   independent BF16 dequantize-plus-reference-matmul accuracy comparison for
   production shapes, and run upstream W4A16 FP16 reference coverage. Preserve
   supported bias/asymmetric paths when diagnosing any primitive failure.
3. Where isolation requires an attribute-on/off comparison, build two controlled
   artifacts against the **same** new dependency stack and run separate fresh
   processes with identical tensors and allocation/handle interventions. The
   patched arm must replay exactly; the unpatched arm need not fail every shape
   to justify maintaining an explicit policy. An off-only primitive success
   identifies a compatibility issue to solve, not approval to drop the policy.
4. Then qualify actual-profile model load, health, ordinary generation,
   target/MTP2/vision and repeated fresh-process requests. Service success is
   necessary and does not replace the targeted repeatability check.
5. Compare released versus candidate prefill/decode latency with the same
   deterministic setting and workload; the user wants low-concurrency speed.
   Optional controlled attribute-on/off measurements can quantify policy cost
   before discussing a policy change. No current speedup or cost is established.

The historical release audit recorded passing W4 batch-separability cases and
captured-projection replay. Those are prior evidence only. An unresolved native
primitive-creation failure must remain a release qualification blocker until the
parent obtains a working candidate or an evidence-backed replacement.
