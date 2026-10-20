# K01 — deterministic oneDNN W4A16 matmul

**Disposition: KEEP, unchanged.** The selected kernel base has no equivalent
determinism policy. vLLM 0.32 retains the same XPU W4A16 implementation and
dispatch. Preserve the attribute and its tests as one independent fix; the new
quantized DFlash2 workload expands qualification, not the required source delta.

## Frozen assignment

- Commit: `dcbbeac2f2453948471569de43a33ded09fae0f8`,
  `fix(onednn): make W4A16 matmul deterministic`.
- Parent, old base, and selected kernel target:
  `ddf336d86e3c8602888572a3502f951abd51df12`.
- Kernel old tip: `dbb684f4c9b38a8b3bbf52bda0c490ee47049221`.
- Independently verified stable patch ID:
  `7ea50ac3b0ceb331108743cc12894af0bd2fce23`.
- Companion vLLM old base: `db9527a46873454610df6dbedf79a36d6bf1a7f6`;
  old tip: `dcba4007ef6af8ba143d2794c194c36b58397339`;
  target v0.32.0: `5df63a7203cdb9328d824a27b682e77c7b9928be`.

Read the audit contract, kernel `AGENTS.md`, and complete paired inventory in
[inventory.json](../inventory.json). Read both the prior
[K01 audit](../../release-v031-20261010/audits/k01-deterministic-matmul.md) and
[numerical/discovery review](../../release-v031-20261010/audits/w4-reference-review.md).
No source, Git refs, services, or GPU execution were changed in this audit.

## Requirement and effective final old stack

Brutus's Arc Pro B70 runs the symmetric group-128 compressed-tensors W4A16 target
`RedHatAI/Qwen3.8-27B-INT4`, revision
`bf08f3dbd9a324e53956920aad378a1f1b6dd24a`, with BF16 compute. Identical projection
inputs and shape must use repeatable accumulation. The fix adds
`pattr.set_deterministic(true)` to `oneDNN::dnnl_matmul_w4a16_int4`, excluding
floating-point global K-parallel strategies with variable accumulation order.
This is a reproducibility requirement, with no claimed speed improvement.

The accompanying regression file exercises nine fixed-shape BF16 replay cases
(M=1,107,162), alternates cached primitive weight/scale pointers, and applies
allocation pressure. Six cases compare M=4 to four M=1 calls. Those six allow
`atol=rtol=0.015625` and record exactness separately: deterministic primitive
execution is not a guarantee of bitwise identity across different M or of
whole-model/request invariance. The native attribute also affects FP16 and
asymmetric users of this same operation.

Both changed files are byte-identical between this commit and the old fork tip.
None of k02–k07 repairs, reverts, or supersedes K01.

## Upstream coverage and changed call paths

The selected kernel old-to-new upstream delta is empty: vLLM v0.32.0's frozen
`requirements/xpu.txt` still pins `vllm_xpu_kernels==0.1.15.4`. Inspection of the
actual target implementation, rather than this identity alone, shows:

- `csrc/xpu/onednn/int4_gemm_w4a16.h:60` configures scratchpad, weight scales,
  zero points, and FP16/BF16 math mode; the upstream attribute callback never
  sets deterministic execution.
- `csrc/xpu/onednn/onednn_matmul.cpp:257` directly calls that implementation.
  `csrc/xpu/torch_bindings.cpp:48` registers the existing seven-argument
  `_xpu_C::int4_gemm_w4a16` schema without a policy argument.
- `csrc/xpu/onednn/onednn_ext.h:800` constructs the primitive-cache key from
  strides, M/N/K, batch dimension, bias and group sizes. On a miss it constructs
  a fresh `primitive_attr`, invokes the callback, and builds the primitive.
  There is no equivalent policy or bridge from Torch deterministic controls
  anywhere in the oneDNN implementation directory.
- `CMakeLists.txt:58` continues to pin custom oneDNN source
  `0e2a5bfeef1bfbffc3137464606540233086ce9b`; upgrading the Python vLLM code
  does not replace this extension with Torch's internal matmul backend.

At the exact new vLLM target:

- `model_executor/layers/quantization/compressed_tensors/schemes/`
  `compressed_tensors_wNa16.py:119` constructs an `MPLinearLayerConfig` and
  chooses a mixed-precision kernel. The XPU list in
  `model_executor/kernels/linear/__init__.py:540` remains
  `XPUW4A8IntLinearKernel, XPUwNa16LinearKernel`.
- `mixed_precision/xpu.py:20` accepts BF16/FP16, supported packed uint4 types,
  and group sizes divisible by 32. `apply_weights:91` flattens the activations
  and calls `_xpu_C.int4_gemm_w4a16` directly. The entire W4A16 class is
  structurally identical between v0.31.0 and v0.32.0.
- The only upstream change to that file is
  `acec1b1d84c428ced19d53935979107eeab922a2` (#57511), replacing
  `register_parameter("weight", None)` with `delattr` in **W4A8** repacking.
  It neither changes W4A16 dispatch nor supplies deterministic execution.
- `determinism/batch_invariant.py:1340` still overrides Aten matrix operations
  for XPU, not this custom INT4 op. `platforms/xpu.py:333` rejects global batch
  invariance for weight-quantized models and quantized KV caches. That alternate
  mode cannot replace K01 for the user's W4A16/KVarN profile.
- W4A8 changes the activation numerical contract. The CUDA/ROCm Triton W4A16
  candidates are absent from the XPU list. Neither is an established substitute.
- `models/qwen3_dflash.py:523` distinguishes unquantized context projections
  from quantized ones. The quantized branch of `_project_context_kv:621` calls
  each QKV projection through its own quant method, rather than raw `F.linear`
  over packed weights. Provided the selected drafter's quantization loads
  correctly, its quantized layers therefore share the retained native policy;
  runner V2 does not itself bypass it. Correct quantization configuration and
  checkpoint loading remain separate release checks.

Upstream `tests/test_int4_gemm_onednn.py:41` supplies FP16 reference coverage,
including quantization modes and activation ordering, but no equivalent BF16
replay gate.

## Minimal residual and dependencies

Keep the three source lines and `tests/test_int4_gemm_determinism.py` unchanged.
No source adaptation, ABI/schema change, or paired vLLM edit is required for
K01. Preserve its existing authorship and sign-off. It remains independent of
GDN synchronization, convolution, KVarN, and runner lifecycle changes; do not
fold those distinct fixes into it.

Do not add a mutable per-request deterministic toggle without changing the
primitive-cache identity and proving policy transitions: the current cache key
does not contain that attribute. A silent nondeterministic fallback on primitive
creation failure would remove the retained requirement.

## Validation observed and still required

Executed a stdlib-only audit probe with the candidate kernel repository's
`.venv/bin/python`: **8/8 checks passed**. It verified upstream absence of the
attribute, exactly one fork attribute, no subsequent changes to either assigned
file, stable patch ID, nine replay fixtures, six separability fixtures, AST
identity of the v0.31/v0.32 W4A16 class, and the exact v0.32 kernel requirement.
Also inspected the complete relevant upstream diffs, primitive creation/cache,
registration, alternative dispatch, and quantized DFlash context projection.
These checks establish source facts, not native execution or performance.

Historical evidence is stronger than the earlier unresolved primitive error:
the prior numerical/discovery review records **18/18** calibrated BF16 reference
cases passing on the final installed composed runtime, with all 18 native output
hashes matching the preserved baseline. Both M2048 production primitives also
executed. It separately reproduced the earlier primitive failure as missing
OpenCL ICD discovery and repaired it by selecting the pinned provider, without
changing K01. Its one-hot experiment established RTZ weight dequantization for
the attested B70 implementation. Preserve that fixed calibrated reference and
the original failed RNE diagnostic; do not reinterpret numerical rounding as
inter-run nondeterminism or loosen the tolerance adaptively.

No GPU test was run by this read-only agent while the parent owns the device.
Before stable publication, the parent should:

1. Run the nine replay and six separability cases against the final composed
   candidate, recording actual loaded native-library identity and pinned ICD.
   Retain the calibrated BF16 reference matrix and applicable FP16 coverage.
2. Add focused replay/reference coverage for actual DFlash2 quantized projection
   shapes and proposal batches, including different weight/scale handles and
   context precomputation. Existing target fixtures alone do not qualify every
   new drafter shape. Confirm checkpoint dispatch chooses W4A16 as intended.
3. Qualify real target-plus-drafter loading and generation at the retained
   262K limit, image handling, c1/c4, RAM prefix restore, and speculation
   acceptance/correctness. Native replay is necessary but does not imply
   whole-model stability or KV correctness.
4. Compare baseline and candidate latency/throughput with the deterministic
   policy held fixed. No attribute cost or DFlash speedup is established by this
   audit. Attribute-on/off controls, if needed to diagnose a native failure,
   require separate fresh processes and otherwise identical artifacts.
