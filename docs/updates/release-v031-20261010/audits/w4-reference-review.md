# W4A16 numerical and discovery review — reference rounding and OpenCL discovery

The final composed installed runtime passes all 18 calibrated BF16 W4A16
accuracy cases and produces byte-identical native outputs to the preserved
v0.30 runtime on identical fixtures. Both M2048 production profile primitives
also execute successfully. The final loaded libraries and raw evidence hashes
are recorded in [the correctness record](../w4-correctness-review.json).
Actual server qualification remains a separate gate.

The new BF16 reference failure is an existing numerical behavior, not a v0.31
regression on the controlled fixtures. The original harness assumes round to
nearest even (RNE) when dequantizing INT4 weights to BF16. The observed native
implementation on B70 truncates these dequantized weights toward zero (RTZ).
A one-hot input isolates this before any reduction and matches RTZ exactly.
The original failed RNE diagnostic remains unchanged and must not be reported
as a pass.

Separately, the old service's OpenCL failure is now reproduced and isolated:
the exact same old-runtime M2048 down-projection fails with the ICD variable
unset and succeeds when the pinned ICD is selected. No native code or
deterministic attribute was changed for either comparison.

## Controlled runtimes and native identities

- Final composed candidate runtime:
  `/nix/store/cp0c15jk173smzp9d77kgcx36a7ibx1m-v031-brutus-qualification-env`.
  Torch `2.14.0+xpu`; oneAPI `2026.1.0.192`; oneDNN `3.13.0.N/A`.
  Native module:
  `/nix/store/vl22cdzf3wbclwy9zsfgac1mv6jb5dl9-vllm-xpu-kernels-0.1.15.4+unstable.2026.10.10.gdbb684f/lib/python3.12/site-packages/vllm_xpu_kernels/_xpu_C.abi3.so`.
  SHA-256:
  `7076ec3059ab05eb40e777f1edd194f050b78677ae3f0e370641e8cfa5937fc6`.
- Preliminary candidate runtime:
  `/nix/store/p1in9i5v8s3184xm50x392vlgcz6310w-v031-preliminary-w4-component-env`.
  Torch `2.14.0+xpu`; oneAPI 2026.1; oneDNN `3.13.0.N/A`.
  `_xpu_C.abi3.so` SHA-256:
  `2703b6388649eb63044c90428a427d7bfe3e3597b1cd69e98e7e7015a880e69a`.
- Preserved baseline runtime:
  `/nix/store/zhv84h5afvyx99fkw0l56rpj8rldhivl-v030-brutus-qualification-env`.
  Torch `2.13.0+xpu`; oneAPI 2026.0; oneDNN `3.13.0.N/A`.
  `_xpu_C.abi3.so` SHA-256:
  `585c44159565a8cbb7aaf468521d1b7cb3d43227a0eeb1a379f8e16eef9b5de3`.
- All use Arc Pro B70, driver `1.15.39122`, packaged compute runtime
  `26.27.39122.11`, group-128 symmetric INT4, BF16 activations/scales/output.
  Native diagnostic logs identify `jit:gemm:any`, `attr-fpmath:bf16:true`,
  and `attr-deterministic:1`.
- Candidate runs unset the inherited `OCL_ICD_VENDORS` and receive the package
  wrapper default. Old numerical controls explicitly select that same pinned
  provider because the old wrapper lacks a default. Module imports are
  asserted under `/nix/store`. JSON records library paths/hashes and fixtures.

This diagnostic investigation changed no source, tolerance, or services.
Each native process checked both services were stopped and the GPU was free.
GPU ownership was coordinated with the parent; processes ran sequentially.

## Original failure and baseline comparisons

`preliminary-w4-reference.json` and `.log` retain the failed original gate.
Independent diagnostic script `w4-reference-diagnose.py` regenerated the same
inputs on CPU, recorded tensor SHA-256 hashes, and compared several fixed
rounding hypotheses without using them as adaptive pass criteria.

| Shape (M,K,N), random BF16 scales | Old/new native output | RNE-BF16 reference failures | FP32-dequant reference failures | RTZ-BF16 reference failures |
| --- | --- | ---: | ---: | ---: |
| 1,5120,8192 | Identical raw bytes | 876 / 8192 | 440 / 8192 | 0 / 8192 |
| 162,5120,8192 | Identical raw bytes | 137212 / 1327104 | 68835 / 1327104 | 0 / 1327104 |

All comparisons retain `atol=rtol=0.01`. Old and new fixture hashes and native
output hashes match. This is a direct baseline comparison, not an inference
from unchanged source. The M1 output hash is
`729e420e888465571766d641e298733937f711e8d693a5ef81a9bac60ee54189`;
the M162 output hash is
`96f11bb97eae4900022c156f36f6ee256b09f6bd2cdaf31abb5a8365c8b1f205`.

The RTZ reference has three differing BF16 values in the full M1 output and
325 in M162, all within the unchanged tolerance. These are distinct from the
weight rounding issue; a CPU/GPU accumulation-order difference can affect final
BF16 rounding even after agreeing on every dequantized weight. No exact
CPU/GPU sum equivalence is claimed.

Artifacts: `w4-diagnose-{new,old}-m1.{json,log,pt}` and
`w4-diagnose-{new,old}-m162.{json,log,pt}`.

## Independent rounding isolation and source scope

The one-hot control keeps the production M1/K5120/N8192 shape, identical
packed weights and random scales, and replaces the input with a single one
at K index 128. Its output exposes that one dequantized weight row without a
nontrivial summation. Result:

- RTZ reference: **0 numerical differences and 0 raw-word differences** across
  8192 outputs.
- RNE reference: 1500 numerical differences. At output column 20, native/RTZ
  is `-0.1396484375` and RNE is `-0.140625`.
- The small differences themselves satisfy the .01 tolerance for this one-hot
  workload; the causal discriminator here is exact representation, not a
  relaxed tolerance.

Artifact: `w4-diagnose-new-basis-m1.{json,log,pt}`. The reference bit operation
clears the lower sixteen FP32 bits after exact FP32 `(code - 8) * scale`,
which implements RTZ for these finite values. Dyadic scales are a control
because the weight products are exactly representable in BF16, so RNE and RTZ
are identical.

Pinned-source evidence is consistent with this localization:

- Kernel `csrc/xpu/onednn/int4_gemm_w4a16.h` supplies BF16 math mode and BF16
  group scales to the native oneDNN primitive. This directory and oneDNN source
  pin are unchanged between the two fork bases.
- At oneDNN commit `0e2a5bfeef1bfbffc3137464606540233086ce9b`,
  `src/gpu/intel/gemm/jit/generator/pieces/quantization.cxx`,
  `gemmMake2DQuantizationLayouts` sets internal grouped scales to FP32 for BF16
  targets (around line 80). `gemmDequantizeAB` invokes
  `gemmDequantizeOperation` for the scale multiplication; the latter emits
  `emul` to the destination type. The BF16 result can therefore be rounded
  before the matrix accumulation, as the one-hot experiment observes.
- `generator/pieces/emulation.cxx:118` delegates the ordinary non-Xe3p
  multiplication to nGEN emulation; the pinned code does not expose an obvious
  universal `RTZ` policy flag at this call site. No generated ISA disassembly
  or architecture-wide rounding guarantee is claimed. The **RTZ behavior is
  established empirically for this installed B70 implementation**, rather
  than inferred from a generic oneDNN default-rounding statement.
- Upstream `tests/test_int4_gemm_onednn.py` exercises FP16 only. Reusing its
  RNE-dequantized reference with BF16 was a new harness assumption, not a
  previously established BF16 correctness contract. The original RNE/FP32
  comparisons remain useful numerical diagnostics and fail on the old stack
  as well.

This finding supports correcting the reference model for this native gate.
It does not make the historical discrepancy from fully FP32-dequantized math
disappear, prove model-quality equivalence to exact real arithmetic, or promise
that another GPU/kernel-selection path must use the same rounding.

## Complete calibrated matrix

A separate script, `w4-reference-calibrated.py`, fixes the RTZ weight model
before testing all nine original shapes with both dyadic and random scales.
It keeps the original .01 absolute/relative tolerances, records RNE and FP32
results as diagnostics, and never picks the best reference after seeing a
result. It asserts all 18 cases execute; old/new comparison also checks fixture
hashes and raw native output hashes. The original harness/results are untouched.

Preliminary candidate result `w4-calibrated-new-matrix.json`: **18/18 pass the
fixed RTZ reference**. Baseline result `w4-calibrated-old-matrix.json`: **18/18 pass**, and
**all 18 native output byte hashes match the candidate** on identical fixture
hashes. Both processes exited successfully. There are no RTZ tolerance failures
in either complete matrix. All nine dyadic cases also pass RNE and FP32
reference checks. All nine random-scale cases retain RNE/FP32 discrepancies.
Each result has a corresponding `.log` and saved CPU output `.pt` artifact.

The final installed runtime was directly rerun using the same calibrated
script and saved old-runtime outputs. `w4-final-installed-matrix.json` records
**18/18 passes**, matching fixture hashes, and **18/18 raw-byte-identical native
outputs to the old runtime**. The original .01 tolerances and all RNE/FP32
diagnostics remain unchanged. This final result is independently attested;
the composed `_xpu_C` bytes differ from the preliminary component, so its
success is established by execution rather than inferred from that component.

`w4a16-profile-primitives.json` records successful final installed native
execution for both M2048 gate-up `(2048,5120,34816)` and down-projection
`(2048,17408,5120)` shapes. These are primitive-creation/execution checks with
finite BF16 output, without a full CPU accuracy reference. Both final processes
imported the composed modules under `/nix/store`, loaded the same attested
native libraries, and selected the packaged 26.27 ICD provider. Their loaded
`libigdrcl.so` SHA-256 is
`ea81168c31b750918a588cecc249b2f687c72a8f5124d9aac955e79596ead5cd`;
the complete paths and hashes are in the correctness record. Recorded hashes
were also checked against the installed files during this review.

This closes the new-reference regression investigation for the bounded BF16,
symmetric group-128 production-shape matrix. The separately
named calibrated gate can be used while retaining the failed original RNE experiment and
its explanation. The rule is fixed RTZ dequantization for this attested native
implementation, not changing tolerance or accepting whichever reference fits.
FP16, asymmetric zero points, bias and other hardware remain outside this
specific matrix and keep their existing dedicated coverage requirements.

## Reproduced OpenCL discovery failure

`w4-diagnose-old-profile-no-icd.{json,log}` runs one native M2048/K17408/N5120
down-projection in a fresh old-runtime process with `OCL_ICD_VENDORS` unset.
It fails with the exact pre-task chain:
`OpenCL device not found` at `src/gpu/intel/sycl/utils.cpp:270`, then
`RuntimeError: could not create a primitive`.

`w4-diagnose-old-profile-pinned-icd.{json,log,pt}` reruns the identical fixture
and runtime with only the ICD provider set to the pinned 26.27 directory.
The native primitive executes successfully and returns finite BF16 output.
Both cases have identical input/packed-weight/scale hashes. The successful
profile check is an execution/discovery control, not a full CPU accuracy check.

Mapped-library evidence refines the earlier source-only OpenCL review:
both cases use oneAPI 2026.0's `libOpenCL.so.1.0.0`; the failed unset case maps
**no `libigdrcl.so`**, whereas the pinned-provider case maps the expected
26.27 `libigdrcl.so`. The earlier Nixpkgs-ocl-icd default-path observation does
not establish the default search path of this actual oneAPI loader. A version
collision with the newer host provider is unnecessary to explain the observed
failure and is not claimed here. Selecting the packaged ICD is now a direct,
controlled successful repair for this native old-runtime trigger.

Final composed-DSO identity checks and the bounded W4 gates are complete.
Actual wrapped server qualification remains necessary; these numerical and
discovery diagnostics do not establish server health or a host deployment.

Raw artifacts are retained privately in `benchmark-results/release-v031-20261010/`.
