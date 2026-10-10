# Packaged OpenCL discovery

Before this task, the deployed xpu-v1.11.0 worker failed during its M2048 model
profile with `RuntimeError: could not create a primitive`. The oneDNN diagnostic
immediately before it was `Intel(R) Arc(TM) Pro B70 Graphics OpenCL device not
found`, at `src/gpu/intel/sycl/utils.cpp:270`. The old service was already stopped
and remained stopped throughout preparation.

The wrapper already exposed the pinned Level Zero, compute-runtime and IGC
libraries through LD_LIBRARY_PATH, but did not select a matching OpenCL ICD
provider. The host's registered provider is compute-runtime 26.31.39395.13;
the package uses 26.27.39122.11. The reported oneDNN source line means its
OpenCL UUID device map was empty. Fresh old-runtime controls now reproduce
the original failure with the ICD variable unset and pass the identical
fixture with only the pinned provider selected. Both use oneAPI's OpenCL
loader; the failed run maps no libigdrcl, while the passing run maps the
pinned libigdrcl. A collision with the newer host provider is not needed to
explain this measured discovery failure.
The native reference harness had previously supplied OCL_ICD_VENDORS explicitly;
that test-only environment did not fix ordinary systemd startup.

The candidate now defaults OCL_ICD_VENDORS to the pinned compute-runtime's
`etc/OpenCL/vendors` directory, consistent with the existing quantization wrapper.
No driver version changes. An explicitly configured operator value can still
override the default. Native qualification and the Brutus-derived foreground
launcher remove inherited harness overrides, so they exercise packaged discovery.

Both M2048 MLP primitives passed on B70 in a preliminary environment containing
the rebuilt candidate base extension and its wrapper defaults, with no externally
supplied OCL setting. Process maps show the matching pinned libigdrcl. This checks
component startup. Both primitives also passed in the composed runtime. The
final package then loaded the actual model and passed HTTP health, chat,
reasoning, image and tool-formatting checks without an external ICD override.

A separate numerical check initially failed because its BF16 dequantization
reference assumed round-to-nearest-even. A one-hot control isolates the native
implementation's truncation toward zero. All 18 cases pass the separately fixed
RTZ reference at unchanged tolerances, and all 18 old/new native outputs match
byte for byte. The original RNE/FP32 discrepancies remain recorded. See the
[numerical review](audits/w4-reference-review.md).

Status: component and composed-runtime primitives, calibrated numerical controls
and real-model startup/short requests passed. The complete foreground suite is
recorded separately in foreground-qualification-review.json.
Private original diagnostic: `benchmark-results/release-v031-20261010/pre-task-service-journal.log`.
