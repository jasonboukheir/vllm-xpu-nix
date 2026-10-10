# Independent packaging closure review

Reviewer: dedicated `v05_kvarn` agent, read-only follow-up after replay.

No new release blockers found. The review compared the frozen v0.30 to v0.31
common/XPU requirement delta against source, built dependency metadata, prior
substrate logs, and offline Nix evaluation of the locked closure and host override.

- Torch 2.14.0+xpu, Triton 3.8.0+xpu, torchvision 0.29.0+xpu, kernels 0.1.15.4,
  xgrammar 0.2.7 and oss-harmony 0.0.11 match the new requirements. Existing
  Transformers 5.14.1 meets the new upper bound of 5.18.
- Evaluated PYTHONPATH has one Torch and only oss-harmony, without an
  openai-harmony collision. Built metadata pairs Torch, torchvision, Triton and
  pyzes correctly. Substrate logs resolve libsycl.so.9 and PTI1 into the new toolkit.
- The selected ICD file exists and references the package's pinned compute
  runtime. The host unit has no OCL override, and the wrapper environment reaches
  subprocesses. The prepared launcher clears ambient PYTHONPATH and OCL overrides.
- Exact host candidate and kernel derivations match candidate-derivations.json.
  The host specialization, vision, eager V1, K4V2 and MTP2 settings are preserved.

The package-level vLLM import check is lazy. This review did not run candidate
imports, builds, GPU operations or API requests; installed CPU, native and actual
service qualification remain separate gates. The previously documented FastAPI
constraint relaxation remains an explicit qualification limitation.

## Build-metadata finding after the static review

The candidate Python build exposed a **release blocker missed by the initial
review above**. Upstream v0.31's `pyproject.toml` still declares the common CUDA
build requirement `torch == 2.13.0`, while its `requirements/xpu.txt` and this
candidate correctly select `torch == 2.14.0` / `2.14.0+xpu`. The package's
existing `postPatch` did not reconcile these separate build and runtime pins.

The actual `pypaBuildPhase` completed `get_requires_for_build_wheel` / `egg_info`
then rejected the environment before building the wheel:

```text
ERROR Unmet dependencies:
    torch==2.13.0
            wanted: ==2.13.0
            found: 2.14.0+xpu
```

The OpenCL-enabled attempt failed in resolved derivation
`/nix/store/gif711rhaqi61dnv5cl5li74pmcclwbb-python3.12-vllm-xpu-0.31.0+unstable.2026.10.10.gef92dc5.drv`.
Retained evidence is `benchmark-results/release-v031-20261010/candidate-build-opencl.log`,
SHA256 `15be99695c927187c01c8c0f8ca20b0b68cf4603e134c03b22ed0a38f1d7ad82`.
The native build's success does not resolve this Python metadata failure.

The release owner's repair in `nix/vllm-xpu.nix` is appropriate: for v0.31 and
later, require the exact `torch==<selected public version>` line in the upstream
XPU requirements, then replace the known common build pin with that same exact
version. `substituteInPlace --replace-fail` preserves a failure if the expected
build metadata changes. The selected public version is `2.14.0`; PEP 440's
`==2.14.0` accepts the selected `2.14.0+xpu` wheel. Keep build dependency checks
enabled. `--no-isolation` does not disable them, and Nix's `pythonRelaxDeps` hook
operates on wheel metadata after the build, too late to repair this failure.

Read-only, in-memory validation of the actual proposed replacement confirmed
exactly one Torch entry in each source, the resulting exact pin's acceptance of
`2.14.0+xpu`, rejection of `2.13.0+xpu`, and preservation of all other build
requirements. No additional likely build-metadata blocker was identified:
the build frontend reported only Torch unmet after the existing setuptools and
Rust transformations; the remaining CMake, Ninja, packaging, setuptools-scm,
wheel and Jinja requirements are supplied by native build inputs. Rust stubs
and `setup.py` metadata generation already executed successfully in this failed
attempt. The repaired wheel build and installed-runtime qualification remain
pending; this follow-up changed only this audit document and used no GPU.

## oneCCL plugin discovery and final CPU rerun

After the Torch build-pin repair, the candidate built and its installed CPU
suite passed. Actual server startup then exposed a separate runtime blocker:
Torch 2.14's oneCCL 2 C API could load `libccl.so.2` but could not discover
`libccl_legacy.so` by name. It reported `Could not load any plugin` before model
loading. This dynamic plugin lookup was not exercised by the package import
check or the CPU tests.

The release owner's isolated single-rank XCCL probe reproduced the failure.
Adding the existing pinned toolkit's `ccl/latest/lib` directory to
`LD_LIBRARY_PATH` enabled initialization and an XPU all-reduce; repeating the
probe through the rebuilt final wrapper also passed. This reviewer read those
existing probe logs without running GPU operations. The final probe maps show
the toolkit's `libccl.so.2.0`, `libccl.so.1.0`, and `libccl_legacy.so`, together
with the existing Nix libfabric. The missing optional `libccl_null.so` message
does not prevent the successfully selected legacy plugin from running.

The one-line wrapper prefix in `nix/vllm-xpu.nix` is appropriate. Both the
installed vLLM launcher and qualification Python wrapper contain the same
pinned `ccl/latest/lib` path, where the required plugin exists. This preserves
the existing host oneCCL environment and makes the path available to spawned
workers. No source revision, native kernel output, or dependency pin changed.

Retained parent-run probe evidence under
`benchmark-results/release-v031-20261010/`:

| Log | SHA256 |
| --- | --- |
| `xccl-probe-unfixed.log` | `dd111b808d512ecfe9edebd5d2e49b2cd80bc7fae415f64c62640ac0377bdf71` |
| `xccl-probe-library-path.log` | `a419f0cc9d1aee226f1a60e0c8e44cc0041ee1080a8de931eb8af22f36b44ce7` |
| `xccl-probe-final-wrapper.log` | `529a2ed0d7c8d5e991ee99bb9b5c67ba43221d7c316b351cd8c7d1591e1f54e1` |

The final CPU rerun used package
`/nix/store/mcyp3ljp8yl1hwfllnfzf22mw3j1z42y-python3.12-vllm-xpu-0.31.0+unstable.2026.10.10.gef92dc5`
through runtime
`/nix/store/x70lv8ammab6cam04rz95gqzf202bz22-v031-brutus-qualification-env`.
It passed **660 unique tests**, with the same **14 CUDA-only setup skips** and
zero failures. All 915 recorded vLLM module files belong to the final installed
package; their recorded Python hashes match the preceding candidate exactly.
[The installed CPU review](../installed-cpu-review.json) now identifies the
final runtime and retains the earlier run's identities and evidence hashes.
Counts are for the final invocation alone. Native and service qualification
remain separately recorded by the release owner.
