# Independent review — package-controlled OpenCL discovery

**Recommendation: retain the narrow wrapper correction and qualify it without
a harness-supplied OpenCL provider.** The pre-task error and exact pinned oneDNN
source strongly support a failed OpenCL discovery diagnosis. The proposed
default supplies the matching packaged provider. This review establishes the
causal mechanism and package contents, not a successful repaired execution.

Reviewed change: `nix/vllm-xpu.nix`, `makeWrapperArgs`, adds:

```nix
"--set-default OCL_ICD_VENDORS ${intel-compute-runtime}/etc/OpenCL/vendors"
```

The parent made this change; this reviewer made no source edits, started no
build/GPU process, and changed no services or refs. The other current edits in
that Nix file are outside this narrow review.

## Observed failure and exact source mechanism

The retained private `benchmark-results/release-v031-20261010/pre-task-service-journal.log`
shows the pre-existing v0.30/Torch-2.13 service at October 10 00:55:48 reporting
`Intel(R) Arc(TM) Pro B70 Graphics OpenCL device not found` from
`src/gpu/intel/sycl/utils.cpp:270`, immediately before model profiling fails with
`could not create a primitive`. The traceback reaches `qwen2_moe.py:115`, which
is explicitly `self.down_proj(out)`, then the native W4A16 operation. This
narrows the model-side location beyond the earlier gate/up-or-down hypothesis.
The retained journal is not evidence that the new candidate has failed.

The source inspected is
`/nix/store/avla7mq41rah7v3nlpb2y3a740gdhhjw-source`.
`nix hash path` gives
`sha256-wgYcZT04nL6ALG0sNkA4fjfkYag/l4CQY4P6S5TrJZo=`, exactly the fixed source
hash in `nix/mk-kernels.nix` for oneDNN commit
`0e2a5bfeef1bfbffc3137464606540233086ce9b`. Thus the following line references
are to the actual pinned source, not an assumed contemporary implementation:

- `src/gpu/intel/sycl/utils.cpp:40`, `create_ocl_engine`, handles a Level Zero
  SYCL backend by calling `sycl_dev2ocl_dev` at line 63 and creating an OpenCL
  context for the corresponding device. Level Zero selection therefore does
  not remove OpenCL discovery from every oneDNN native-kernel path.
- `src/gpu/intel/sycl/engine.cpp:176` and line 218 call that helper for cached
  kernels and program creation. This explains a failure emerging during a
  particular primitive's creation after earlier XPU work already succeeded.
- `src/gpu/intel/sycl/utils.cpp:237` builds a process-static map from OpenCL
  device UUIDs. It enumerates GPU devices, including subdevices, and later looks
  up the Level Zero device UUID.
- **The logged line 270 is the empty-map branch.** The different nonempty-map
  branch for a missing matching UUID reports at line 280. The recorded failure
  therefore indicates no usable enumerated Intel OpenCL GPU entered this map,
  not merely that an already enumerated OpenCL GPU had the wrong UUID.
- `src/xpu/ocl/utils_shared.cpp:191` returns success with an empty list for
  `CL_PLATFORM_NOT_FOUND_KHR`, filters for Intel platforms/vendor `0x8086`, and
  also returns success if it finds no devices. Consequently the empty-map
  message can result from an undiscovered or unloadable Intel provider or from
  a provider exposing no usable GPU; the message alone does not distinguish
  every underlying driver/library failure.

The cache is process-static. Any controlled provider change must start a fresh
process; changing the environment after oneDNN's first discovery is not a
valid repair test. None of this evidence implicates K01's deterministic
attribute specifically. Do not drop determinism to conceal this discovery
failure.

## ICD and wrapper evidence

The candidate's pinned compute runtime is
`/nix/store/gcwa832p0329y80drcqhslmnzhw49bh8-intel-compute-runtime-26.27.39122.11`.
Its `etc/OpenCL/vendors/intel-neo.icd` exists and contains the absolute path:

```text
/nix/store/gcwa832p0329y80drcqhslmnzhw49bh8-intel-compute-runtime-26.27.39122.11/lib/intel-opencl/libigdrcl.so
```

The ICD file SHA-256 is
`24f498243b49f1283390200389da5d5355b9e1a86ee911439f157effc61dc589`.
The target shared library exists. Read-only ELF inspection shows its RUNPATH
includes the matching packaged graphics compiler `2.38.2`, gmmlib and related
libraries. This proves a concrete provider is supplied by the proposed directory;
it does not prove device enumeration or native execution succeeds.

The installed failing v0.30 package's `bin/vllm` wrapper was also inspected. It
has no `OCL_ICD_VENDORS` assignment, while its `LD_LIBRARY_PATH` includes the
26.27 compute-runtime root `lib`, the separate Level Zero driver output and
graphics compiler `2.38.2`. A shared-library search path does not itself select
the `etc/OpenCL/vendors` registration directory. The Intel OpenCL library is
additionally under `lib/intel-opencl`, with an absolute path supplied by the ICD.

There **is** a system ICD today: `/run/opengl-driver/etc/OpenCL/vendors/intel-neo.icd`
points to a different compute runtime,
`/nix/store/89m4c8k7lnww6va8aj6ibb0vs51c8q21-intel-compute-runtime-26.31.39395.13/lib/intel-opencl/libigdrcl.so`.
The Nixpkgs ocl-icd package configuration inspected in available source copies
uses `/run/opengl-driver/etc/OpenCL/vendors` as its custom default directory.
It would be inaccurate to describe this host as simply having no ICD file.
The evidenced packaging omission is failure to select its own matched provider;
mixing a host provider with the package's library paths is a credible explanation
for the empty enumeration. This review did not reproduce or prove the precise
loader/version-conflict mechanism in the old service process.

The proposed wrapper default is appropriately scoped to the packaged
application, keeps a supplied override, and is consistent with the provider
choice already made by `nix/quantize.nix`. One small semantic distinction:
the inspected `makeWrapper --set-default` implementation uses `${VAR-default}`
and preserves an explicitly empty value, while quantize uses `${VAR:-default}`
and replaces empty values. Testing with `env -u OCL_ICD_VENDORS` correctly
exercises the proposed default. No need to broaden this repair based on a
hypothetical empty override absent from the actual host configuration.

`qualification-packages.nix` builds the Python qualification wrapper from the
candidate's `makeWrapperArgs`, filtering only the PYTHONPATH argument. The ICD
default will therefore reach those Python-native checks as well as `bin/vllm`.
The current `qualification-environment.json` has no ICD override and explicitly
records that discovery must come from the package. The native batch launcher
also removes inherited `OCL_ICD_VENDORS`. Those changes address the previous
harness's ability to mask an incomplete serving wrapper.

## Required validation before claiming the repair works

1. Inspect the completed `bin/vllm` and qualification-Python wrappers. In a
   fresh invocation with `OCL_ICD_VENDORS` unset, record that the effective value
   is the exact pinned directory above. A Python environment print needs no GPU.
   Record preservation of a deliberately supplied value separately without
   running GPU work under an invalid override.
2. Run the focused M=2048 W4A16 profile gates in fresh processes with the parent
   environment's ICD variable removed. Prioritize MLP down, K=17408/N=5120;
   also run gate/up, K=5120/N=34816. Require actual native primitive execution,
   finite BF16 output, then the nine-shape accuracy and retained replay checks.
   Keep the deterministic policy unchanged.
3. Record effective `OCL_ICD_VENDORS`, the actual mapped `libOpenCL`,
   `libigdrcl.so`, Level Zero and compiler libraries, extension hashes and
   oneDNN identity. The prepared W4 reference harness currently records many
   identities but does not include the ICD environment variable or
   `libigdrcl.so` in its map filter; supplement that evidence when executing it.
4. Launch the actual host-equivalent foreground `bin/vllm` command with the
   ICD variable unset and the real model/profile. Require successful profile
   run, server health/model discovery, generation, MTP and the planned workload
   checks. A direct Python-native pass alone does not exercise service wrapping
   and worker environment inheritance.
5. If discovery still fails, capture oneDNN verbose diagnostics on a fresh
   diagnostic retry and inspect the effective provider/library maps. A controlled
   old-versus-fixed wrapper comparison would strengthen causal attribution but
   is not already observed here. No production restart is required merely to
   reproduce the earlier baseline.

Observed review checks were source-hash verification, exact failure-path/source
inspection, ICD contents and ELF RUNPATH inspection, old-wrapper inspection and
qualification-wrapper/environment review. A read-only Nix daemon closure query
was unavailable in the sandbox; it was unnecessary for inspecting the existing
store files and no escalation was attempted. GPU/build/serving qualification
remains with the parent. No repaired-run success is claimed by this report.
