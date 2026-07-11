{
  lib,
  fetchurl,
  python3Packages,
  autoPatchelfHook,
  stdenv,
  intel-oneapi-base,
  intel-pti,
  level-zero,
  intel-compute-runtime,
  ocl-icd,
  zlib,
}:

python3Packages.buildPythonPackage rec {
  pname = "torch";
  # 2026-05-24 XPU nightly: linked against the oneAPI 2026.0 ABI
  # (libsycl.so.9, libmkl_*.so.3 / libmkl_sycl_*.so.6, oneccl 2022.0.0).
  # Stable 2.11 / 2.12 GA wheels still link the 2025.x ABI (libsycl.so.8 +
  # libmkl_*.so.2 / .so.5) so cannot be patchelfed against the unified
  # 2026.0 toolkit. Tracks nightly until a 2.13+ GA wheel against oneAPI
  # 2026.0 ships; revisit on each toolkit bump.
  #
  # Motivation for living on nightly: the 2026.0 SYCL runtime ships the
  # work-group scratch-memory + SYCL Graph extension fixes that let vLLM
  # capture FULL decode graphs (compilation_config.cudagraph_mode =
  # FULL_AND_PIECEWISE). On 2025.3 + torch 2.12 stable the FA2 varlen
  # kernel trips `sycl_ext_oneapi_work_group_scratch_memory feature is
  # not yet available for use with the SYCL Graph extension` and forces
  # cudagraphMode = "PIECEWISE" as a workaround.
  #
  # Pinned at 2026-05-24, the latest nightly that avoids BOTH known
  # XPU-breaking regressions in this window:
  #
  #  1. pytorch#182630 / 2a81e91563 "Add device-wide synchronization"
  #     (merged 2026-05-29, first in dev20260529). Makes
  #     c10::xpu::syncStreamsOnDevice prefer
  #     device.ext_oneapi_wait_and_throw() on SYCL >= 2026 when the device
  #     exposes the ext_oneapi_device_wait aspect. Arc/BMG on libsycl.so.9
  #     (oneAPI 2026.0.0.198) advertises the aspect but its device::wait()
  #     UAFs urQueueRelease after any oneCCL all_reduce, segfaulting every
  #     torch.xpu.synchronize() once xccl is initialised.  -> need <= 0528.
  #
  #  2. pytorch#184589/#184592 "Use PyTorch Min/Max in Inductor index
  #     propagation" (entered dev20260526/0527). Changes inductor gather
  #     index-bound propagation and drops a clamp, so a torch.compile
  #     `expand_kernel` gathers out of bounds and the Level-Zero driver
  #     aborts with `index out of bounds < 248320` (NEO AssertHandler) on
  #     Xe2/BMG. No upstream fix on main as of 2026-06-03.  -> need <= 0525.
  #
  # 0524 satisfies both. Revisit when oneAPI 2026.1 (or a fixed 2026.0.x)
  # ships a non-UAF ext_oneapi_device_wait AND the inductor Min/Max
  # regression is fixed, then re-bump to the latest nightly.
  version = "2.13.0.dev20260524+xpu";
  format = "wheel";

  src = fetchurl {
    url = "https://download.pytorch.org/whl/nightly/xpu/torch-2.13.0.dev20260524%2Bxpu-cp312-cp312-manylinux_2_28_x86_64.whl";
    hash = "sha256-V+Qsm8sHhFboaZrVGlKfXWOE0akGKU4oALH3uz0Viqk=";
  };

  nativeBuildInputs = [
    autoPatchelfHook
  ];

  buildInputs = [
    stdenv.cc.cc.lib
    intel-oneapi-base
    intel-pti
    level-zero
    intel-compute-runtime
    ocl-icd
    zlib
  ];

  propagatedBuildInputs = with python3Packages; [
    filelock
    typing-extensions
    sympy
    networkx
    jinja2
    fsspec
    setuptools
    numpy
  ];

  autoPatchelfIgnoreMissingDeps = [
    "libcuda.so.1"
  ];

  dontCheckRuntimeDeps = true;

  dontStrip = true;

  postInstall = ''
    metadata="$out/${python3Packages.python.sitePackages}/torch-${version}.dist-info/METADATA"
    if [ -f "$metadata" ]; then
      sed -i -E '/^Requires-Dist: (intel-cmplr-lib-rt|intel-cmplr-lib-ur|intel-cmplr-lic-rt|intel-sycl-rt|oneccl|oneccl-devel|impi-rt|onemkl-license|onemkl-sycl-blas|onemkl-sycl-dft|onemkl-sycl-lapack|onemkl-sycl-rng|onemkl-sycl-sparse|intel-opencl-rt|intel-openmp|intel-pti|mkl|dpcpp-cpp-rt|tcmlib|umf|tbb|triton-xpu)([^A-Za-z]|$)/d' "$metadata"
      # The wheel bounds `setuptools<82` (added preemptively for the
      # pkg_resources removal in setuptools 82) but the pinned nixpkgs ships
      # 82.x; pypa build's --no-isolation dependency check in downstream
      # builds (vllm-xpu-kernels, vllm-xpu) validates transitive
      # requirements and refuses the env over it. Upstream already removed
      # the bound — torch.utils.cpp_extension never used pkg_resources
      # (https://github.com/pytorch/pytorch/pull/187262) — the pinned
      # nightly just predates that. Done via sed because
      # pythonRelaxDepsHook mishandles the wheel's +xpu local version when
      # locating the dist-info directory.
      # TODO: drop when the pinned nightly moves past pytorch#187262
      sed -i 's/^Requires-Dist: setuptools<82$/Requires-Dist: setuptools/' "$metadata"
    fi
  '';

  pythonImportsCheck = [ "torch" ];

  # Stock nixpkgs torch exposes these for downstream consumers (notably
  # torchvision) that do `inherit (torch) cudaCapabilities cudaPackages
  # cudaSupport;`. torch-xpu has no CUDA, so stub them with the same
  # cudaSupport=false defaults a CPU-only torch would carry.
  passthru = {
    cudaSupport = false;
    cudaCapabilities = [ ];
    cudaPackages = { };
    rocmSupport = false;
  };

  meta = {
    description = "PyTorch ${version} with Intel XPU (SYCL/Level-Zero) backend";
    homepage = "https://pytorch.org";
    license = lib.licenses.bsd3;
    platforms = [ "x86_64-linux" ];
  };
}
