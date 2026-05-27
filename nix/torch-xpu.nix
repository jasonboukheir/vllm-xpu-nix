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
  # 2026-05-21 nightly: first torch+xpu line linked against the new oneAPI
  # 2026.0 ABI (libsycl.so.9, libmkl_*.so.3 / libmkl_sycl_*.so.6). Stable
  # 2.11 / 2.12 GA wheels still link libsycl.so.8 + libmkl_*.so.2 / .so.5
  # and cannot be patchelfed against the unified 2026.0 toolkit. Hold here
  # until a 2.13+ GA wheel against oneAPI 2026.0 lands; revisit on each
  # toolkit bump.
  version = "2.13.0.dev20260521+xpu";
  format = "wheel";

  src = fetchurl {
    url = "https://download.pytorch.org/whl/nightly/xpu/torch-2.13.0.dev20260521%2Bxpu-cp312-cp312-manylinux_2_28_x86_64.whl";
    hash = "sha256-DvCjjlkoqo2hmC1biLINIA/VDEWG7Fu49Wq2bZNgrf0=";
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
