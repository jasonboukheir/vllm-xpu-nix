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
  triton-xpu,
}:

let
  # Stable torch 2.14.0+xpu declares pyzes==0.1.2 for fork-safe device
  # discovery and XPU telemetry. Upstream pyzes hardcodes Debian's Level Zero
  # loader path, so point the tiny pure-Python binding at the Nix package.
  pyzes = python3Packages.buildPythonPackage rec {
    pname = "pyzes";
    version = "0.1.2";
    format = "wheel";

    src = fetchurl {
      url = "https://files.pythonhosted.org/packages/74/46/90e1741b3926e3b8590dfaa819439891865568fc5c90a0f5596b76ee6bff/pyzes-${version}-py3-none-any.whl";
      hash = "sha256-VjhoQ5kEmUBBQHGvr/+R7ofxiAdFezyw0xzV1aC32yM=";
    };

    postInstall = ''
      substituteInPlace "$out/${python3Packages.python.sitePackages}/pyzes.py" \
        --replace-fail \
          'libName = "/usr/lib/x86_64-linux-gnu/lib" + libName + ".so.1"' \
          'libName = "${lib.getLib level-zero}/lib/libze_loader.so.1"'
    '';

    pythonImportsCheck = [ "pyzes" ];

    meta = {
      description = "Python bindings for the Level Zero Sysman API";
      homepage = "https://pypi.org/project/pyzes/";
      license = lib.licenses.mit;
      platforms = [ "x86_64-linux" ];
    };
  };
in
python3Packages.buildPythonPackage rec {
  pname = "torch";
  # vLLM 0.31 pairs Torch 2.14 with Triton 3.8 and oneAPI 2026.1.
  version = "2.14.0+xpu";
  format = "wheel";

  src = fetchurl {
    url = "https://download-r2.pytorch.org/whl/xpu/torch-2.14.0%2Bxpu-cp312-cp312-manylinux_2_28_x86_64.whl";
    hash = "sha256-BfjxMOJcJNT3kDJmrxZRRekjT66M5z7CYTrfwBXLo0o=";
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
    pyzes
    triton-xpu
  ];

  autoPatchelfIgnoreMissingDeps = [
    "libcuda.so.1"
  ];

  dontCheckRuntimeDeps = true;

  dontStrip = true;

  postInstall = ''
    metadata="$out/${python3Packages.python.sitePackages}/torch-${version}.dist-info/METADATA"
    if [ -f "$metadata" ]; then
      sed -i -E '/^Requires-Dist: (intel-cmplr-lib-rt|intel-cmplr-lib-ur|intel-cmplr-lic-rt|intel-sycl-rt|oneccl|oneccl-devel|impi-rt|onemkl-license|onemkl-sycl-blas|onemkl-sycl-dft|onemkl-sycl-lapack|onemkl-sycl-rng|onemkl-sycl-sparse|intel-opencl-rt|intel-openmp|intel-pti|mkl|dpcpp-cpp-rt|tcmlib|umf|tbb)([^A-Za-z]|$)/d' "$metadata"
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
