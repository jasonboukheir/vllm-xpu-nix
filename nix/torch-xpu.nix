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
  # Stable torch 2.13.0+xpu declares pyzes==0.1.1 for fork-safe device
  # discovery and XPU telemetry. Upstream pyzes hardcodes Debian's Level Zero
  # loader path, so point the tiny pure-Python binding at the Nix package.
  pyzes = python3Packages.buildPythonPackage rec {
    pname = "pyzes";
    version = "0.1.1";
    format = "wheel";

    src = fetchurl {
      url = "https://files.pythonhosted.org/packages/51/18/2ad3193ae512d91541f8cf7c0eec076904e54a18ac0db141d5a243078602/pyzes-${version}-py3-none-any.whl";
      hash = "sha256-SqkGfX8hGznlW082viy7DtF2RcqvqFKWKfhlRV/juLM=";
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
  # Stable 2.13 XPU is the release paired with vLLM's torch 2.13 and
  # triton-xpu 3.7.2 pins. Its wheel targets the oneAPI 2026.0 ABI
  # (libsycl.so.9, oneCCL 2022.0.0). The v2.13 branch also contains the
  # BMG-specific revert of device-wide synchronization (#187423).
  #
  # The earlier Inductor Min/Max gather regression has no identified source
  # fix, so the release gate still exercises compiled Qwen3.8 graph capture
  # and generation at vocabulary extent 248320 on Brutus.
  version = "2.13.0+xpu";
  format = "wheel";

  src = fetchurl {
    url = "https://download-r2.pytorch.org/whl/xpu/torch-2.13.0%2Bxpu-cp312-cp312-manylinux_2_28_x86_64.whl";
    hash = "sha256-njm89P85dNfX0fpOJOOuTkiV2pfYdkejDG8WG5E23fw=";
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
