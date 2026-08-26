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

let
  backend = python3Packages.buildPythonPackage rec {
    pname = "triton-xpu";
    version = "3.7.2";
    format = "wheel";

    src = fetchurl {
      url = "https://download-r2.pytorch.org/whl/triton_xpu-${version}-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl";
      hash = "sha256-F+iBwtuilEUIxNe++r5VkLW0Wfqoo3xXdA+FstC2LU8=";
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
      pyelftools
      setuptools
    ];

    autoPatchelfIgnoreMissingDeps = [
      "libcuda.so.1"
    ];

    dontCheckRuntimeDeps = true;

    dontStrip = true;

    pythonImportsCheck = [ "triton" ];

    meta = {
      description = "Triton ${version} Intel XPU backend";
      homepage = "https://github.com/intel/intel-xpu-backend-for-triton";
      license = lib.licenses.mit;
      platforms = [ "x86_64-linux" ];
    };
  };
in
python3Packages.buildPythonPackage rec {
  pname = "triton";
  version = "3.7.2+xpu";
  format = "wheel";

  # Official compatibility shim: provides the distribution/version vLLM pins
  # while propagating the native triton-xpu backend wheel above.
  src = fetchurl {
    url = "https://github.com/intel/intel-xpu-backend-for-triton/releases/download/v3.7.2/triton-3.7.2%2Bxpu-py3-none-any.whl";
    hash = "sha256-PIIvc+mHBRL1mm7PXcMFpLyrEfpiP5zpEBH2BDFSJ+k=";
  };

  propagatedBuildInputs = [ backend ];

  dontStrip = true;
  pythonImportsCheck = [ "triton" ];

  passthru = {
    inherit backend;
  };

  meta = {
    description = "Triton ${version} compatibility shim with Intel XPU backend";
    homepage = "https://github.com/intel/intel-xpu-backend-for-triton";
    license = lib.licenses.mit;
    platforms = [ "x86_64-linux" ];
  };
}
