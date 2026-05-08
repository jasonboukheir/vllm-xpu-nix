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
  pname = "triton-xpu";
  version = "3.7.0";
  format = "wheel";

  src = fetchurl {
    url = "https://download.pytorch.org/whl/triton_xpu-${version}-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl";
    hash = "sha256-pmY+vkPjwNVg/3dHCGMtenUgjuZKKRwXJO1cFqktHHI=";
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
    description = "Triton 3.7.0 with Intel XPU backend";
    homepage = "https://github.com/intel/intel-xpu-backend-for-triton";
    license = lib.licenses.mit;
    platforms = [ "x86_64-linux" ];
  };
}
