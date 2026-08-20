{
  lib,
  writeShellApplication,
  python3Packages,
  auto-round-xpu,
  llm-compressor-xpu,
  intel-oneapi-base,
  level-zero,
  intel-graphics-compiler,
  intel-compute-runtime,
  ocl-icd,
  stdenv,
}: let
  autoRoundEnv = python3Packages.python.withPackages (_: [
    auto-round-xpu
    llm-compressor-xpu
    python3Packages.hf-xet
  ]);
in
  writeShellApplication {
    name = "quantize";
    runtimeInputs = [autoRoundEnv];
    text = ''
      export LD_LIBRARY_PATH=${lib.makeLibraryPath [
        level-zero
        intel-graphics-compiler
        intel-compute-runtime
        intel-compute-runtime.drivers
        ocl-icd
      ]}''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
      export OCL_ICD_VENDORS="''${OCL_ICD_VENDORS:-${intel-compute-runtime}/etc/OpenCL/vendors}"
      export ONEAPI_ROOT="''${ONEAPI_ROOT:-${intel-oneapi-base}}"
      export SYCL_HOME="''${SYCL_HOME:-${intel-oneapi-base}/compiler/latest}"
      export CMPLR_ROOT="''${CMPLR_ROOT:-${intel-oneapi-base}/compiler/latest}"
      export LEVEL_ZERO_V1_SDK_PATH="''${LEVEL_ZERO_V1_SDK_PATH:-${level-zero}}"
      export LIBRARY_PATH=${level-zero}/lib''${LIBRARY_PATH:+:$LIBRARY_PATH}
      export CC="''${CC:-${stdenv.cc}/bin/cc}"
      export CXX="''${CXX:-${stdenv.cc}/bin/c++}"
      export PATH=${intel-compute-runtime}/bin:$PATH
      export PYTHONPATH=${../scripts}''${PYTHONPATH:+:$PYTHONPATH}
      export QUANTIZE_LLMCOMPRESSOR_SCRIPT=${../scripts/llmcompressor_quantize.py}
      export QUANTIZE_TOOLCHAIN_ID=${autoRoundEnv}
      exec ${autoRoundEnv}/bin/python ${../scripts/quantize.py} "$@"
    '';
  }
