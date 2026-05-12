{
  lib,
  src,
  cutlass-src,
  python3Packages,
  cmake,
  ninja,
  git,
  autoPatchelfHook,
  stdenv,
  intel-oneapi-base,
  intel-pti,
  oneccl-bmg,
  torch-xpu,
  level-zero,
  intel-compute-runtime,
  intel-graphics-compiler,
  ocl-icd,
  zlib,
  which,
  attn-kernels-xe-2,
  gdn-attn-kernels-xe-2,
  mqa-logits-kernels-xe-2,
  grouped-gemm-xe-2,
  grouped-gemm-xe-default,
  # SYCL AOT target list. Three modes:
  #   null (default) -> don't export VLLM_XPU_AOT_DEVICES /
  #     VLLM_XPU_XE2_AOT_DEVICES; upstream's CMakeLists default
  #     `pvc,bmg,bmg-g21-a0,bmg-g31-a0` kicks in.
  #   []             -> export empty string; upstream treats that
  #     as "skip AOT entirely". Kernels ship as SPIR-V and IGC
  #     specializes them at first dispatch. The 256-GRF hint is
  #     still emitted (see patches/0006-decouple-256grf-from-aot.patch)
  #     so JIT codegen matches AOT codegen quality.
  #   [ "bmg" ...]   -> AOT for the listed devices. Each entry adds
  #     one ocloc invocation at link time, so multi-device builds
  #     get expensive fast.
  aotDevices ? null,
}:

let
  syclHome = "${intel-oneapi-base}/compiler/latest";
  aotDevicesExport = lib.optionalString (aotDevices != null) ''
    export VLLM_XPU_AOT_DEVICES="${lib.concatStringsSep "," aotDevices}"
    export VLLM_XPU_XE2_AOT_DEVICES="${lib.concatStringsSep "," aotDevices}"
  '';
in
python3Packages.buildPythonPackage {
  pname = "vllm-xpu-kernels";
  version = "0.1.7-dev";
  format = "pyproject";

  inherit src;

  nativeBuildInputs = [
    cmake
    ninja
    git
    autoPatchelfHook
    which
    python3Packages.setuptools
    python3Packages.setuptools-scm
    python3Packages.wheel
    python3Packages.packaging
    python3Packages.jinja2
    python3Packages.regex
    python3Packages.psutil
    python3Packages.cmake
    python3Packages.ninja
  ];

  buildInputs = [
    stdenv.cc.cc.lib
    intel-oneapi-base
    intel-pti
    oneccl-bmg
    level-zero
    intel-compute-runtime
    intel-graphics-compiler
    ocl-icd
    zlib
    attn-kernels-xe-2
    gdn-attn-kernels-xe-2
    mqa-logits-kernels-xe-2
    grouped-gemm-xe-2
    grouped-gemm-xe-default
  ];

  propagatedBuildInputs = [
    torch-xpu
  ];

  dontUseCmakeConfigure = true;

  patches = [
    ./patches/0001-split-kernel-libs.patch
    ./patches/0002-dev-lib-override.patch
    ./patches/0003-include-project-root.patch
    ./patches/0004-skip-prebuilt-additional-libs.patch
    ./patches/0005-reduce-kernel-build-memory.patch
    ./patches/0006-decouple-256grf-from-aot.patch
  ];

  postPatch = ''
    substituteInPlace pyproject.toml \
      --replace 'torch == 2.11.0+xpu' 'torch' \
      --replace 'setuptools>=77.0.3,<80.0.0' 'setuptools'
  '';

  preBuild = ''
    mkdir -p $TMPDIR/bin
    ln -sf ${intel-compute-runtime}/bin/ocloc-* $TMPDIR/bin/ocloc
    export PATH=$TMPDIR/bin:${syclHome}/bin:$PATH
    export LD_LIBRARY_PATH=${intel-graphics-compiler}/lib:${intel-compute-runtime}/lib:$LD_LIBRARY_PATH
    export SYCL_HOME=${syclHome}
    export CMPLR_ROOT=${syclHome}
    export MKLROOT=${intel-oneapi-base}/mkl/latest
    export CC=${syclHome}/bin/icx
    export CXX=${syclHome}/bin/icpx
    icpxToolchainFlags="--gcc-toolchain=${stdenv.cc.cc} -B${stdenv.cc.libc}/lib -L${stdenv.cc.libc}/lib -L${stdenv.cc.cc.lib}/lib -idirafter ${stdenv.cc.libc.dev}/include"
    export CFLAGS="$icpxToolchainFlags $CFLAGS"
    export CXXFLAGS="$icpxToolchainFlags $CXXFLAGS"
    export LDFLAGS="-L${stdenv.cc.libc}/lib -L${stdenv.cc.cc.lib}/lib $LDFLAGS"
    export LIBRARY_PATH=${stdenv.cc.libc}/lib:${stdenv.cc.cc.lib}/lib:$LIBRARY_PATH
    export CPATH=${stdenv.cc.libc.dev}/include:$CPATH
    export CMAKE_PREFIX_PATH=${intel-oneapi-base}:$CMAKE_PREFIX_PATH
    export VLLM_CUTLASS_SRC_DIR=${cutlass-src}
    ${aotDevicesExport}
    export CMAKE_BUILD_TYPE=Release

    export VLLM_XPU_PREBUILT_ATTN_KERNELS_XE_2_LIB=${attn-kernels-xe-2}/lib/libattn_kernels_xe_2.so
    export VLLM_XPU_PREBUILT_GDN_ATTN_KERNELS_XE_2_LIB=${gdn-attn-kernels-xe-2}/lib/libgdn_attn_kernels_xe_2.so
    export VLLM_XPU_PREBUILT_MQA_LOGITS_KERNELS_XE_2_LIB=${mqa-logits-kernels-xe-2}/lib/libmqa_logits_kernels_xe_2.so
    export VLLM_XPU_PREBUILT_GROUPED_GEMM_XE_2_LIB=${grouped-gemm-xe-2}/lib/libgrouped_gemm_xe_2.so
    export VLLM_XPU_PREBUILT_GROUPED_GEMM_XE_DEFAULT_LIB=${grouped-gemm-xe-default}/lib/libgrouped_gemm_xe_default.so

    export MAX_JOBS=''${NIX_BUILD_CORES:-1}
  '';

  autoPatchelfIgnoreMissingDeps = [
    "libcuda.so.1"
  ];

  dontCheckRuntimeDeps = true;
  dontStrip = true;

  pythonImportsCheck = [ "vllm_xpu_kernels" ];

  meta = {
    description = "vLLM XPU kernels (SYCL/CUTLASS-SYCL) for Intel Arc / Battlemage / PVC";
    homepage = "https://github.com/vllm-project/vllm-xpu-kernels";
    license = lib.licenses.asl20;
    platforms = [ "x86_64-linux" ];
  };
}
