{
  lib,
  src,
  version,
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
  ccache,
  attn-kernels-xe-2,
  gdn-attn-kernels-xe-2,
  mqa-logits-kernels-xe-2,
  grouped-gemm-xe-2,
  grouped-gemm-xe-default,
  # SYCL AOT target list. Exported as VLLM_XPU_AOT_DEVICES /
  # VLLM_XPU_XE2_AOT_DEVICES (upstream's CMakeLists honours both at
  # ~line 186).
  #   [] (default) -> empty-string export; upstream skips AOT.
  #     Kernels ship as SPIR-V and IGC specializes them at first
  #     dispatch. The 256-GRF hint is still emitted via
  #     patches/0006-decouple-256grf-from-aot.patch so JIT codegen
  #     matches AOT codegen quality.
  #   [ "bmg" ...] -> AOT for the listed devices. Each entry adds
  #     one ocloc invocation at link time, so multi-device builds
  #     get expensive fast.
  aotDevices ? [ ],
  # Same toggle as vllm-xpu-lib.nix. Upstream setup.py auto-detects
  # ccache via `which("ccache")`, so having ccache in nativeBuildInputs
  # is enough to flip on -DCMAKE_{C,CXX}_COMPILER_LAUNCHER=ccache.
  useCcache ? true,
}:

let
  syclHome = "${intel-oneapi-base}/compiler/latest";
  aotDevicesStr = lib.concatStringsSep "," aotDevices;

  # See vllm-xpu-lib.nix for the full rationale on these values and
  # why they live on the derivation (rather than impureEnvVars).
  ccacheEnvAttrs = lib.optionalAttrs useCcache {
    CCACHE_DIR = "/var/cache/ccache";
    CCACHE_COMPRESS = "1";
    CCACHE_SLOPPINESS = "random_seed,time_macros,include_file_mtime,include_file_ctime,pch_defines";
    CCACHE_NOHASHDIR = "1";
    CCACHE_UMASK = "007";
  };

  ccachePreBuild = lib.optionalString useCcache ''
    export CCACHE_BASEDIR=$NIX_BUILD_TOP
  '';
in
python3Packages.buildPythonPackage ({
  pname = "vllm-xpu-kernels";
  inherit version;
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
  ] ++ lib.optional useCcache ccache;

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
    # Drop the upstream torch release pin (any version) — we build against the
    # torch-xpu nightly this flake provides, not the kernels' pinned wheel.
    # Version-agnostic so an upstream pin bump (e.g. #288: 2.11 -> 2.12) does
    # not silently no-op and leave a stale pin that fails the dep check.
    sed -i -E 's/torch == [0-9.]+\+xpu/torch/' pyproject.toml
    substituteInPlace pyproject.toml \
      --replace 'setuptools>=77.0.3,<80.0.0' 'setuptools'
  '';

  preBuild = ''
    ${ccachePreBuild}
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
    export VLLM_XPU_AOT_DEVICES="${aotDevicesStr}"
    export VLLM_XPU_XE2_AOT_DEVICES="${aotDevicesStr}"
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
} // ccacheEnvAttrs)
