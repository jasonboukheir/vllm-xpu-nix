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
}:

{
  libName,
  featureFlags ? [ ],
  # SYCL AOT target list. See vllm-xpu-kernels.nix for the three
  # modes (null = upstream default; [] = JIT; non-empty list = AOT
  # for the listed devices).
  aotDevices ? null,
}:

let
  syclHome = "${intel-oneapi-base}/compiler/latest";
  aotDevicesExport = lib.optionalString (aotDevices != null) ''
    export VLLM_XPU_AOT_DEVICES="${lib.concatStringsSep "," aotDevices}"
    export VLLM_XPU_XE2_AOT_DEVICES="${lib.concatStringsSep "," aotDevices}"
  '';
in
stdenv.mkDerivation {
  pname = "vllm-xpu-${lib.replaceStrings [ "_" ] [ "-" ] libName}";
  version = "0.1.7-dev";

  inherit src;

  patches = [
    ./patches/0001-split-kernel-libs.patch
    ./patches/0005-reduce-kernel-build-memory.patch
    ./patches/0006-decouple-256grf-from-aot.patch
  ];

  nativeBuildInputs = [
    cmake
    ninja
    git
    autoPatchelfHook
    which
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
    torch-xpu
    python3Packages.python
  ];

  dontUseCmakeConfigure = false;
  cmakeBuildType = "Release";
  enableParallelBuilding = true;

  # Content-addressed: the kernel .so is mostly SYCL device-image binary
  # produced by the SYCL-TLA compile pipeline. RUNPATH does encode some
  # store-path inputs (torch-xpu, intel-oneapi), so torch-xpu bumps will
  # invalidate; smaller upstream churn (nativeBuildInputs, helper tools)
  # leaves the .so byte-identical and the CA hash hits.
  __contentAddressed = true;

  # Each SYCL-TLA template instantiation peaks ~5 GiB RSS in icpx, with
  # the heavier head-dim/policy combos pushing ~40 GiB. ninja -j$(nproc)
  # on a 24-core box stacks ~24 of these and OOM-kills the build long
  # before any head128/b16 TU finishes. Match the dev-shell default
  # (MAX_JOBS=2) so multiple lib drvs can still run concurrently under
  # Nix's outer scheduler without overrunning a 96 GiB box. Raise via
  # cores= override on builders with more RAM headroom.
  preBuild = ''
    export NIX_BUILD_CORES=2
  '';

  preConfigure = ''
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
    ${aotDevicesExport}
  '';

  cmakeFlags = [
    "-GNinja"
    "-DVLLM_XPU_LIBS_ONLY=ON"
    "-DVLLM_PYTHON_EXECUTABLE=${python3Packages.python}/bin/python"
    "-DVLLM_CUTLASS_SRC_DIR=${cutlass-src}"
    "-DCMAKE_BUILD_TYPE=Release"
    "-DBUILD_SYCL_TLA_KERNELS=ON"
  ]
  ++ featureFlags;

  ninjaFlags = [ libName ];

  dontInstall = false;
  installPhase = ''
    runHook preInstall
    mkdir -p $out/lib
    cp lib${libName}.so $out/lib/
    runHook postInstall
  '';

  autoPatchelfIgnoreMissingDeps = [ "libcuda.so.1" ];
  dontStrip = true;

  meta = {
    description = "vLLM XPU kernel SHARED lib: ${libName}";
    homepage = "https://github.com/vllm-project/vllm-xpu-kernels";
    license = lib.licenses.asl20;
    platforms = [ "x86_64-linux" ];
  };
}
