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
  sccache,
}:

{
  libName,
  featureFlags ? [ ],
  # SYCL AOT target list. See vllm-xpu-kernels.nix: [] (default)
  # is JIT, non-empty list is AOT for those devices.
  aotDevices ? [ ],
}:

let
  syclHome = "${intel-oneapi-base}/compiler/latest";
  aotDevicesStr = lib.concatStringsSep "," aotDevices;
in
stdenv.mkDerivation {
  pname = "vllm-xpu-${lib.replaceStrings [ "_" ] [ "-" ] libName}";
  version = "0.1.7-dev";

  inherit src;

  patches = [
    ./patches/0001-split-kernel-libs.patch
    ./patches/0005-reduce-kernel-build-memory.patch
    ./patches/0006-decouple-256grf-from-aot.patch
    # 0007 widens the FA2 Cartesian sweep (Q/KV dtype as template params)
    # so each TU instantiates one SYCL pipeline instead of six, dropping
    # per-TU peak RSS from ~40 GB to ~7 GB on the worst-case attn TU.
    # Inert for non-attn libs (FA2_KERNELS_ENABLED=OFF skips the
    # affected sources entirely); applied here so the attn lib variant
    # picks it up.
    ./patches/0007-fa2-dtype-split.patch
    # 0008 splits the chunk_prefill / paged_decode dispatcher TUs
    # (fmha_xe2.cpp, paged_decode_xe2.cpp) into per-policy shards so the
    # entry TUs no longer parse the full CUTLASS / SYCL kernel pipeline
    # headers and the recursive trampoline forest distributes across 6 +
    # 12 generated .cpps. Closes the per-dispatcher-TU OOM that survived
    # 0007 at -j$(nproc) on a 24-core / 96-GiB box.
    ./patches/0008-fa2-dispatcher-split.patch
  ];

  nativeBuildInputs = [
    cmake
    ninja
    git
    autoPatchelfHook
    which
    sccache
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

  # sccache is wired as the C/C++ compiler launcher. Local (SCCACHE_DIR)
  # and remote (SCCACHE_BUCKET / SCCACHE_REDIS) backends both work; the
  # sandbox inherits credentials via impureEnvVars below. With S3 as the
  # backend the warm-cache rebuild for an unchanged TU is a single GET
  # per .o — the cache property that matters for iteration speed without
  # paying the overhead of per-TU Nix derivations.
  impureEnvVars = [
    "SCCACHE_DIR"
    "SCCACHE_BUCKET"
    "SCCACHE_REGION"
    "SCCACHE_ENDPOINT"
    "SCCACHE_S3_USE_SSL"
    "SCCACHE_S3_KEY_PREFIX"
    "SCCACHE_S3_NO_CREDENTIALS"
    "SCCACHE_REDIS"
    "SCCACHE_REDIS_KEY_PREFIX"
    "AWS_ACCESS_KEY_ID"
    "AWS_SECRET_ACCESS_KEY"
    "AWS_SESSION_TOKEN"
  ];

  # NIX_BUILD_CORES is inherited from the daemon (defaults to `nproc`).
  # After 0007-fa2-dtype-split.patch the kernel_template TUs peak at
  # ~7 GiB RSS; after 0008-fa2-dispatcher-split.patch the dispatcher TUs
  # (previously the OOM hotspot at -j$(nproc)) split into per-policy
  # shards that sit below that ceiling. A fat-RAM box can run the full
  # nproc fan-out. Cap with `nix build --cores N` (or `cores = N` in
  # nix.conf) on memory-constrained hosts; that value also reaches
  # -fsycl-max-parallel-link-jobs via the cmakeFlagsArray append below.
  preConfigure = ''
    # sccache falls back to $HOME/.cache/sccache when SCCACHE_DIR isn't
    # set; the Nix sandbox HOME (/homeless-shelter) is unwritable, so
    # the fallback errors at mkdir time even when the user has S3/Redis
    # configured. Point HOME at $TMPDIR so the fallback lands somewhere
    # writable. Persistent caching still works via SCCACHE_* inherited
    # through impureEnvVars.
    export HOME=$TMPDIR

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
    export VLLM_XPU_AOT_DEVICES="${aotDevicesStr}"
    export VLLM_XPU_XE2_AOT_DEVICES="${aotDevicesStr}"

    # cmakeFlagsArray, not cmakeFlags: the latter is word-split without
    # recursive expansion, so $NIX_BUILD_CORES would reach icpx literal.
    cmakeFlagsArray+=("-DVLLM_XPU_SYCL_LINK_PARALLELISM=$NIX_BUILD_CORES")
  '';

  cmakeFlags = [
    "-GNinja"
    "-DVLLM_XPU_LIBS_ONLY=ON"
    "-DVLLM_PYTHON_EXECUTABLE=${python3Packages.python}/bin/python"
    "-DVLLM_CUTLASS_SRC_DIR=${cutlass-src}"
    "-DCMAKE_BUILD_TYPE=Release"
    "-DBUILD_SYCL_TLA_KERNELS=ON"
    "-DVLLM_XPU_CUTLASS_TEMPLATE_BACKTRACE_LIMIT=10"
    "-DCMAKE_C_COMPILER_LAUNCHER=sccache"
    "-DCMAKE_CXX_COMPILER_LAUNCHER=sccache"
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
