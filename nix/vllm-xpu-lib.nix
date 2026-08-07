{
  lib,
  src,
  version,
  cutlass-src,
  onednn-src,
  python3Packages,
  cmake,
  ninja,
  git,
  autoPatchelfHook,
  stdenv,
  intel-oneapi-base,
  intel-pti,
  torch-xpu,
  level-zero,
  intel-compute-runtime,
  intel-graphics-compiler,
  ocl-icd,
  zlib,
  which,
  ccache,
}:

{
  libName,
  featureFlags ? [ ],
  # SYCL AOT target list. See vllm-xpu-kernels.nix: [] (default)
  # is JIT, non-empty list is AOT for those devices.
  aotDevices ? [ ],
  # Wire ccache as the C/C++ compiler launcher. Set false to bypass
  # it for one-off / CI builds on hosts that don't have
  # /var/cache/ccache mounted via extra-sandbox-paths.
  useCcache ? true,
  # Partial-buildout kernel-config selectors (upstream #324). A bare preset
  # name ("chunk_prefill_default") or .conf filename resolves under
  # csrc/xpu/attn/kernel_configs/; null keeps upstream's default, which
  # compiles the full Cartesian kernel sweep. Only the attn lib
  # (FA2_KERNELS_ENABLED=ON) reads these; other libs ignore them.
  kernelChunkPrefillConfig ? null,
  kernelPagedDecodeConfig ? null,
  # Extra kernel-config lines appended to the selected preset's .conf at build
  # time, e.g. add a model's head_size=256 variants on top of
  # "chunk_prefill_default" without forking the preset. Each string is one
  # config line; requires the matching kernel*Config to name a preset.
  kernelChunkPrefillExtra ? [ ],
  kernelPagedDecodeExtra ? [ ],
  # Derivations that must finish before this memory-heavy SYCL build starts.
  # They order the daemon's fan-out and need not be runtime dependencies.
  buildDependencies ? [ ],
  # Optional per-library compile cap. This is intentionally separate from
  # NIX_BUILD_CORES, which still controls SYCL device-link parallelism.
  compileJobs ? null,
}:

let
  syclHome = "${intel-oneapi-base}/compiler/latest";
  aotDevicesStr = lib.concatStringsSep "," aotDevices;

  # Append extra config lines to a named preset's .conf in the unpacked source
  # (build-time, so no import-from-derivation). Strips an optional .conf suffix.
  appendConfLines = cfgName: extra:
    lib.optionalString (extra != [ ] && cfgName != null) (
      let confFile =
        "csrc/xpu/attn/kernel_configs/${lib.removeSuffix ".conf" cfgName}.conf";
      in ''
        printf '\n# --- appended by vllm-xpu-nix kernel*Extra ---\n' >> ${confFile}
        printf '%s\n' ${lib.escapeShellArgs extra} >> ${confFile}
      '');

  # Derivation-level env attrs. `impureEnvVars` / nix.conf
  # `impure-env` gate on `!isSandboxed()` and skip CA / input-addressed
  # builds, so top-level mkDerivation attrs are the only mechanism
  # that crosses the sandbox boundary unconditionally for this build.
  ccacheEnvAttrs = lib.optionalAttrs useCcache {
    CCACHE_DIR = "/var/cache/ccache";
    CCACHE_COMPRESS = "1";
    # random_seed: icpx derives -frandom-seed from the output path,
    #   so without this every drv hash bumps the .o hash and the
    #   cache is useless across builds.
    # time_macros / include_file_{mtime,ctime}: store-path inputs
    #   have varying mtimes between rebuilds; cache on content, not
    #   mtime.
    # pch_defines: PCH-built TUs embed extra defines we don't want
    #   participating in the cache key.
    CCACHE_SLOPPINESS = "random_seed,time_macros,include_file_mtime,include_file_ctime,pch_defines";
    # Skip hashing the CWD (icpx can embed it via __FILE__ / debug
    # records); the per-build /build/<hash>/ root would otherwise
    # poison the cache key. CCACHE_BASEDIR (set in preConfigure)
    # handles the complementary case of absolute /build paths in
    # compiler flags.
    CCACHE_NOHASHDIR = "1";
    # 0770 dir owned root:nixbld -> 007 umask so cached objects stay
    # group-readable for the next builder.
    CCACHE_UMASK = "007";
  };

  ccacheCmakeFlags = lib.optionals useCcache [
    "-DCMAKE_C_COMPILER_LAUNCHER=ccache"
    "-DCMAKE_CXX_COMPILER_LAUNCHER=ccache"
  ];

  # CCACHE_BASEDIR=$NIX_BUILD_TOP is set at builder time (not
  # eval-time), so it lives in preConfigure rather than as a
  # derivation attr.
  ccachePreConfigure = lib.optionalString useCcache ''
    export CCACHE_BASEDIR=$NIX_BUILD_TOP
  '';
in
stdenv.mkDerivation ({
  pname = "vllm-xpu-${lib.replaceStrings [ "_" ] [ "-" ] libName}";
  inherit version;

  inherit src;

  patches = [
    ./patches/0001-split-kernel-libs.patch
    ./patches/0005-reduce-kernel-build-memory.patch
    ./patches/0006-decouple-256grf-from-aot.patch
    # The former 0007-fa2-dtype-split / 0008-fa2-dispatcher-split patches
    # were dropped: upstream #324 ("refactor template gen") rewrote
    # chunk_prefill_configure.cmake to emit one TU per kernel variant, so
    # per-TU peak RSS is bounded by the generator itself. The same refactor
    # adds the kernel_configs partial-buildout system, which the
    # kernelChunkPrefillConfig / kernelPagedDecodeConfig params below drive
    # to cut both build time and total memory by compiling only the variants
    # a deployment dispatches to (see VLLM_{CHUNK_PREFILL,PAGED_DECODE}_CONFIG).
  ];

  postPatch =
    appendConfLines kernelChunkPrefillConfig kernelChunkPrefillExtra
    + appendConfLines kernelPagedDecodeConfig kernelPagedDecodeExtra;

  nativeBuildInputs = [
    cmake
    ninja
    git
    autoPatchelfHook
    which
  ] ++ lib.optional useCcache ccache ++ buildDependencies;

  buildInputs = [
    stdenv.cc.cc.lib
    intel-oneapi-base
    intel-pti
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

  # NIX_BUILD_CORES is inherited from the daemon (defaults to `nproc`).
  # Upstream #324 emits one TU per kernel variant, so per-TU peak RSS is
  # bounded by the generator and a fat-RAM box can run the full nproc
  # fan-out; a narrow kernelChunkPrefillConfig / kernelPagedDecodeConfig
  # shrinks the variant count further. Cap with `nix build --cores N` (or
  # `cores = N` in nix.conf) on memory-constrained hosts; that value also
  # reaches -fsycl-max-parallel-link-jobs via the cmakeFlagsArray append.
  preConfigure = ''
    ${ccachePreConfigure}
    mkdir -p $TMPDIR/bin
    ln -sf ${intel-compute-runtime}/bin/ocloc $TMPDIR/bin/ocloc
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

  # Run after configure has captured the daemon-wide value for device linking,
  # but before cmakeBuildHook constructs `-j$NIX_BUILD_CORES`.
  preBuild = lib.optionalString (compileJobs != null) ''
    export NIX_BUILD_CORES=${toString compileJobs}
  '';

  cmakeFlags = [
    "-GNinja"
    "-DVLLM_XPU_LIBS_ONLY=ON"
    "-DVLLM_PYTHON_EXECUTABLE=${python3Packages.python}/bin/python"
    "-DVLLM_CUTLASS_SRC_DIR=${cutlass-src}"
    # FindoneDNN.cmake FetchContent-clones oneDNN at configure time; redirect
    # it to the prefetched local checkout so the sandbox needs no network.
    "-DFETCHCONTENT_SOURCE_DIR_ONEDNN=${onednn-src}"
    "-DFETCHCONTENT_FULLY_DISCONNECTED=ON"
    "-DCMAKE_BUILD_TYPE=Release"
    "-DBUILD_SYCL_TLA_KERNELS=ON"
    "-DVLLM_XPU_CUTLASS_TEMPLATE_BACKTRACE_LIMIT=10"
  ]
  ++ lib.optional (kernelChunkPrefillConfig != null)
    "-DVLLM_CHUNK_PREFILL_CONFIG=${kernelChunkPrefillConfig}"
  ++ lib.optional (kernelPagedDecodeConfig != null)
    "-DVLLM_PAGED_DECODE_CONFIG=${kernelPagedDecodeConfig}"
  ++ ccacheCmakeFlags
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
} // ccacheEnvAttrs)
