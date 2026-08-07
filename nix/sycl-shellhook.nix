# Shared shellHook fragment that wires the oneAPI SYCL toolchain (icx/icpx),
# MKL, Level-Zero loader and the compute-runtime driver into a dev shell, so
# in-tree builds and on-GPU runs (pytest, vllm serve) work from inside the
# shell.
{
  pkgs,
  intel-oneapi,
  cutlass-src,
}: ''
  syclHome="${intel-oneapi}/compiler/latest"
  mkdir -p .dev-bin
  ln -sf ${pkgs.intel-compute-runtime}/bin/ocloc .dev-bin/ocloc
  export PATH="$PWD/.dev-bin:$syclHome/bin:$PATH"
  # Build needs igc + compute-runtime; *running* on the GPU from inside
  # the shell (e.g. pytest tests/) additionally needs the Level-Zero
  # loader and the libze_intel_gpu.so.1 driver, which lives in
  # intel-compute-runtime's separate `drivers` output. Without these the
  # SYCL runtime finds 0 platforms and torch.xpu.is_available() is False.
  export LD_LIBRARY_PATH="${pkgs.level-zero}/lib:${pkgs.intel-graphics-compiler}/lib:${pkgs.intel-compute-runtime}/lib:${pkgs.intel-compute-runtime.drivers}/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  export SYCL_HOME="$syclHome"
  export CMPLR_ROOT="$syclHome"
  export MKLROOT="${intel-oneapi}/mkl/latest"
  export CC="$syclHome/bin/icx"
  export CXX="$syclHome/bin/icpx"
  icpxToolchainFlags="--gcc-toolchain=${pkgs.stdenv.cc.cc} -B${pkgs.stdenv.cc.libc}/lib -L${pkgs.stdenv.cc.libc}/lib -L${pkgs.stdenv.cc.cc.lib}/lib -idirafter ${pkgs.stdenv.cc.libc.dev}/include"
  export CFLAGS="$icpxToolchainFlags ''${CFLAGS:-}"
  export CXXFLAGS="$icpxToolchainFlags ''${CXXFLAGS:-}"
  export LDFLAGS="-L${pkgs.stdenv.cc.libc}/lib -L${pkgs.stdenv.cc.cc.lib}/lib ''${LDFLAGS:-}"
  export LIBRARY_PATH="${pkgs.stdenv.cc.libc}/lib:${pkgs.stdenv.cc.cc.lib}/lib''${LIBRARY_PATH:+:$LIBRARY_PATH}"
  export CPATH="${pkgs.stdenv.cc.libc.dev}/include''${CPATH:+:$CPATH}"
  export CMAKE_PREFIX_PATH="${intel-oneapi}''${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
  export VLLM_CUTLASS_SRC_DIR="${cutlass-src}"
''
