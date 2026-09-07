{
  lib,
  stdenv,
  fetchFromGitHub,
  cmake,
  ninja,
  pkg-config,
  python3,
  git,
  level-zero,
  libdrm,
}:

let
  zeHeaders = fetchFromGitHub {
    owner = "oneapi-src";
    repo = "level-zero";
    rev = "v1.26.0";
    hash = "sha256-RprUt+unPjp6+deUx67RE1Hxt1DcKHsP5V/YVPXVZbY=";
  };
  computeHeaders = fetchFromGitHub {
    owner = "intel";
    repo = "compute-runtime";
    rev = "aa5ab7a8288f96f1bed187cec0d359bd65354ef3";
    hash = "sha256-SIGkF6/4U6lQZE3qCpJ0apgF7csAEMWPMCbkNI98+Jg=";
  };
in
stdenv.mkDerivation {
  pname = "intel-unitrace";
  version = "0.49.28-c71e831";
  src = fetchFromGitHub {
    owner = "intel";
    repo = "pti-gpu";
    rev = "c71e8316e19bb5316157b9046d877b5eff0e262c";
    hash = "sha256-gjTMV6CR5ryAbPOZieefeveOFSN6KYFkuHd3yQLUbcA=";
  };
  sourceRoot = "source/tools/unitrace";
  nativeBuildInputs = [
    cmake
    ninja
    pkg-config
    python3
    git
  ];
  buildInputs = [
    level-zero
    libdrm
  ];
  cmakeFlags = [
    "-DCMAKE_BUILD_TYPE=Release"
    "-DBUILD_WITH_MPI=OFF"
    "-DBUILD_WITH_ITT=0"
    "-DBUILD_WITH_XPTI=0"
    "-DBUILD_WITH_OMP=0"
    "-DBUILD_WITH_OPENCL=0"
    "-DBUILD_WITH_PERFETTO=OFF"
  ];
  postPatch = ''
    substituteInPlace ../../build_utils/get_compute_runtime_headers.py \
      --replace-fail 'build_utils.clone(url, commit, clone_path)' \
        'clone_path = "${computeHeaders}"'
    substituteInPlace ../../build_utils/get_ze_headers.py \
      --replace-fail 'build_utils.clone(url, commit, clone_path)' \
        'os.makedirs(clone_path, exist_ok=True)' \
      --replace-fail 'src_path = os.path.join(clone_path, "include")' \
        'src_path = os.environ["LEVEL_ZERO_ROOT"]' \
      --replace-fail 'src_path = os.path.join(clone_path, "include", "layers")' \
        'src_path = os.path.join(os.environ["LEVEL_ZERO_ROOT"], "layers")'
  '';
  preConfigure = ''
    export LEVEL_ZERO_ROOT=${zeHeaders}/include
  '';
  # Upstream's install RPATH rewrite conflicts with the Nix compiler wrapper.
  # Keep both the launcher and injection library; normal Nix fixup handles RPATH.
  installPhase = ''
    runHook preInstall
    install -Dm755 unitrace "$out/bin/unitrace"
    install -Dm755 libunitrace_tool.so "$out/lib/libunitrace_tool.so"
    mkdir -p "$out/share/unitrace"
    cp -r ../scripts/metrics/config "$out/share/unitrace/"
    runHook postInstall
  '';
  meta = {
    description = "Intel PTI unified tracing and profiling tool";
    homepage = "https://github.com/intel/pti-gpu/tree/master/tools/unitrace";
    license = lib.licenses.mit;
    platforms = [ "x86_64-linux" ];
  };
}
