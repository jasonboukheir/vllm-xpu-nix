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

# Dynamic-derivations POC for attn_kernels_xe_2.
#
# Three stages:
#   1. configureDrv  — runs cmake configure (-DCMAKE_EXPORT_COMPILE_COMMANDS=ON)
#                      on the upstream tree so configure_file expansions
#                      materialise under build/csrc/xpu/attn/xe_2/. Captures
#                      the whole tree at $out/repo/ and rewrites
#                      compile_commands.json paths from the build sandbox to
#                      $out/repo so per-TU drvs can replay commands directly.
#   2. mkTU          — one drv per .cpp TU. Reads its compile command from
#                      configureDrv's compile_commands.json, strips dep-tracking
#                      flags (-MD/-MT/-MF) and overrides -o → $out. The .o
#                      ends up at $out (single-file output).
#   3. linkDrv       — icpx -fsycl -shared with the upstream
#                      SYCL_DEVICE_LINK_FLAGS, AOT for bmg. Produces
#                      $out/lib/libattn_kernels_xe_2.so.
#
# Scope: 2-3 hand-picked generated TUs only. The launcher TUs (fmha_xe2.cpp,
# paged_decode_xe2.cpp) reference every policy specialisation by name and
# would fail to link with this subset, so they are deliberately excluded.
# The resulting .so is NOT functionally complete; it exists to validate that
# (a) per-TU compile-then-link via separate Nix derivations works on
# Determinate Nix 3.19.0 with the experimental flags turned on, and
# (b) SYCL device images survive per-.o linking — i.e. the resulting .so
# still contains _GLOBAL__sub_I_* static initializers and __CLANG_OFFLOAD_BUNDLE__
# / sycl_offloading sections (verifiable via scripts/verify-attn-shards.sh).

let
  syclHome = "${intel-oneapi-base}/compiler/latest";

  envSetup = ''
    mkdir -p $TMPDIR/bin
    ln -sf ${intel-compute-runtime}/bin/ocloc-* $TMPDIR/bin/ocloc
    export PATH=$TMPDIR/bin:${syclHome}/bin:$PATH
    export LD_LIBRARY_PATH=${intel-graphics-compiler}/lib:${intel-compute-runtime}/lib:''${LD_LIBRARY_PATH:-}
    export SYCL_HOME=${syclHome}
    export CMPLR_ROOT=${syclHome}
    export MKLROOT=${intel-oneapi-base}/mkl/latest
    export CC=${syclHome}/bin/icx
    export CXX=${syclHome}/bin/icpx
    icpxToolchainFlags="--gcc-toolchain=${stdenv.cc.cc} -B${stdenv.cc.libc}/lib -L${stdenv.cc.libc}/lib -L${stdenv.cc.cc.lib}/lib -idirafter ${stdenv.cc.libc.dev}/include"
    export CFLAGS="$icpxToolchainFlags ''${CFLAGS:-}"
    export CXXFLAGS="$icpxToolchainFlags ''${CXXFLAGS:-}"
    export LDFLAGS="-L${stdenv.cc.libc}/lib -L${stdenv.cc.cc.lib}/lib ''${LDFLAGS:-}"
    export LIBRARY_PATH=${stdenv.cc.libc}/lib:${stdenv.cc.cc.lib}/lib''${LIBRARY_PATH:+:$LIBRARY_PATH}
    export CPATH=${stdenv.cc.libc.dev}/include''${CPATH:+:$CPATH}
    export CMAKE_PREFIX_PATH=${intel-oneapi-base}''${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}
  '';

  baseBuildInputs = [
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

  configureDrv = stdenv.mkDerivation {
    pname = "vllm-xpu-attn-dyndrv-configure";
    version = "0.1.7-dev";

    inherit src;

    patches = [ ./patches/0001-split-kernel-libs.patch ];

    nativeBuildInputs = [
      cmake
      ninja
      git
      which
      python3Packages.python
    ];

    buildInputs = baseBuildInputs;

    dontUseCmakeConfigure = true;
    dontStrip = true;

    buildPhase = ''
      runHook preBuild
      ${envSetup}

      echo "$NIX_BUILD_TOP/source" > .src-root-path

      mkdir -p build
      cmake -S . -B build \
        -GNinja \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DVLLM_XPU_LIBS_ONLY=ON \
        -DVLLM_PYTHON_EXECUTABLE=${python3Packages.python}/bin/python \
        -DVLLM_CUTLASS_SRC_DIR=${cutlass-src} \
        -DCMAKE_BUILD_TYPE=Release \
        -DXE2_AOT_DEVICES=bmg \
        -DBUILD_SYCL_TLA_KERNELS=ON \
        -DVLLM_XPU_ENABLE_XE_DEFAULT=OFF \
        -DBASIC_KERNELS_ENABLED=OFF \
        -DFA2_KERNELS_ENABLED=ON \
        -DMOE_KERNELS_ENABLED=OFF \
        -DGDN_KERNELS_ENABLED=OFF \
        -DMQA_LOGITS_KERNELS_ENABLED=OFF \
        -DXPU_SPECIFIC_KERNELS_ENABLED=OFF \
        -DXPUMEM_ALLOCATOR_ENABLED=OFF

      runHook postBuild
    '';

    installPhase = ''
      runHook preInstall

      mkdir -p $out/repo
      cp -a . $out/repo/

      src_root="$(cat .src-root-path)"
      python3 - <<PYEOF
import json
cc_path = "$out/repo/build/compile_commands.json"
src_root = "$src_root"
new_root = "$out/repo"
with open(cc_path) as f:
    cc = json.load(f)
def sub(s):
    return s.replace(src_root, new_root)
for entry in cc:
    for k in ("directory", "file"):
        if k in entry:
            entry[k] = sub(entry[k])
    if "command" in entry:
        entry["command"] = sub(entry["command"])
    if "arguments" in entry:
        entry["arguments"] = [sub(a) for a in entry["arguments"]]
with open(cc_path, "w") as f:
    json.dump(cc, f, indent=2)
print(f"rewrote {len(cc)} compile_commands.json entries: {src_root} -> {new_root}")
PYEOF

      runHook postInstall
    '';
  };

  pocTUNames = [
    "chunk_prefill_kernel_template_chunk_policy_head64_fffff.cpp"
    "paged_decode_kernel_template_q8_h64_p16_fff.cpp"
    "paged_decode_kernel_template_q16_h128_p32_fff.cpp"
  ];

  mkTU = relName: stdenv.mkDerivation {
    pname = "vllm-xpu-attn-dyndrv-tu-${lib.removeSuffix ".cpp" relName}";
    version = "0.1.7-dev";

    dontUnpack = true;
    dontStrip = true;

    nativeBuildInputs = [
      which
      python3Packages.python
    ];

    buildInputs = baseBuildInputs;

    # Output is a directory containing tu.o; the .o extension is load-bearing.
    # Without it, icpx at link time treats the file as something other than
    # an object and skips the SYCL offload-bundler / device-image pipeline,
    # silently producing a host-only .so with no kernels.
    buildPhase = ''
      runHook preBuild
      ${envSetup}

      cat > "$TMPDIR/extract-cmd.py" <<'PYEOF'
import json, os, shlex, sys
cc_json = os.environ["CC_JSON"]
tu_name = os.environ["TU_NAME"]
out_obj = os.environ["OUT_OBJ"]
with open(cc_json) as f:
    cc = json.load(f)
match = next((e for e in cc if e["file"].endswith("/" + tu_name)), None)
if match is None:
    sys.stderr.write(f"TU {tu_name} not in compile_commands.json\n")
    sys.exit(1)
if "arguments" in match:
    args = list(match["arguments"])
else:
    args = shlex.split(match["command"])
cleaned, i = [], 0
while i < len(args):
    a = args[i]
    if a in ("-MD", "-MMD"):
        i += 1
    elif a in ("-MT", "-MF"):
        i += 2
    elif a == "-o":
        cleaned += ["-o", out_obj]
        i += 2
    else:
        cleaned.append(a)
        i += 1
print(shlex.join(cleaned))
PYEOF

      mkdir -p $out
      cmd=$(env \
        CC_JSON="${configureDrv}/repo/build/compile_commands.json" \
        TU_NAME="${relName}" \
        OUT_OBJ="$out/tu.o" \
        python3 "$TMPDIR/extract-cmd.py")

      echo "POC TU compile:"
      echo "  $cmd"
      eval "$cmd"

      runHook postBuild
    '';

    dontInstall = true;
  };

  pocObjs = map mkTU pocTUNames;
  pocObjPaths = map (o: "${o}/tu.o") pocObjs;

  linkDrv = stdenv.mkDerivation {
    pname = "vllm-xpu-attn-kernels-xe-2-dyndrv-poc";
    version = "0.1.7-dev";

    dontUnpack = true;
    dontStrip = true;

    nativeBuildInputs = [
      autoPatchelfHook
      which
    ];

    buildInputs = baseBuildInputs;

    buildPhase = ''
      runHook preBuild
      ${envSetup}

      # POC: skip torch link to keep the experiment focused on the SYCL
      # link pipeline. The .o files reference torch types but we let those
      # become undefined dynamic symbols (--unresolved-symbols=ignore-all).
      # The resulting .so won't be dlopen'able but it lets us verify that
      # OFFLOAD_DEVICE_CODE / .tgtimg / .tgtsym sections survive the link.
      $CXX -fsycl -shared -fPIC \
        --gcc-toolchain=${stdenv.cc.cc} \
        -B${stdenv.cc.libc}/lib \
        -L${stdenv.cc.libc}/lib \
        -L${stdenv.cc.cc.lib}/lib \
        -fsycl-max-parallel-link-jobs=''${NIX_BUILD_CORES:-1} \
        -flink-huge-device-code \
        -Xspirv-translator -spirv-ext=+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate \
        -fsycl-targets=spir64_gen \
        -Xsycl-target-backend=spir64_gen '-device bmg -internal_options -cl-intel-256-GRF-per-thread' \
        ${lib.concatStringsSep " " pocObjPaths} \
        -Wl,--unresolved-symbols=ignore-all \
        -o libattn_kernels_xe_2.so

      runHook postBuild
    '';

    installPhase = ''
      runHook preInstall
      mkdir -p $out/lib
      cp libattn_kernels_xe_2.so $out/lib/
      runHook postInstall
    '';

    autoPatchelfIgnoreMissingDeps = [ "libcuda.so.1" ];

    meta = {
      description = "POC: dynamic-derivations build of attn_kernels_xe_2 (2-3 TUs only, NOT functionally complete)";
      homepage = "https://github.com/vllm-project/vllm-xpu-kernels";
      license = lib.licenses.asl20;
      platforms = [ "x86_64-linux" ];
    };
  };
in
linkDrv
