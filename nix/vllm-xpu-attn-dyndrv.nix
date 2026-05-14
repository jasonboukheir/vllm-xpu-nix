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
  # Optional pruning of the FA2 Cartesian TU set. null -> keep all (current
  # behaviour). Attrs with `headDims` / `dtypes` lists drop any TU whose
  # parsed parameters fall outside the filter. Hard-fails at configure time
  # if the filter has empty intersection. Per-TU CA caching is unaffected:
  # filtering changes which TUs are realized, not the bytes of any single TU.
  kernelSet ? null,
  # SYCL AOT target list. See vllm-xpu-kernels.nix: [] (default) is
  # JIT, non-empty list is AOT for those devices.
  aotDevices ? [ ],
}:

# Dynamic-derivations build of attn_kernels_xe_2.
#
# Four stages:
#   1. configureDrv — runs cmake configure on the upstream tree, captures
#                     compile_commands.json + build.ninja, then extracts
#                     (via vllm-xpu-attn-dyndrv-extract.py):
#                       - tu_manifest.json: every .cpp the link consumes,
#                         with absolute src path under $out/repo and the
#                         ninja-relative .o path it expects.
#                       - link_meta.json: cmake's full link command for
#                         attn_kernels_xe_2 with all $vars resolved except
#                         $in (replaced with the sentinel __INPUTS__).
#                       - pch_meta.json + pch_command.txt: the cmake-
#                         emitted PCH compile command (with -o replaced
#                         by __PCH_OUT__) and the .pch path embedded into
#                         every per-TU command. Split so readFile+fromJSON
#                         at eval time stays free of /nix/store mentions.
#   2. pchDrv      — runs the cmake-emitted PCH compile command once,
#                    producing $out/cmake_pch.hxx.pch. Every mkTU build
#                    consumes this artifact via -Xclang -include-pch.
#                    Frontend-parse cost for cute / CUTLASS-SYCL / sycl
#                    is paid here once, not per-TU. CA so torch / nixpkgs
#                    bumps that don't touch the umbrella header don't
#                    invalidate downstream TUs.
#   3. mkTU        — one drv per .cpp TU. Reads its compile command from
#                    configureDrv's compile_commands.json (matched by
#                    absolute src path), strips dep-tracking flags
#                    (-MD/-MT/-MF), overrides -o → $out/tu.o, and rewrites
#                    the -include-pch path to point at pchDrv's .pch.
#                    The .o extension is load-bearing — without it
#                    icpx -fsycl at link time skips the offload pipeline.
#   4. linkDrv     — replays cmake's captured link command verbatim,
#                    with __INPUTS__ replaced (at Nix eval time) by the
#                    per-TU $out/tu.o paths in the same order ninja listed
#                    them. Produces $out/lib/libattn_kernels_xe_2.so with
#                    real torch linkage and full XE2_GPU_LINK_FLAGS.
#
# Previous design (kept here for historical context, ripped out in this
# revision): a profileTU stage measured each TU's preprocessed source
# size, an aggregator collapsed those into tu_profile.json, and a
# `heavyChain` arrangement serialised the top 10% (by byte count) via
# fake buildInput edges to keep concurrent icpx processes under the
# memory budget. After patches 0007-fa2-dtype-split and 0008-fa2-pch
# land in the kernel src, per-TU peak RSS drops from ~40 GB worst-case
# to ~4-5 GB across the whole matrix — the heavy tail no longer exists
# and the profile + chain machinery is dead weight.
#
# TU enumeration goes through IFD on $out/tu_manifest.json rather than
# pure builtins.outputOf so the per-TU drv graph can be built at eval
# time once configureDrv realises.

let
  syclHome = "${intel-oneapi-base}/compiler/latest";
  aotDevicesStr = lib.concatStringsSep "," aotDevices;

  # envSetup is split into compile- and link-variants so per-TU drvs
  # don't carry Nix string-context references to ocloc / IGC /
  # compute-runtime store paths they never use. Those paths bump
  # independently of the SYCL frontend (oneapi point releases, IGC
  # bumps, ocl-icd nixpkgs churn) and would otherwise rewrite every
  # per-TU drv hash on each bump, forcing ~600 CA realisations to
  # re-run even when their .o bytes would be byte-identical.
  #
  # forLink = true  : full env (ocloc on PATH, IGC/L0 on
  #                   LD_LIBRARY_PATH). configureDrv (cmake probes
  #                   for L0/OpenCL/ocloc) and linkDrv (icpx -fsycl
  #                   link calls ocloc for AOT device-image gen)
  #                   both need this.
  # forLink = false : compile-only env (icpx for .cpp -> .o
  #                   fat-object). The SYCL frontend doesn't shell
  #                   out to ocloc or load IGC until link/AOT time.
  mkEnvSetup = { forLink ? false }: ''
    ${lib.optionalString forLink ''
      mkdir -p $TMPDIR/bin
      ln -sf ${intel-compute-runtime}/bin/ocloc-* $TMPDIR/bin/ocloc
      export PATH=$TMPDIR/bin:${syclHome}/bin:$PATH
      export LD_LIBRARY_PATH=${intel-graphics-compiler}/lib:${intel-compute-runtime}/lib:''${LD_LIBRARY_PATH:-}
    ''}
    ${lib.optionalString (!forLink) ''
      export PATH=${syclHome}/bin:$PATH
    ''}
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
    # Honoured by upstream's CMakeLists.txt (DEFINED ENV check at
    # line ~186). Empty string -> upstream skips AOT (JIT mode);
    # comma-joined device list -> AOT for those devices.
    export VLLM_XPU_AOT_DEVICES="${aotDevicesStr}"
    export VLLM_XPU_XE2_AOT_DEVICES="${aotDevicesStr}"
  '';

  # Minimal closure for a .cpp -> .o SYCL fat-object compile: just
  # the icpx frontend (intel-oneapi-base) + stdenv glue + torch
  # headers + python (cmake's compile_commands.json invokes
  # python interpreter-detection probes during configure but the
  # per-TU compile only needs python on PATH for nativeBuildInputs).
  compileInputs = [
    stdenv.cc.cc.lib
    intel-oneapi-base
    torch-xpu
    python3Packages.python
  ];

  # Full closure for cmake configure + final link/AOT: adds the
  # runtime + link-time deps (L0, IGC, compute-runtime/ocloc,
  # OpenCL ICD, profiler, CCL, zlib).
  linkInputs = compileInputs ++ [
    intel-pti
    oneccl-bmg
    level-zero
    intel-compute-runtime
    intel-graphics-compiler
    ocl-icd
    zlib
  ];

  configureDrv = stdenv.mkDerivation {
    pname = "vllm-xpu-attn-dyndrv-configure";
    version = "0.1.7-dev";

    inherit src;

    patches = [
      ./patches/0001-split-kernel-libs.patch
      ./patches/0005-reduce-kernel-build-memory.patch
      ./patches/0006-decouple-256grf-from-aot.patch
      ./patches/0007-fa2-dtype-split.patch
      # 0008-fa2-pch.patch is still disabled — but for a smaller reason
      # than before. The patch itself now forward-declares
      # compat::detail::memcpy_3d_detail / compat_memcpy_3d_detail_usmnone
      # at namespace scope, so cmake's -fpch-instantiate-templates no
      # longer trips the SYCL kernel-name validator
      # ("kernel name should be forward declarable at namespace scope")
      # on cute/util/compat/memory.hpp's ETS tags.
      #
      # The hard blocker is upstream icpx 2025.3: SYCL+PCH bundling
      # invokes clang-offload-bundler with --type=pchi, which the bundler
      # doesn't accept ("'pchi': invalid file type specified" — only
      # i, ii, cui, hipi, d, ll, bc, s, o, gch, ast, a, ao, aoo are
      # registered). Probed with -save-temps the icpx driver says
      # outright "precompiled header generation is not supported with
      # '-fsycl'", so this is deliberate, not an oversight. Manual
      # replay of the two -cc1 -emit-pch passes + re-bundling with
      # --type=ast|gch produces a file the per-TU PCH consumer rejects
      # ("doesn't start with precompiled file magic") — the bundler's
      # bundle header isn't the CPCH magic clang's -include-pch expects,
      # and -Xsycl-target-frontend can't route a separate per-pass
      # -include-pch ("options requiring arguments are unsupported").
      #
      # Re-enable once icpx ships SYCL+PCH support (toolchain >2025.3).
      ./patches/0008-fa2-pch.patch
    ];

    nativeBuildInputs = [
      cmake
      ninja
      git
      which
      python3Packages.python
    ];

    buildInputs = linkInputs;

    dontUseCmakeConfigure = true;
    dontStrip = true;

    # Content-addressed: when irrelevant inputs bump (cmake patch
    # refresh, unrelated nixpkgs churn, a torch revision that doesn't
    # actually change cmake's output) the configureDrv input hash
    # changes but its output bytes (compile_commands.json,
    # link_meta.json, tu_manifest.json, the unpacked repo) are
    # byte-identical, so the CA output path stays the same. Per-TU
    # drvs string-interpolate `${configureDrv}/repo/...` literally
    # into their buildPhase, so a stable configureDrv output path
    # preserves per-TU drv hashes and the realisation cache.
    #
    # Eval-time IFD via `builtins.readFile "${configureDrv}/.../..."`
    # forces a CA realisation at evaluation, which is fine: every
    # consumer of this flake already passes
    # `--extra-experimental-features 'dynamic-derivations
    # ca-derivations recursive-nix'` (the per-TU + link drvs are CA
    # too), so the daemon-level requirement holds anyway.
    __contentAddressed = true;

    buildPhase = ''
      runHook preBuild
      ${mkEnvSetup { forLink = true; }}

      echo "$NIX_BUILD_TOP/source" > .src-root-path

      mkdir -p build
      cmake -S . -B build \
        -GNinja \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
        -DVLLM_XPU_LIBS_ONLY=ON \
        -DVLLM_PYTHON_EXECUTABLE=${python3Packages.python}/bin/python \
        -DVLLM_CUTLASS_SRC_DIR=${cutlass-src} \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SYCL_TLA_KERNELS=ON \
        -DVLLM_XPU_ENABLE_XE_DEFAULT=OFF \
        -DBASIC_KERNELS_ENABLED=OFF \
        -DFA2_KERNELS_ENABLED=ON \
        -DMOE_KERNELS_ENABLED=OFF \
        -DGDN_KERNELS_ENABLED=OFF \
        -DMQA_LOGITS_KERNELS_ENABLED=OFF \
        -DXPU_SPECIFIC_KERNELS_ENABLED=OFF \
        -DXPUMEM_ALLOCATOR_ENABLED=OFF \
        -DVLLM_XPU_SYCL_LINK_PARALLELISM=__VLLM_XPU_LINK_PAR__ \
        -DVLLM_XPU_CUTLASS_TEMPLATE_BACKTRACE_LIMIT=10

      runHook postBuild
    '';

    installPhase = ''
      runHook preInstall

      mkdir -p $out/repo
      cp -a . $out/repo/

      src_root="$(cat .src-root-path)"
      python3 ${./vllm-xpu-attn-dyndrv-extract.py} \
        --repo "$out/repo" \
        --src-root "$src_root" \
        --target attn_kernels_xe_2 \
        --soname libattn_kernels_xe_2.so \
        ${lib.optionalString (kernelSet != null)
          "--kernel-set ${lib.escapeShellArg (builtins.toJSON {
            head_dims = kernelSet.headDims or null;
            dtypes = kernelSet.dtypes or null;
          })}"}

      runHook postInstall
    '';

    meta.description = "vllm-xpu-kernels cmake configure output + per-TU manifest + captured attn_kernels_xe_2 link command";
  };

  manifest = builtins.fromJSON (
    builtins.readFile "${configureDrv}/repo/tu_manifest.json"
  );

  linkMeta = builtins.fromJSON (
    builtins.readFile "${configureDrv}/repo/link_meta.json"
  );

  # PCH metadata is optional. extract.py only writes pch_meta.json /
  # pch_command.txt when cmake produces a `cmake_pch.hxx.cxx` source,
  # which is gated on `target_precompile_headers(...)` being wired into
  # the attn_kernels_xe_2 target. patches/0008-fa2-pch.patch currently
  # stages the umbrella header as a no-op sentinel (icpx 2025.3 blocks
  # SYCL+PCH — see intel/llvm#21491 for the in-flight upstream fix), so
  # the PCH artifacts are absent today. When the patch flips on, the
  # files appear and pchDrv + per-TU PCH-path rewrite kick in
  # automatically.
  pchMeta =
    if builtins.pathExists "${configureDrv}/repo/pch_meta.json"
    then builtins.fromJSON (builtins.readFile "${configureDrv}/repo/pch_meta.json")
    else null;

  # Single drv that runs cmake's PCH compile command once. Output is
  # $out/cmake_pch.hxx.pch — the same .pch every per-TU compile would
  # otherwise pay to re-parse the headers for. The PCH source
  # (cmake_pch.hxx.cxx) and its sibling cmake_pch.hxx (a thin include
  # wrapper cmake generates) both live inside ${configureDrv}/repo/${...},
  # so this drv just realises and reads from configureDrv.
  #
  # Content-addressed: PCH bytes are deterministic given the umbrella
  # header content + compile flags. CUTLASS-SYCL / cute bumps invalidate
  # the PCH (correctly); torch / nixpkgs bumps that don't change the
  # umbrella header's textual closure get to keep their PCH cache entry.
  pchDrv = if pchMeta == null then null else stdenv.mkDerivation {
    pname = "vllm-xpu-attn-dyndrv-pch";
    version = "0.1.7-dev";

    dontUnpack = true;
    dontStrip = true;

    __contentAddressed = true;

    nativeBuildInputs = [
      which
      python3Packages.python
    ];

    buildInputs = compileInputs;

    buildPhase = ''
      runHook preBuild
      ${mkEnvSetup { }}

      mkdir -p $out

      # pch_command.txt is the cmake-emitted PCH compile command with
      # the configureDrv repo path already baked in (extract.py rewrote
      # $src_root -> ${configureDrv}/repo) and the -o argument replaced
      # by the placeholder __PCH_OUT__. Reading via $(cat ...) at build
      # time keeps the icpx / torch / sycl store paths out of eval-time
      # readFile+fromJSON, which would trip Nix's context safety check
      # (same pattern linkDrv uses for link_command.txt + __INPUTS__).
      cmd=$(cat ${configureDrv}/repo/pch_command.txt)
      cmd=''${cmd//__PCH_OUT__/$out/cmake_pch.hxx.pch}

      echo "PCH compile:"
      echo "  cmd: $cmd"
      eval "$cmd"

      runHook postBuild
    '';

    dontInstall = true;

    meta.description = "Shared precompiled header for attn_kernels_xe_2 per-TU compiles";
  };

  # Absolute path of the .pch as embedded in cmake's compile_commands.json
  # (after extract.py's repo-path rewrite). mkTU rewrites this substring
  # to ${pchDrv}/cmake_pch.hxx.pch inside the per-TU compile command.
  # Both are empty strings when PCH is disabled — the per-TU
  # extract-cmd.py sees no matching arg and skips the rewrite cleanly.
  originalPchPath = if pchMeta == null then "" else "${configureDrv}/repo/${pchMeta.pch_out_rel_path}";
  newPchPath = if pchDrv == null then "" else "${pchDrv}/cmake_pch.hxx.pch";

  mkTU = tu: stdenv.mkDerivation {
    pname = "vllm-xpu-attn-dyndrv-tu-${tu.safe_name}";
    version = "0.1.7-dev";

    dontUnpack = true;
    dontStrip = true;

    nativeBuildInputs = [
      which
      python3Packages.python
    ];

    buildInputs = compileInputs;

    # Content-addressed: the .o is just a SYCL device-image bundle for one
    # source file. icpx with -O3 -DNDEBUG (no -g) doesn't embed sandbox
    # paths, so the bytes stay stable across nixpkgs / torch-xpu store-path
    # bumps that don't actually change a header. This is the single
    # highest-leverage CA target — 600+ TUs all share their cache entries
    # whenever input churn doesn't touch what they actually compile.
    __contentAddressed = true;

    # Output is a directory containing tu.o. The .o extension is load-bearing:
    # without it, icpx at link time treats the file as something other than
    # an object and skips the SYCL offload-bundler / device-image pipeline,
    # silently producing a host-only .so with no kernels.
    #
    # ORIGINAL_PCH_PATH is the .pch path cmake embedded in the per-TU
    # command (pointing into ${configureDrv}/repo/build/...). We rewrite
    # it to ${pchDrv}/cmake_pch.hxx.pch so icpx loads the shared,
    # CA-cached PCH from pchDrv instead of failing on the configureDrv
    # path (which never holds the .pch — only pchDrv produces it).
    buildPhase = ''
      runHook preBuild
      ${mkEnvSetup { }}

      cat > "$TMPDIR/extract-cmd.py" <<'PYEOF'
import json, os, shlex, sys
cc_json = os.environ["CC_JSON"]
src_path = os.environ["SRC_PATH"]
out_obj = os.environ["OUT_OBJ"]
old_pch = os.environ["ORIGINAL_PCH_PATH"]
new_pch = os.environ["NEW_PCH_PATH"]
with open(cc_json) as f:
    cc = json.load(f)
match = next((e for e in cc if e["file"] == src_path), None)
if match is None:
    sys.stderr.write(f"src {src_path} not in compile_commands.json\n")
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
        cleaned.append(a.replace(old_pch, new_pch) if a == old_pch else a)
        i += 1
print(shlex.join(cleaned))
PYEOF

      mkdir -p $out
      src_path="${configureDrv}/repo/${tu.src_rel_path}"
      cmd=$(env \
        CC_JSON="${configureDrv}/repo/build/compile_commands.json" \
        SRC_PATH="$src_path" \
        OUT_OBJ="$out/tu.o" \
        ORIGINAL_PCH_PATH="${originalPchPath}" \
        NEW_PCH_PATH="${newPchPath}" \
        python3 "$TMPDIR/extract-cmd.py")

      echo "TU compile (${tu.safe_name}):"
      echo "  src: $src_path"
      echo "  cmd: $cmd"
      eval "$cmd"

      runHook postBuild
    '';

    dontInstall = true;
  };

  # All TUs are independent — patches 0007/0008 collapsed the per-TU
  # peak RSS so the heavy-tail serial chain is no longer needed.
  tuByObjPath = lib.listToAttrs (
    map (tu: { name = tu.obj_rel_path; value = mkTU tu; }) manifest
  );

  inputObjPaths = map
    (objRel: "${tuByObjPath.${objRel}}/tu.o")
    linkMeta.inputs;

  linkDrv = stdenv.mkDerivation {
    pname = "vllm-xpu-attn-kernels-xe-2";
    version = "0.1.7-dev";

    dontUnpack = true;
    dontStrip = true;

    nativeBuildInputs = [
      autoPatchelfHook
      which
    ];

    buildInputs = linkInputs;

    # Content-addressed: the link output is libattn_kernels_xe_2.so. Its
    # bytes embed RUNPATH/NEEDED entries pointing at torch-xpu /
    # intel-oneapi store paths, so a torch-xpu rev bump WILL change the
    # CA hash. CA still helps when input churn doesn't reach the linker
    # line (e.g. only nativeBuildInputs / patchelf hook updated). When
    # the link genuinely changes, the per-TU CA cache is still preserved
    # — only this single drv re-realises.
    __contentAddressed = true;

    # The cmake-emitted link command (with __INPUTS__ as the placeholder
    # for $in, and __VLLM_XPU_LINK_PAR__ as the placeholder for
    # -fsycl-max-parallel-link-jobs) lives at
    # ${configureDrv}/repo/link_command.txt. We read it at build time
    # and substitute:
    #   - __INPUTS__ → per-TU .o store paths in ninja's original order
    #   - __VLLM_XPU_LINK_PAR__ → this drv's NIX_BUILD_CORES, so the
    #     SYCL device-link parallelism scales with whatever --cores
    #     budget the consumer allocated to linkDrv specifically (which
    #     can differ from configureDrv's --cores).
    buildPhase = ''
      runHook preBuild
      ${mkEnvSetup { forLink = true; }}

      cd "$TMPDIR"

      inputs=${lib.escapeShellArg (lib.concatStringsSep " " inputObjPaths)}
      cmd=$(cat ${configureDrv}/repo/link_command.txt)
      cmd=''${cmd//__INPUTS__/$inputs}
      cmd=''${cmd//__VLLM_XPU_LINK_PAR__/$NIX_BUILD_CORES}

      printf '%s\n' "$cmd" > cmd.sh
      echo "=== link command (first 4 KB) ==="
      head -c 4096 cmd.sh || true
      echo
      echo "=== link command size ==="
      wc -c cmd.sh
      echo "=== running with -fsycl-max-parallel-link-jobs=$NIX_BUILD_CORES ==="
      bash cmd.sh

      runHook postBuild
    '';

    installPhase = ''
      runHook preInstall
      mkdir -p $out/lib
      cp "$TMPDIR/${linkMeta.target_path}" $out/lib/libattn_kernels_xe_2.so
      runHook postInstall
    '';

    autoPatchelfIgnoreMissingDeps = [ "libcuda.so.1" ];

    meta = {
      description = "vLLM XPU attn_kernels_xe_2 (dynamic-derivations build, AOT'd for bmg)";
      homepage = "https://github.com/vllm-project/vllm-xpu-kernels";
      license = lib.licenses.asl20;
      platforms = [ "x86_64-linux" ];
    };
  };
in
linkDrv
