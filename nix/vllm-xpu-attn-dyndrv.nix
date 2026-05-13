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
#                       - pch_meta.json: cmake's PCH compile command +
#                         the .pch path embedded into every per-TU command.
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
    # Honoured by upstream's CMakeLists.txt (DEFINED ENV check at
    # line ~186). Empty string -> upstream skips AOT (JIT mode);
    # comma-joined device list -> AOT for those devices.
    export VLLM_XPU_AOT_DEVICES="${aotDevicesStr}"
    export VLLM_XPU_XE2_AOT_DEVICES="${aotDevicesStr}"
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

    patches = [
      ./patches/0001-split-kernel-libs.patch
      ./patches/0005-reduce-kernel-build-memory.patch
      ./patches/0006-decouple-256grf-from-aot.patch
      ./patches/0007-fa2-dtype-split.patch
      ./patches/0008-fa2-pch.patch
    ];

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

    # NOT content-addressed: the eval-time IFD `builtins.readFile
    # "${configureDrv}/.../link_meta.json"` would force a CA realisation
    # at evaluation, requiring `ca-derivations` enabled at the daemon
    # level even for `nix eval`. Keeping configureDrv input-addressed
    # confines the CA experimental-feature requirement to actual builds
    # of the per-TU + link drvs. The downstream per-TU drvs are still
    # CA, so cmake/torch bumps that change configureDrv's input hash
    # don't necessarily invalidate per-TU caches — only the configure
    # step re-runs.


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

  pchMeta = builtins.fromJSON (
    builtins.readFile "${configureDrv}/repo/pch_meta.json"
  );

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
  pchDrv = stdenv.mkDerivation {
    pname = "vllm-xpu-attn-dyndrv-pch";
    version = "0.1.7-dev";

    dontUnpack = true;
    dontStrip = true;

    __contentAddressed = true;

    nativeBuildInputs = [
      which
      python3Packages.python
    ];

    buildInputs = baseBuildInputs;

    buildPhase = ''
      runHook preBuild
      ${envSetup}

      mkdir -p $out

      # The pch_meta.json command embeds the configureDrv's repo path
      # already (extract.py rewrites $src_root -> ${configureDrv}/repo).
      # Only the -o argument has to move from cmake's build-dir layout
      # to $out/cmake_pch.hxx.pch.
      cmd=$(python3 -c '
import json, shlex, sys
with open("${configureDrv}/repo/pch_meta.json") as f:
    pm = json.load(f)
args = shlex.split(pm["command"])
out_args = []
i = 0
while i < len(args):
    if args[i] == "-o" and i + 1 < len(args):
        out_args += ["-o", sys.argv[1]]
        i += 2
    else:
        out_args.append(args[i])
        i += 1
print(shlex.join(out_args))
' "$out/cmake_pch.hxx.pch")

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
  originalPchPath = "${configureDrv}/repo/${pchMeta.pch_out_rel_path}";

  mkTU = tu: stdenv.mkDerivation {
    pname = "vllm-xpu-attn-dyndrv-tu-${tu.safe_name}";
    version = "0.1.7-dev";

    dontUnpack = true;
    dontStrip = true;

    nativeBuildInputs = [
      which
      python3Packages.python
    ];

    buildInputs = baseBuildInputs;

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
      ${envSetup}

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
        NEW_PCH_PATH="${pchDrv}/cmake_pch.hxx.pch" \
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

    buildInputs = baseBuildInputs;

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
      ${envSetup}

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
