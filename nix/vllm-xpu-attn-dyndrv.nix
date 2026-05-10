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
  # Fraction of TUs (by measured `icpx -E` byte count) to classify as
  # heavy and serialise via the heavy chain. Default 10 ⇒ top 10% by
  # preprocessed-source size are chained one-at-a-time; the rest run
  # at full max-jobs parallelism. Picking by percentile (rather than
  # absolute byte threshold) means upstream additions of cheap TUs
  # don't accidentally promote previous mediums to heavy.
  heavyPercentile ? 10,
}:

# Dynamic-derivations build of attn_kernels_xe_2.
#
# Five stages:
#   1. configureDrv      — runs cmake configure on the upstream tree, captures
#                          compile_commands.json + build.ninja, then extracts
#                          (via vllm-xpu-attn-dyndrv-extract.py):
#                            - tu_manifest.json: every .cpp the link consumes,
#                              with absolute src path under $out/repo and the
#                              ninja-relative .o path it expects.
#                            - link_meta.json: cmake's full link command for
#                              attn_kernels_xe_2 with all $vars resolved except
#                              $in (replaced with the sentinel __INPUTS__).
#   2. mkProfileTU       — per-TU CA drv that runs the TU's compile command
#                          with `-E` (preprocessor only) and writes the
#                          resulting byte count to $out/preproc.bytes.
#                          Bounded by header expansion (~250–500 MB peak),
#                          no template instantiation or codegen — safe at
#                          full max-jobs parallelism.
#   3. profileAggregator — single CA drv that ingests every profileTU's
#                          preproc.bytes into $out/tu_profile.json
#                          ({ src_rel_path = bytes; }). One IFD on this
#                          batches all 600 profile drvs into one
#                          eval-time realisation pass, instead of
#                          serialising IFDs per TU.
#   4. mkTU              — one drv per .cpp TU. Reads its compile command from
#                          configureDrv's compile_commands.json (matched by
#                          absolute src path), strips dep-tracking flags
#                          (-MD/-MT/-MF) and overrides -o → $out/tu.o. The .o
#                          extension is load-bearing — without it icpx -fsycl
#                          at link time skips the offload pipeline. Top
#                          heavyPercentile% of TUs (by profile bytes) are
#                          chained via fake buildInputs to serialise the
#                          icpx RSS-spiking heavy tail without throttling
#                          the light TUs.
#   5. linkDrv           — replays cmake's captured link command verbatim,
#                          with __INPUTS__ replaced (at Nix eval time) by the
#                          per-TU $out/tu.o paths in the same order ninja
#                          listed them. Produces $out/lib/libattn_kernels_xe_2.so
#                          with real torch linkage and full XE2_GPU_LINK_FLAGS
#                          (AOT'd for bmg).
#
# Step 2 of the dyn-drv plan: scaled from the 3-TU POC (which validated
# that SYCL device-image registration survives per-.o linking) to the
# full ~600-TU kernel set. TU enumeration goes through IFD on
# $out/tu_manifest.json rather than pure builtins.outputOf so the per-TU
# drv graph can be built at eval time once configureDrv realises. This
# keeps the link drv free of build-time recursive-nix calls; if we later
# need to push enumeration into the build phase, configureDrv would also
# need __contentAddressed = true for builtins.outputOf to resolve cleanly.

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
    # Match the consumer (vllm-xpu-kernels.nix) — only AOT for bmg, not the
    # default pvc,bmg,bmg-g21-a0,bmg-g31-a0 list. ocloc runs once per device
    # variant at link time, so this cuts the SYCL link ~4x.
    export VLLM_XPU_XE2_AOT_DEVICES=bmg
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
        -DXPUMEM_ALLOCATOR_ENABLED=OFF

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

  # Per-TU preprocessor-only drv. Mirrors mkTU's compile-command extraction
  # but rewrites the command to `-E` (preprocess to stdout) and counts the
  # bytes. The byte count is the proxy for compile cost used downstream to
  # classify heavy TUs — preprocessed-source size correlates with the
  # depth of header / template-policy expansion that drives icpx RSS
  # during the actual `-c` compile.
  #
  # CA so that single-TU upstream changes invalidate only the affected
  # profile, not all 600. Output is one tiny file ($out/preproc.bytes
  # holding a single integer), which keeps store-write cost negligible.
  mkProfileTU = tu: stdenv.mkDerivation {
    pname = "vllm-xpu-attn-dyndrv-profile-${tu.safe_name}";
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

      cat > "$TMPDIR/extract-cmd.py" <<'PYEOF'
import json, os, shlex, sys
cc_json = os.environ["CC_JSON"]
src_path = os.environ["SRC_PATH"]
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
        i += 2
    elif a == "-c":
        i += 1
    else:
        cleaned.append(a)
        i += 1
cleaned.append("-E")
print(shlex.join(cleaned))
PYEOF

      mkdir -p $out
      src_path="${configureDrv}/repo/${tu.src_rel_path}"
      cmd=$(env \
        CC_JSON="${configureDrv}/repo/build/compile_commands.json" \
        SRC_PATH="$src_path" \
        python3 "$TMPDIR/extract-cmd.py")

      echo "TU profile (${tu.safe_name}):"
      echo "  src: $src_path"
      echo "  cmd: $cmd"
      out_size=$(eval "$cmd" | wc -c)
      printf '%s' "$out_size" > $out/preproc.bytes
      echo "  preproc bytes: $out_size"

      runHook postBuild
    '';

    dontInstall = true;
  };

  profileTUs = lib.listToAttrs (
    map (tu: { name = tu.obj_rel_path; value = mkProfileTU tu; }) manifest
  );

  # Aggregates per-TU preproc byte counts into one JSON map. Separate
  # stage (rather than IFD per profileTU) so eval-time realisation
  # batches all 600 profile drvs in a single Nix scheduler pass instead
  # of serialising one realisation per IFD. The JSON values are plain
  # integers, no store paths, so the downstream `builtins.fromJSON` does
  # not run into the context-less-store-path footgun documented in
  # vllm-xpu-attn-dyndrv-extract.py.
  profileAggregator = stdenv.mkDerivation {
    pname = "vllm-xpu-attn-dyndrv-profile-aggregator";
    version = "0.1.7-dev";

    dontUnpack = true;
    dontStrip = true;

    __contentAddressed = true;

    nativeBuildInputs = [ python3Packages.python ];

    buildInputs = lib.attrValues profileTUs;

    # The manifest holds ~600 entries; once each profileTU's CA placeholder
    # is replaced with its resolved store path, an inline `${profileEntries}`
    # in `buildPhase` clears Linux's MAX_ARG_STRLEN (32 pages = 128 KiB) per
    # envp string. Hand the entries off via a sidecar file so `buildPhase`
    # stays small.
    profileEntries = lib.concatMapStringsSep "\n" (tu:
      "${tu.src_rel_path}\t${profileTUs.${tu.obj_rel_path}}/preproc.bytes"
    ) manifest;
    passAsFile = [ "profileEntries" ];

    buildPhase = ''
      runHook preBuild

      mkdir -p $out
      python3 - > $out/tu_profile.json <<'PROFEOF'
import json, os
profiles = {}
with open(os.environ["profileEntriesPath"]) as f:
    for line in f:
        line = line.rstrip("\n")
        if not line:
            continue
        src, path = line.split("\t", 1)
        profiles[src] = int(open(path).read())
print(json.dumps(profiles, indent=2, sort_keys=True))
PROFEOF

      runHook postBuild
    '';

    dontInstall = true;
  };

  profile = builtins.fromJSON (
    builtins.readFile "${profileAggregator}/tu_profile.json"
  );

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
    buildPhase = ''
      runHook preBuild
      ${envSetup}

      cat > "$TMPDIR/extract-cmd.py" <<'PYEOF'
import json, os, shlex, sys
cc_json = os.environ["CC_JSON"]
src_path = os.environ["SRC_PATH"]
out_obj = os.environ["OUT_OBJ"]
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
        cleaned.append(a)
        i += 1
print(shlex.join(cleaned))
PYEOF

      mkdir -p $out
      src_path="${configureDrv}/repo/${tu.src_rel_path}"
      cmd=$(env \
        CC_JSON="${configureDrv}/repo/build/compile_commands.json" \
        SRC_PATH="$src_path" \
        OUT_OBJ="$out/tu.o" \
        python3 "$TMPDIR/extract-cmd.py")

      echo "TU compile (${tu.safe_name}):"
      echo "  src: $src_path"
      echo "  cmd: $cmd"
      eval "$cmd"

      runHook postBuild
    '';

    dontInstall = true;
  };

  # SYCL-TLA's per-TU peak RSS is bimodal: most TUs land near a ~5 GiB
  # median in icpx, a tail spikes to ~40 GiB. With pure independent drvs
  # the consumer's nix.settings.max-jobs has to be sized for the heavy
  # tail, leaving cores idle on the median. We chain heavy TUs into a
  # serial DAG via a fake buildInput edge — Nix's scheduler then runs
  # them one at a time regardless of max-jobs, while light TUs stay
  # fully parallel. Per-TU CA caching is preserved (the chain edge
  # changes the drv graph, not the build inputs icpx sees).
  #
  # Heavy/light split is driven by measured profileAggregator output:
  # top heavyPercentile% by preprocessor byte count are heavy. Sort
  # descending by bytes, take the prefix, then re-sort the heavy set
  # alphabetically by safe_name so chain-successor identity stays
  # stable across upstream reorderings of cmake's build input list
  # (the boundary case becomes "a new upstream TU" rather than "cmake
  # reshuffled inputs"). At least one TU is always classified heavy
  # so the chain is well-defined for tiny pruned kernel sets.
  heavyCount = lib.max 1 ((lib.length manifest * heavyPercentile) / 100);
  manifestSortedBySize = lib.sort
    (a: b: (profile.${a.src_rel_path} or 0) > (profile.${b.src_rel_path} or 0))
    manifest;
  heavySrcSet = lib.genAttrs
    (map (tu: tu.src_rel_path) (lib.take heavyCount manifestSortedBySize))
    (_: true);
  heavyTUs = lib.sort (a: b: a.safe_name < b.safe_name)
    (lib.filter (tu: heavySrcSet.${tu.src_rel_path} or false) manifest);
  lightTUs = lib.filter (tu: !(heavySrcSet.${tu.src_rel_path} or false)) manifest;

  heavyChain = lib.foldl' (acc: tu:
    let
      prev = if acc == [] then null else (lib.last acc).drv;
      drv = (mkTU tu).overrideAttrs (old: {
        buildInputs = (old.buildInputs or [])
          ++ lib.optional (prev != null) prev;
      });
    in acc ++ [ { inherit tu drv; } ]
  ) [] heavyTUs;

  heavyDrvByObj = lib.listToAttrs (
    map (e: { name = e.tu.obj_rel_path; value = e.drv; }) heavyChain
  );

  lightDrvByObj = lib.listToAttrs (
    map (tu: { name = tu.obj_rel_path; value = mkTU tu; }) lightTUs
  );

  # Look up TUs by their ninja .o path. The link command's input list is
  # the canonical order; we reassemble per-TU drvs in that order so the
  # final link command keeps ninja's link-time ordering.
  tuByObjPath = lightDrvByObj // heavyDrvByObj;

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
    # for $in) lives at ${configureDrv}/repo/link_command.txt. We read it
    # at build time and substitute the per-TU .o store paths in ninja's
    # original input order. inputObjPaths is interpolated directly so the
    # /nix/store/.../tu.o references become proper store deps of linkDrv.
    buildPhase = ''
      runHook preBuild
      ${envSetup}

      cd "$TMPDIR"

      inputs=${lib.escapeShellArg (lib.concatStringsSep " " inputObjPaths)}
      cmd=$(cat ${configureDrv}/repo/link_command.txt)
      cmd=''${cmd//__INPUTS__/$inputs}

      printf '%s\n' "$cmd" > cmd.sh
      echo "=== link command (first 4 KB) ==="
      head -c 4096 cmd.sh || true
      echo
      echo "=== link command size ==="
      wc -c cmd.sh
      echo "=== running ==="
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
