# Single-card Xe2 experiment on the released Python, oneAPI, and kernel stack.
{
  pkgs,
  current,
  vllm-xpu-unstable,
  sycl-tla-unstable-src,
  vllm-xpu-kernels-unstable-src,
}:
let
  inherit (pkgs) lib stdenv python312;
  pluginRevision = "677db18d0129703b529270fe75052272f0e2834a";
  pluginSrc = pkgs.fetchFromGitHub {
    owner = "mgaruccio";
    repo = "vllm-inkling2-xpu";
    rev = pluginRevision;
    hash = "sha256-uvVFXTdI2CIFVaZuYW5+Nf/D/xeGtkpCJg0b0Tt8LAw=";
  };

  # Brutus's released BMG specialization. Reuse its compiled kernel closure.
  base = vllm-xpu-unstable.override {
    withTorchvision = true;
    aotDevices = [ "bmg" ];
    kernelConfig = {
      chunkPrefill = "chunk_prefill_default";
      chunkPrefillExtra = [
        "256,true,true,false,false,false"
        "256,false,true,false,false,false"
        "256,false,true,false,false,true"
        "64,false,false,false,false,false"
      ];
      pagedDecode = "paged_decode_default";
      pagedDecodeExtra = [
        "8,256,16,true,false,false"
        "8,256,32,true,false,false"
        "8,256,64,true,false,false"
        "8,256,64,false,false,false"
      ];
    };
  };
  # Inkling Mini landed after 0.34. Keep its two new Python dependencies
  # private to this core; the host's released vLLM closure remains usable.
  py = current.python312PackagesXpu;
  experimentalPython = py // {
    compressed-tensors = py.buildPythonPackage {
      pname = "compressed-tensors";
      version = "0.20.1";
      format = "wheel";
      src = pkgs.fetchurl {
        url = "https://files.pythonhosted.org/packages/f7/d3/25ba771806ff12be9a518d644ec7de9c2858c5fa0667b928f95553929390/compressed_tensors-0.20.1-py3-none-any.whl";
        hash = "sha256-FTB8kEkEIZyvy1fVXxaxvWuvEI0EIbZd+nnuzgFnI4I=";
      };
      dependencies = [
        current.torch-xpu
        py.transformers
        py.pydantic
        py.loguru
        py.psutil
      ];
      pythonImportsCheck = [ "compressed_tensors" ];
    };
    xgrammar = py.buildPythonPackage {
      pname = "xgrammar";
      version = "0.2.9";
      format = "wheel";
      src = pkgs.fetchurl {
        url = "https://files.pythonhosted.org/packages/f8/c4/92e38d6a1282331e22ca55135a5c8ecd177c877d46be1a12cbb792817ad0/xgrammar-0.2.9-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl";
        hash = "sha256-MybJ+1hkki/ee1Lv6px+g7kaj1v4HbtfUWTSNBDf2uQ=";
      };
      nativeBuildInputs = [ pkgs.autoPatchelfHook ];
      buildInputs = [ stdenv.cc.cc.lib ];
      dependencies = [
        py.apache-tvm-ffi
        py.pydantic
        current.torch-xpu
        py.transformers
        current.triton-xpu
        py.numpy
        py.typing-extensions
      ];
      preFixup = ''
        addAutoPatchelfSearchPath ${py.apache-tvm-ffi}/${python312.sitePackages}/tvm_ffi/lib
      '';
      pythonImportsCheck = [ "xgrammar" ];
    };
  };
  coreRevision = "0085facc58dedeef757aca3f51d6508e2ef9b148";
  mkCore =
    suffix:
    (import ./mk-vllm.nix {
      inherit pkgs;
      inherit (current)
        intel-oneapi
        intel-pti
        torch-xpu
        triton-xpu
        flash-linear-attention
        ;
      python3Packages = experimentalPython;
    })
      {
        src = pkgs.fetchFromGitHub {
          owner = "vllm-project";
          repo = "vllm";
          rev = coreRevision;
          hash = "sha256-nO1prQYtSQnqPIJvz6kzu1q5hCGGwPTXYZXNYjdZ3L8=";
        };
        version = "0.34.0.post1+g${builtins.substring 0 10 coreRevision}.inkling${suffix}";
        kernels = base.kernelPackage;
        withTorchvision = true;
        mcpPackage = py.mcp-v2;
      };

  core = mkCore "";
  # The long-context recipe uses both commits of vllm-project/vllm#69444.
  # The second commit charges the reserved discard block in admission checks.
  mkSparseCore =
    suffix:
    (mkCore suffix).overrideAttrs (old: {
      patches = (old.patches or [ ]) ++ [
        (pkgs.fetchurl {
          url = "https://github.com/vllm-project/vllm/commit/0811005c59b750e63f9bd6658842fa7d529b8e21.patch";
          hash = "sha256-CrJ3ws3gZOKZvCjsYLv0Kv3mrBLnc/4fHvdeY2JTcd8=";
        })
        (pkgs.fetchurl {
          url = "https://github.com/vllm-project/vllm/commit/34a0975f5c9dcee1c88742aacb38de841c3eb7e2.patch";
          hash = "sha256-ARAXzBQPUVVS4jShi0GZDakgJ68k2QNLW/zKEA/qzS4=";
        })
      ];
    });

  sparseCore = mkSparseCore ".sparse";
  kvarnCore = (mkSparseCore ".sparse.kvarn").overrideAttrs (old: {
    patches = old.patches ++ [
      ../experiments/inkling-b70/kvarn-plugin/sycl-extensible-cache.patch
      ../experiments/inkling-b70/kvarn-plugin/xpu-vmm-memory-baseline.patch
      ../experiments/inkling-b70/kvarn-plugin/request-slot-generation.patch
      ../experiments/inkling-b70/kvarn-plugin/active-only-offload.patch
    ];
  });

  syclHome = "${current.intel-oneapi}/compiler/latest";
  # Upstream invokes icpx directly. Supply the same GCC and glibc paths as the
  # released Xe2 kernels, and bound the offline compiler's link parallelism.
  icpx = pkgs.writeShellScriptBin "icpx" ''
    exec ${syclHome}/bin/icpx --gcc-toolchain=${stdenv.cc.cc} \
      -B${stdenv.cc.libc}/lib -L${stdenv.cc.libc}/lib -L${stdenv.cc.cc.lib}/lib \
      -idirafter ${stdenv.cc.libc.dev}/include -fsycl-max-parallel-link-jobs="$NIX_BUILD_CORES" "$@"
  '';
  plugin =
    withCutlass:
    current.python312PackagesXpu.buildPythonPackage {
      pname = "vllm-inkling2-xpu";
      version = "0.1.0";
      src = pluginSrc;
      pyproject = true;
      build-system = [ current.python312PackagesXpu.setuptools ];
      nativeBuildInputs = lib.optionals withCutlass [ pkgs.intel-compute-runtime ];
      buildInputs = lib.optionals withCutlass [ current.torch-xpu ];
      postInstall = lib.optionalString withCutlass ''
        export PATH=${icpx}/bin:${syclHome}/bin:$PATH
        export PYTHON=${python312.interpreter}
        export PYTHONPATH="$out/${python312.sitePackages}:$PYTHONPATH"
        export SYCL_TLA_DIR=${sycl-tla-unstable-src}
        export VLLM_XPU_KERNELS_DIR=${vllm-xpu-kernels-unstable-src}
        export LD_LIBRARY_PATH=${pkgs.intel-graphics-compiler}/lib:${pkgs.intel-compute-runtime}/lib:''${LD_LIBRARY_PATH:-}
        export LIBRARY_PATH=${stdenv.cc.libc}/lib:${stdenv.cc.cc.lib}/lib:''${LIBRARY_PATH:-}
        bash scripts/build-cutlass-xe2.sh
      '';
      passthru.revision = pluginRevision;
    };

  # The KVarN additions are Python/Triton. Reuse the matching native extension
  # from the exact plugin revision; Python-source adjustments do not rebuild it.
  kvarnPlugin = (plugin false).overridePythonAttrs (old: {
    patches = (old.patches or [ ]) ++ [
      ../experiments/inkling-b70/kvarn-plugin/inkling-kvarn-encoder.patch
      ../experiments/inkling-b70/kvarn-plugin/inkling-kvarn-k4v4.patch
      ../experiments/inkling-b70/kvarn-plugin/inkling-kvarn-flash-prefill.patch
      ../experiments/inkling-b70/kvarn-plugin/inkling-kvarn-flash-hook.patch
    ];
    postInstall = (old.postInstall or "") + ''
      ln -s ${(plugin true)}/${python312.sitePackages}/vllm_inkling2_xpu/ops/_inkling_nvfp4_xe2*.so \
        "$out/${python312.sitePackages}/vllm_inkling2_xpu/ops/"
    '';
  });

  # The serving process, engine workers, and ModelRegistry inspection children
  # must discover the plugin using the same installed Python distribution.
  experiment =
    {
      name,
      withCutlass ? false,
      sparse ? false,
      kvarn ? false,
    }:
    let
      selectedCore =
        if kvarn then
          kvarnCore
        else if sparse then
          sparseCore
        else
          core;
      installedPlugin = if kvarn then kvarnPlugin else plugin withCutlass;
      python = python312.withPackages (_: [
        selectedCore
        installedPlugin
      ]);
      extra =
        lib.optionalString withCutlass "--set-default INK2_CUTLASS_MOE 1"
        + lib.optionalString sparse " --set-default INK2_SPARSE_CONV_HISTORY 1 --set-default INK2_HYBRID_PREFIX_SDPA 1"
        # Block-outermost layout packs the native records at their physical
        # size instead of padding each global-attention page to FP8 width.
        + lib.optionalString kvarn " --set-default INK2_KVARN_K4V4 1 --set-default VLLM_KV_CACHE_LAYOUT BLNHC";
      runtimeArgs = builtins.filter (
        arg: !(lib.hasPrefix "--prefix PYTHONPATH " arg)
      ) selectedCore.makeWrapperArgs;
    in
    pkgs.runCommand "vllm-xpu-inkling2${name}"
      {
        nativeBuildInputs = [ pkgs.makeWrapper ];
        meta.mainProgram = "vllm";
        passthru = {
          core = selectedCore;
          plugin = installedPlugin;
          kernelPackage = base.kernelPackage;
        };
      }
      ''
        mkdir -p "$out/bin"
        makeWrapper ${selectedCore}/bin/vllm "$out/bin/vllm" \
          --prefix PYTHONPATH : ${installedPlugin}/${python312.sitePackages} ${extra}
        makeWrapper ${python}/bin/python "$out/bin/python" \
          --prefix PYTHONPATH : ${python}/${python312.sitePackages} ${extra} \
          ${lib.concatStringsSep " " runtimeArgs}
      '';
in
{
  vllm-xpu-inkling2 = experiment { name = ""; };
  vllm-xpu-inkling2-cutlass = experiment {
    name = "-cutlass";
    withCutlass = true;
  };
  vllm-xpu-inkling2-sparse = experiment {
    name = "-sparse";
    withCutlass = true;
    sparse = true;
  };
  vllm-xpu-inkling2-kvarn = experiment {
    name = "-kvarn";
    withCutlass = true;
    sparse = true;
    kvarn = true;
  };
}
