# Kernel-build factories for vllm-xpu-kernels. Returns:
#   - mkKernelLibs: the six per-feature kernel *lib* derivations
#     (attn / gdn-attn / mqa-logits / mhc / grouped-gemm xe-2 / xe-default)
#   - mkVllmXpuKernels: a composed Python package with reusable non-FA2 glue
#     and a narrow FA2 binding that wires those libs together;
#     makeOverridable, so consumers tune it via
#     `.override { aotDevices = [...]; kernelConfig = {...}; useCcache = ...; }`
{
  pkgs,
  intel-oneapi,
  intel-pti,
  torch-xpu,
  cutlass-src,
}:
let
  mkKernelLibSrc = import ./lib/kernel-lib-src.nix { inherit (pkgs) lib; };
  mkKernelGlueSrc = import ./lib/kernel-glue-src.nix { inherit (pkgs) lib; };

  # The final composed Python package retains the upstream revision-bearing
  # version. Split .so and native-glue components use the filtered source-store
  # hash as their artifact source identity. Consequently, a change confined to
  # a later target subtree (notably attention, which is last in the ordering
  # chain below) leaves earlier derivations stable. Shared CMake/common-source
  # changes intentionally change every projection, and predecessor changes also
  # invalidate successors through buildDependencies.
  mkProjectedVersion =
    version: src:
    let
      baseVersion = builtins.head (pkgs.lib.splitString "+" version);
      sourceStoreHash = builtins.head (pkgs.lib.splitString "-" (builtins.baseNameOf (toString src)));
    in
    "${baseVersion}+src.${sourceStoreHash}";

  # oneDNN is no longer vendored as the `third_party/oneDNN` submodule; the
  # kernels' cmake/Modules/FindoneDNN.cmake clones it via FetchContent at
  # configure time, which the offline Nix sandbox cannot do. Prefetch it as a
  # fixed-output derivation and point FetchContent at the local checkout (see
  # vllm-xpu-lib.nix / vllm-xpu-kernels.nix cmake wiring).
  # TODO: keep `rev` in sync with ONEDNN_GIT_TAG in the kernels' CMakeLists.txt
  #   (https://github.com/vllm-project/vllm-xpu-kernels/blob/main/CMakeLists.txt).
  onednn-src = pkgs.fetchFromGitHub {
    owner = "uxlfoundation";
    repo = "oneDNN";
    rev = "0e2a5bfeef1bfbffc3137464606540233086ce9b";
    hash = "sha256-wgYcZT04nL6ALG0sNkA4fjfkYag/l4CQY4P6S5TrJZo=";
  };

  mkXpuLibFactory =
    {
      src,
      version,
      sourceRevision ? null,
      aotDevices ? [ ],
      useCcache ? true,
      kernelChunkPrefillConfig ? null,
      kernelPagedDecodeConfig ? null,
      kernelChunkPrefillExtra ? [ ],
      kernelPagedDecodeExtra ? [ ],
    }:
    {
      libName,
      featureFlags ? [ ],
      buildDependencies ? [ ],
      compileJobs ? null,
    }:
    let
      libSrc = mkKernelLibSrc {
        inherit src libName;
      };
      libVersion = mkProjectedVersion version libSrc;
      isAttn = libName == "attn_kernels_xe_2";
      factory = pkgs.callPackage ./vllm-xpu-lib.nix {
        intel-oneapi-base = intel-oneapi;
        inherit intel-pti torch-xpu;
        python3Packages = pkgs.python312Packages;
        src = libSrc;
        version = libVersion;
        inherit cutlass-src onednn-src;
      };
    in
    (factory {
      inherit
        libName
        featureFlags
        aotDevices
        useCcache
        buildDependencies
        compileJobs
        ;
      # These selectors mutate/reference attention config files.  Keeping them
      # out of sibling derivations avoids both false invalidation and missing
      # paths in their filtered sources.
      kernelChunkPrefillConfig = if isAttn then kernelChunkPrefillConfig else null;
      kernelPagedDecodeConfig = if isAttn then kernelPagedDecodeConfig else null;
      kernelChunkPrefillExtra = if isAttn then kernelChunkPrefillExtra else [ ];
      kernelPagedDecodeExtra = if isAttn then kernelPagedDecodeExtra else [ ];
    }).overrideAttrs
      (old: {
        passthru = (old.passthru or { }) // {
          kernelSourceProvenance = {
            library = libName;
            # This projection is the source identity that participates in the
            # derivation.  Preserve it in benchmark provenance whenever a split
            # library is attested.
            artifactIdentity = {
              scheme = "nix-filtered-source-store-hash-v1";
              filteredSource = toString libSrc;
              filteredSourceStoreHash = builtins.head (
                pkgs.lib.splitString "-" (builtins.baseNameOf (toString libSrc))
              );
            };
            # These fields say that the current aggregate checkout is compatible
            # with the projected source.  They are passthru metadata, deliberately
            # absent from the split derivation identity so an unchanged projection
            # can be reused across upstream commits.
            compatibilityProvenance = {
              upstreamVersion = version;
              upstreamRevision = sourceRevision;
            };
          };
        };
      });

  # Per-lib feature flag matrices: enable only the chosen library's source
  # subdir. Ninja builds the named target directly, so configured extension
  # targets do not add compile work.
  attnFlags = [
    "-DVLLM_XPU_ENABLE_XE_DEFAULT=OFF"
    "-DBASIC_KERNELS_ENABLED=OFF"
    "-DFA2_KERNELS_ENABLED=ON"
    "-DMOE_KERNELS_ENABLED=OFF"
    "-DGDN_KERNELS_ENABLED=OFF"
    "-DMQA_LOGITS_KERNELS_ENABLED=OFF"
    "-DMHC_KERNELS_ENABLED=OFF"
    "-DXPU_SPECIFIC_KERNELS_ENABLED=OFF"
    "-DXPUMEM_ALLOCATOR_ENABLED=OFF"
  ];
  gdnAttnFlags = [
    "-DVLLM_XPU_ENABLE_XE_DEFAULT=OFF"
    "-DBASIC_KERNELS_ENABLED=OFF"
    "-DFA2_KERNELS_ENABLED=OFF"
    "-DMOE_KERNELS_ENABLED=OFF"
    "-DGDN_KERNELS_ENABLED=ON"
    "-DMQA_LOGITS_KERNELS_ENABLED=OFF"
    "-DMHC_KERNELS_ENABLED=OFF"
    "-DXPU_SPECIFIC_KERNELS_ENABLED=OFF"
    "-DXPUMEM_ALLOCATOR_ENABLED=OFF"
  ];
  mqaLogitsFlags = [
    "-DVLLM_XPU_ENABLE_XE_DEFAULT=OFF"
    "-DBASIC_KERNELS_ENABLED=OFF"
    "-DFA2_KERNELS_ENABLED=OFF"
    "-DMOE_KERNELS_ENABLED=OFF"
    "-DGDN_KERNELS_ENABLED=OFF"
    "-DMQA_LOGITS_KERNELS_ENABLED=ON"
    "-DMHC_KERNELS_ENABLED=OFF"
    "-DXPU_SPECIFIC_KERNELS_ENABLED=OFF"
    "-DXPUMEM_ALLOCATOR_ENABLED=OFF"
  ];
  mhcFlags = [
    "-DVLLM_XPU_ENABLE_XE_DEFAULT=OFF"
    "-DBASIC_KERNELS_ENABLED=OFF"
    "-DFA2_KERNELS_ENABLED=OFF"
    "-DMOE_KERNELS_ENABLED=OFF"
    "-DGDN_KERNELS_ENABLED=OFF"
    "-DMQA_LOGITS_KERNELS_ENABLED=OFF"
    "-DMHC_KERNELS_ENABLED=ON"
    "-DXPU_SPECIFIC_KERNELS_ENABLED=OFF"
    "-DXPUMEM_ALLOCATOR_ENABLED=OFF"
  ];
  groupedGemmXe2Flags = [
    "-DVLLM_XPU_ENABLE_XE_DEFAULT=OFF"
    "-DBASIC_KERNELS_ENABLED=OFF"
    "-DFA2_KERNELS_ENABLED=OFF"
    "-DMOE_KERNELS_ENABLED=ON"
    "-DGDN_KERNELS_ENABLED=OFF"
    "-DMQA_LOGITS_KERNELS_ENABLED=OFF"
    "-DMHC_KERNELS_ENABLED=OFF"
    "-DXPU_SPECIFIC_KERNELS_ENABLED=OFF"
    "-DXPUMEM_ALLOCATOR_ENABLED=OFF"
  ];
  groupedGemmXeDefaultFlags = [
    "-DVLLM_XPU_ENABLE_XE2=OFF"
    "-DVLLM_XPU_ENABLE_XE_DEFAULT=ON"
    "-DBASIC_KERNELS_ENABLED=OFF"
    "-DFA2_KERNELS_ENABLED=OFF"
    "-DMOE_KERNELS_ENABLED=ON"
    "-DGDN_KERNELS_ENABLED=OFF"
    "-DMQA_LOGITS_KERNELS_ENABLED=OFF"
    "-DMHC_KERNELS_ENABLED=OFF"
    "-DXPU_SPECIFIC_KERNELS_ENABLED=OFF"
    "-DXPUMEM_ALLOCATOR_ENABLED=OFF"
  ];

  mkKernelLibs =
    {
      src,
      version,
      sourceRevision ? null,
      aotDevices ? [ ],
      useCcache ? true,
      kernelChunkPrefillConfig ? null,
      kernelPagedDecodeConfig ? null,
      kernelChunkPrefillExtra ? [ ],
      kernelPagedDecodeExtra ? [ ],
    }:
    let
      mkLib = mkXpuLibFactory {
        inherit
          src
          version
          sourceRevision
          aotDevices
          useCcache
          kernelChunkPrefillConfig
          kernelPagedDecodeConfig
          kernelChunkPrefillExtra
          kernelPagedDecodeExtra
          ;
      };
      # Each library still builds with NIX_BUILD_CORES internally. Chaining the
      # derivations prevents the daemon from running several SYCL compiler farms
      # at once; GDN alone peaks at tens of GiB under icpx -O3. This is a
      # directional cache tradeoff, not a fully independent graph: changing a
      # predecessor invalidates every successor. Attention is last so the common
      # Kvarn iteration case reuses all five earlier libraries.
      gdnAttn = mkLib {
        libName = "gdn_attn_kernels_xe_2";
        featureFlags = gdnAttnFlags;
      };
      groupedGemmXeDefault = mkLib {
        libName = "grouped_gemm_xe_default";
        featureFlags = groupedGemmXeDefaultFlags;
        buildDependencies = [ gdnAttn ];
      };
      groupedGemmXe2 = mkLib {
        libName = "grouped_gemm_xe_2";
        featureFlags = groupedGemmXe2Flags;
        buildDependencies = [ groupedGemmXeDefault ];
      };
      mhc = mkLib {
        libName = "mhc_kernels_xe_2";
        featureFlags = mhcFlags;
        buildDependencies = [ groupedGemmXe2 ];
      };
      mqaLogits = mkLib {
        libName = "mqa_logits_kernels_xe_2";
        featureFlags = mqaLogitsFlags;
        buildDependencies = [ mhc ];
      };
      attn = mkLib {
        libName = "attn_kernels_xe_2";
        featureFlags = attnFlags;
        buildDependencies = [ mqaLogits ];
        # oneAPI 2026 frontends for the generated FA2 translation units are
        # substantially heavier than the other split libraries. Keep device
        # linking at NIX_BUILD_CORES, but bound simultaneous C++ frontends.
        compileJobs = 12;
      };
    in
    {
      gdn-attn-kernels-xe-2 = gdnAttn;
      grouped-gemm-xe-default = groupedGemmXeDefault;
      grouped-gemm-xe-2 = groupedGemmXe2;
      mhc-kernels-xe-2 = mhc;
      mqa-logits-kernels-xe-2 = mqaLogits;
      attn-kernels-xe-2 = attn;
    };

  # Overridable via the standard nixpkgs `.override` mechanism:
  #
  #   vllm-xpu-kernels.override {
  #     aotDevices = [ "bmg" ];
  #     kernelConfig = {
  #       chunkPrefill = "chunk_prefill_default";
  #       chunkPrefillExtra = [ "256,true,true,false,false,false" ];
  #       pagedDecode = "paged_decode_default";
  #       pagedDecodeExtra = [ "8,256,64,true,false,false" ];
  #     };
  #   }
  #
  # `aotDevices` sets the SYCL AOT target list; the default [] is JIT and
  # kernels ship as SPIR-V for IGC to specialize at first dispatch.
  # `[ "bmg" ]` enables the upstream Battlemage AOT path, including its
  # 256-GRF tuning, and is the stable Brutus configuration.
  #
  # `kernelConfig` is the partial-buildout selector (upstream #324): compile
  # only the attn-kernel variants a deployment dispatches to. Pass preset
  # names, plus optional extra config lines appended to that preset at build
  # time (no fork needed). Omit a field to keep the full sweep for that
  # stage.
  mkVllmXpuKernels = pkgs.lib.makeOverridable (
    {
      src,
      version,
      sourceRevision ? null,
      aotDevices ? [ ],
      useCcache ? true,
      kernelConfig ? { },
    }:
    let
      cfg = {
        chunkPrefill = null;
        pagedDecode = null;
        chunkPrefillExtra = [ ];
        pagedDecodeExtra = [ ];
      }
      // kernelConfig;
      kernelCfg = {
        kernelChunkPrefillConfig = cfg.chunkPrefill;
        kernelPagedDecodeConfig = cfg.pagedDecode;
        kernelChunkPrefillExtra = cfg.chunkPrefillExtra;
        kernelPagedDecodeExtra = cfg.pagedDecodeExtra;
      };
      libs = mkKernelLibs (
        {
          inherit
            src
            version
            sourceRevision
            aotDevices
            useCcache
            ;
        }
        // kernelCfg
      );
      baseGlueSrc = mkKernelGlueSrc {
        inherit src;
        component = "base";
      };
      fa2BindingSrc = mkKernelGlueSrc {
        inherit src;
        component = "fa2";
      };
      componentCommon = {
        intel-oneapi-base = intel-oneapi;
        inherit intel-pti torch-xpu useCcache;
        python3Packages = pkgs.python312Packages;
        inherit aotDevices cutlass-src onednn-src;
      };
      baseGlue = pkgs.callPackage ./vllm-xpu-kernels.nix (
        componentCommon
        // libs
        // {
          pname = "vllm-xpu-kernels-base-glue";
          src = baseGlueSrc;
          version = mkProjectedVersion version baseGlueSrc;
          featureOptions = {
            BUILD_SYCL_TLA_KERNELS = true;
            VLLM_XPU_ENABLE_XE2 = true;
            VLLM_XPU_ENABLE_XE_DEFAULT = true;
            BASIC_KERNELS_ENABLED = true;
            FA2_KERNELS_ENABLED = false;
            MOE_KERNELS_ENABLED = true;
            GDN_KERNELS_ENABLED = true;
            MQA_LOGITS_KERNELS_ENABLED = true;
            MHC_KERNELS_ENABLED = true;
            XPU_SPECIFIC_KERNELS_ENABLED = true;
            XPUMEM_ALLOCATOR_ENABLED = true;
            VLLM_XPU_ENABLE_ONEDNN = true;
          };
          withAttnLibrary = false;
        }
      );
      fa2Binding =
        (pkgs.callPackage ./vllm-xpu-kernels.nix (
          componentCommon
          // libs
          // {
            pname = "vllm-xpu-kernels-fa2-binding";
            src = fa2BindingSrc;
            version = mkProjectedVersion version fa2BindingSrc;
            featureOptions = {
              BUILD_SYCL_TLA_KERNELS = true;
              VLLM_XPU_ENABLE_XE2 = true;
              VLLM_XPU_ENABLE_XE_DEFAULT = false;
              BASIC_KERNELS_ENABLED = false;
              FA2_KERNELS_ENABLED = true;
              MOE_KERNELS_ENABLED = false;
              GDN_KERNELS_ENABLED = false;
              MQA_LOGITS_KERNELS_ENABLED = false;
              MHC_KERNELS_ENABLED = false;
              XPU_SPECIFIC_KERNELS_ENABLED = false;
              XPUMEM_ALLOCATOR_ENABLED = false;
              VLLM_XPU_ENABLE_ONEDNN = false;
            };
            withGdnAttnLibrary = false;
            withMqaLogitsLibrary = false;
            withMhcLibrary = false;
            withGroupedGemmXe2Library = false;
            withGroupedGemmXeDefaultLibrary = false;
            pythonImportsCheck = [ ];
          }
        )).overrideAttrs
          (old: {
            passthru = (old.passthru or { }) // {
              attentionLibrary = libs.attn-kernels-xe-2;
            };
          });
      composed = pkgs.callPackage ./vllm-xpu-kernels-compose.nix {
        inherit
          baseGlue
          fa2Binding
          torch-xpu
          version
          ;
        python3Packages = pkgs.python312Packages;
      };
    in
    composed.overrideAttrs (old: {
      passthru = (old.passthru or { }) // {
        kernelLibraries = libs;
        kernelComponents = {
          base-glue = baseGlue;
          fa2-binding = fa2Binding;
        };
        kernelPackagingProvenance = {
          scheme = "split-native-glue-v1";
          baseGlueSource = toString baseGlueSrc;
          fa2BindingSource = toString fa2BindingSrc;
          compatibilityRevision = sourceRevision;
        };
      };
    })
  );
in
{
  inherit mkKernelLibs mkVllmXpuKernels;
}
