# Kernel-build factories for vllm-xpu-kernels. Returns:
#   - mkKernelLibs: the six per-feature kernel *lib* derivations
#     (attn / gdn-attn / mqa-logits / mhc / grouped-gemm xe-2 / xe-default)
#   - mkVllmXpuKernels: the python package wiring those libs together;
#     makeOverridable, so consumers tune it via
#     `.override { aotDevices = [...]; kernelConfig = {...}; useCcache = ...; }`
{
  pkgs,
  intel-oneapi,
  intel-pti,
  torch-xpu,
  cutlass-src,
}: let
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
    rev = "80afa71049cd69a3df32adcccb623b12cd7baa22";
    hash = "sha256-t5+DF4/qgEYQpTY8Qox0BTfpykfs5kFqYy6HrEJaVu0=";
  };

  mkXpuLibFactory = {
    src,
    version,
    aotDevices ? [],
    useCcache ? true,
    kernelChunkPrefillConfig ? null,
    kernelPagedDecodeConfig ? null,
    kernelChunkPrefillExtra ? [],
    kernelPagedDecodeExtra ? [],
  }: let
    factory = pkgs.callPackage ./vllm-xpu-lib.nix {
      intel-oneapi-base = intel-oneapi;
      inherit intel-pti torch-xpu;
      python3Packages = pkgs.python312Packages;
      inherit src version;
      inherit cutlass-src onednn-src;
    };
  in
    {
      libName,
      featureFlags ? [],
      buildDependencies ? [],
      compileJobs ? null,
    }:
      factory {inherit libName featureFlags aotDevices useCcache kernelChunkPrefillConfig kernelPagedDecodeConfig kernelChunkPrefillExtra kernelPagedDecodeExtra buildDependencies compileJobs;};

  # Per-lib feature flag matrices: enable only the chosen lib's source
  # subdir, disable all other libs and ext modules. VLLM_XPU_LIBS_ONLY
  # short-circuits the ext-module section.
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

  mkKernelLibs = {
    src,
    version,
    aotDevices ? [],
    useCcache ? true,
    kernelChunkPrefillConfig ? null,
    kernelPagedDecodeConfig ? null,
    kernelChunkPrefillExtra ? [],
    kernelPagedDecodeExtra ? [],
  }: let
    mkLib = mkXpuLibFactory {inherit src version aotDevices useCcache kernelChunkPrefillConfig kernelPagedDecodeConfig kernelChunkPrefillExtra kernelPagedDecodeExtra;};
    # Each library still builds with NIX_BUILD_CORES internally. Chaining the
    # derivations prevents the daemon from running several SYCL compiler farms
    # at once; GDN alone peaks at tens of GiB under icpx -O3.
    gdnAttn = mkLib {
      libName = "gdn_attn_kernels_xe_2";
      featureFlags = gdnAttnFlags;
    };
    groupedGemmXeDefault = mkLib {
      libName = "grouped_gemm_xe_default";
      featureFlags = groupedGemmXeDefaultFlags;
      buildDependencies = [gdnAttn];
    };
    groupedGemmXe2 = mkLib {
      libName = "grouped_gemm_xe_2";
      featureFlags = groupedGemmXe2Flags;
      buildDependencies = [groupedGemmXeDefault];
    };
    mhc = mkLib {
      libName = "mhc_kernels_xe_2";
      featureFlags = mhcFlags;
      buildDependencies = [groupedGemmXe2];
    };
    mqaLogits = mkLib {
      libName = "mqa_logits_kernels_xe_2";
      featureFlags = mqaLogitsFlags;
      buildDependencies = [mhc];
    };
    attn = mkLib {
      libName = "attn_kernels_xe_2";
      featureFlags = attnFlags;
      buildDependencies = [mqaLogits];
      # oneAPI 2026 frontends for the generated FA2 translation units are
      # substantially heavier than the other split libraries. Keep device
      # linking at NIX_BUILD_CORES, but bound simultaneous C++ frontends.
      compileJobs = 12;
    };
  in {
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
  # `aotDevices` sets the SYCL AOT target list; the default [] is JIT:
  # kernels ship as SPIR-V and IGC specializes them at first dispatch (the
  # 256-GRF hint is preserved via patches/0006-decouple-256grf-from-aot.patch
  # so JIT codegen matches AOT codegen quality, only the first-dispatch
  # pause differs). `[ "bmg" ]` is the Battlemage target this project is
  # tuned for.
  #
  # `kernelConfig` is the partial-buildout selector (upstream #324): compile
  # only the attn-kernel variants a deployment dispatches to. Pass preset
  # names, plus optional extra config lines appended to that preset at build
  # time (no fork needed). Omit a field to keep the full sweep for that
  # stage.
  mkVllmXpuKernels = pkgs.lib.makeOverridable ({
    src,
    version,
    aotDevices ? [],
    useCcache ? true,
    kernelConfig ? {},
  }: let
    cfg = {
      chunkPrefill = null;
      pagedDecode = null;
      chunkPrefillExtra = [];
      pagedDecodeExtra = [];
    } // kernelConfig;
    kernelCfg = {
      kernelChunkPrefillConfig = cfg.chunkPrefill;
      kernelPagedDecodeConfig = cfg.pagedDecode;
      kernelChunkPrefillExtra = cfg.chunkPrefillExtra;
      kernelPagedDecodeExtra = cfg.pagedDecodeExtra;
    };
    libs = mkKernelLibs ({inherit src version aotDevices useCcache;} // kernelCfg);
  in
    pkgs.callPackage ./vllm-xpu-kernels.nix ({
        intel-oneapi-base = intel-oneapi;
        inherit intel-pti torch-xpu useCcache;
        python3Packages = pkgs.python312Packages;
        inherit src version aotDevices;
        inherit cutlass-src onednn-src;
      }
      // libs));
in {
  inherit mkKernelLibs mkVllmXpuKernels;
}
