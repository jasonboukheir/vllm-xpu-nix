# Kernel-build factories for vllm-xpu-kernels. Returns:
#   - mkKernelLibs: the five per-feature kernel *lib* derivations
#     (attn / gdn-attn / mqa-logits / grouped-gemm xe-2 / xe-default)
#   - mkVllmXpuKernels: the python package wiring those libs together,
#     with withAotDevices / withJIT / withAOT / withCcache /
#     withKernelConfig passthrus
{
  pkgs,
  intel-oneapi,
  intel-pti,
  torch-xpu,
  cutlass-src,
}: let
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
      inherit cutlass-src;
    };
  in
    {
      libName,
      featureFlags ? [],
    }:
      factory {inherit libName featureFlags aotDevices useCcache kernelChunkPrefillConfig kernelPagedDecodeConfig kernelChunkPrefillExtra kernelPagedDecodeExtra;};

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
  in {
    attn-kernels-xe-2 = mkLib {
      libName = "attn_kernels_xe_2";
      featureFlags = attnFlags;
    };
    gdn-attn-kernels-xe-2 = mkLib {
      libName = "gdn_attn_kernels_xe_2";
      featureFlags = gdnAttnFlags;
    };
    mqa-logits-kernels-xe-2 = mkLib {
      libName = "mqa_logits_kernels_xe_2";
      featureFlags = mqaLogitsFlags;
    };
    grouped-gemm-xe-2 = mkLib {
      libName = "grouped_gemm_xe_2";
      featureFlags = groupedGemmXe2Flags;
    };
    grouped-gemm-xe-default = mkLib {
      libName = "grouped_gemm_xe_default";
      featureFlags = groupedGemmXeDefaultFlags;
    };
  };

  # `withAotDevices` / `withJIT` / `withAOT` re-derive the closure with a
  # different SYCL AOT target list. The default is JIT: kernels ship as
  # SPIR-V and IGC specializes them at first dispatch (the 256-GRF hint is
  # preserved via patches/0006-decouple-256grf-from-aot.patch so JIT codegen
  # matches AOT codegen quality, only the first-dispatch pause differs).
  # `withAOT` is a shortcut for `withAotDevices [ "bmg" ]` — Battlemage
  # being the target this project is tuned for. `withAotDevices [ ... ]` for
  # any other explicit list.
  mkVllmXpuKernels = {
    src,
    version,
    aotDevices ? [],
    useCcache ? true,
    kernelChunkPrefillConfig ? null,
    kernelPagedDecodeConfig ? null,
    kernelChunkPrefillExtra ? [],
    kernelPagedDecodeExtra ? [],
  }: let
    kernelCfg = {inherit kernelChunkPrefillConfig kernelPagedDecodeConfig kernelChunkPrefillExtra kernelPagedDecodeExtra;};
    libs = mkKernelLibs ({inherit src version aotDevices useCcache;} // kernelCfg);
    base = pkgs.callPackage ./vllm-xpu-kernels.nix ({
        intel-oneapi-base = intel-oneapi;
        inherit intel-pti torch-xpu useCcache;
        python3Packages = pkgs.python312Packages;
        inherit src version aotDevices;
        inherit cutlass-src;
      }
      // libs);
  in
    base.overrideAttrs (old: {
      passthru =
        (old.passthru or {})
        // {
          withAotDevices = ds:
            mkVllmXpuKernels ({
                inherit src version useCcache;
                aotDevices = ds;
              }
              // kernelCfg);
          withJIT = mkVllmXpuKernels ({
              inherit src version useCcache;
              aotDevices = [];
            }
            // kernelCfg);
          withAOT = mkVllmXpuKernels ({
              inherit src version useCcache;
              aotDevices = ["bmg"];
            }
            // kernelCfg);
          withCcache = b:
            mkVllmXpuKernels ({
                inherit src version aotDevices;
                useCcache = b;
              }
              // kernelCfg);
          # Partial-buildout selector (upstream #324): compile only the
          # attn-kernel variants a deployment dispatches to. Pass preset
          # names, plus optional extra config lines appended to that
          # preset at build time (no fork needed), e.g.
          #   .withKernelConfig {
          #     chunkPrefill = "chunk_prefill_default";
          #     chunkPrefillExtra = [ "256,true,true,false,false,false" ];
          #     pagedDecode = "paged_decode_default";
          #     pagedDecodeExtra = [ "8,256,64,true,false,false" ];
          #   }
          # Omit a field to keep the full sweep for that stage.
          withKernelConfig = {
            chunkPrefill ? null,
            pagedDecode ? null,
            chunkPrefillExtra ? [],
            pagedDecodeExtra ? [],
          }:
            mkVllmXpuKernels {
              inherit src version aotDevices useCcache;
              kernelChunkPrefillConfig = chunkPrefill;
              kernelPagedDecodeConfig = pagedDecode;
              kernelChunkPrefillExtra = chunkPrefillExtra;
              kernelPagedDecodeExtra = pagedDecodeExtra;
            };
        };
    });
in {
  inherit mkKernelLibs mkVllmXpuKernels;
}
