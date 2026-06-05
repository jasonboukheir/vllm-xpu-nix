{
  description = "Nix-native Intel XPU substrate for vLLM (torch+xpu, triton-xpu, vllm-xpu-kernels, vllm)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";

    vllm-xpu-kernels-src = {
      type = "git";
      url = "https://github.com/vllm-project/vllm-xpu-kernels.git";
      ref = "release/v0.1.9.1";
      submodules = true;
      flake = false;
    };

    vllm-xpu-kernels-unstable-src = {
      type = "git";
      url = "https://github.com/jasonboukheir/vllm-xpu-kernels.git";
      submodules = true;
      flake = false;
    };

    vllm-xpu-src = {
      type = "git";
      url = "https://github.com/vllm-project/vllm.git";
      ref = "refs/heads/releases/v0.22.0";
      flake = false;
    };

    vllm-xpu-unstable-src = {
      type = "git";
      url = "https://github.com/jasonboukheir/vllm.git";
      flake = false;
    };

    sycl-tla-src = {
      url = "github:intel/sycl-tla/cd763790ad2f74d7294435ecf77682bac0062c3a";
      flake = false;
    };
  };

  outputs = { self, nixpkgs, flake-utils, vllm-xpu-kernels-src, vllm-xpu-kernels-unstable-src, vllm-xpu-src, vllm-xpu-unstable-src, sycl-tla-src }:
    let
      systemOutputs = flake-utils.lib.eachSystem [ "x86_64-linux" ] (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };

        # Narrow the vllm-xpu-kernels checkout to just the files the build
        # actually consumes. Without this, the whole repo (docs/, tests/,
        # benchmark/, README.md, Dockerfile.xpu, tools/) feeds configureDrv's
        # input hash, so upstream README/test/CI edits bump configureDrv's
        # store path and cascade through every per-TU drv.
        #
        # `lib.fileset.toSource` would be the idiomatic API for this kind
        # of narrowing, but its `root` must be an eval-time path (it
        # rejects store-path strings under pure-eval), and flake inputs
        # arrive as store paths. `lib.sources.sourceByRegex` filters via
        # `builtins.filterSource`, which accepts store paths and produces
        # an equivalent narrowed source.
        #
        # Patterns cover every file referenced by the cmake graph or read
        # by `pip install`, including everything touched by
        # `nix/patches/000*-*.patch` (CMakeLists.txt, cmake/utils.cmake,
        # setup.py, vllm_xpu_kernels/__init__.py, csrc/xpu/attn/xe_2/*).
        # tools/ is required because setup.py loads tools/envs.py to read
        # VLLM_TARGET_DEVICE. third_party/ stays because
        # cmake/Modules/FindoneDNN.cmake reads third_party/oneDNN (the
        # project's own Find module, not oneAPI).
        mkKernelsSrc = rawSrc: pkgs.lib.sources.sourceByRegex rawSrc [
          "^CMakeLists\\.txt$"
          "^cmake(/.*)?$"
          "^csrc(/.*)?$"
          "^setup\\.py$"
          "^pyproject\\.toml$"
          "^tools(/.*)?$"
          "^vllm_xpu_kernels(/.*)?$"
          "^third_party(/.*)?$"
        ];
        vllm-xpu-kernels-src' = mkKernelsSrc vllm-xpu-kernels-src;
        vllm-xpu-kernels-unstable-src' = mkKernelsSrc vllm-xpu-kernels-unstable-src;

        # Stamp a derivation's version from a flake input's lock metadata
        # rather than a hand-bumped literal. `base` is the upstream release
        # the pin descends from; if omitted, it's parsed from the input's
        # `original.ref` in flake.lock (e.g. "release/v0.1.9.1",
        # "refs/heads/releases/v0.22.0"), so release-tracking pins need
        # zero ceremony. `unstable=true` is for main-tracking pins where
        # every lock bump moves the source — the lock date goes into the
        # local-version suffix so the store path shifts in lockstep.
        #
        # Reading from flake.lock is necessary because inputs reaching
        # `outputs` only carry the resolved metadata (rev, narHash,
        # lastModified, ...) — the `original` spec with the ref is not
        # exposed. The lock file is part of the flake's source, so this
        # is an eval-time file read, not IFD.
        #
        # Output is PEP 440-clean (`+local` with `[a-zA-Z0-9.]` payload):
        # vllm forwards this to VLLM_VERSION_OVERRIDE ->
        # SETUPTOOLS_SCM_PRETEND_VERSION, which rejects '-'-separated
        # local tags. Kernels are Nix-label-only, but using one format
        # keeps the helper trivial.
        flakeLock = builtins.fromJSON (builtins.readFile ./flake.lock);
        mkInputVersion = { name, input, base ? null, unstable ? false }:
          let
            ref = flakeLock.nodes.${name}.original.ref or "";
            matched = builtins.match ".*v([0-9]+(\\.[0-9]+)*)" ref;
            effectiveBase =
              if base != null then base
              else if matched != null then builtins.head matched
              else throw "mkInputVersion: no `base` given and could not parse version from flake.lock ref of ${name} = ${toString ref}";
            rev = input.shortRev or "dirty";
            d   = input.lastModifiedDate or "00000000000000";
            ymd = "${builtins.substring 0 4 d}.${builtins.substring 4 2 d}.${builtins.substring 6 2 d}";
          in
            if unstable
            then "${effectiveBase}+unstable.${ymd}.g${rev}"
            else "${effectiveBase}+g${rev}";

        kernelsStableVersion = mkInputVersion {
          name = "vllm-xpu-kernels-src";
          input = vllm-xpu-kernels-src;
        };
        kernelsUnstableVersion = mkInputVersion {
          name = "vllm-xpu-kernels-unstable-src";
          input = vllm-xpu-kernels-unstable-src;
          base = "0.1.9.1";
          unstable = true;
        };
        vllmStableVersion = mkInputVersion {
          name = "vllm-xpu-src";
          input = vllm-xpu-src;
        };
        vllmUnstableVersion = mkInputVersion {
          name = "vllm-xpu-unstable-src";
          input = vllm-xpu-unstable-src;
          base = "0.22.0";
          unstable = true;
        };

        # Unified oneAPI 2026.0 base toolkit (libsycl.so.9, libmkl_*.so.3 /
        # libmkl_sycl_*.so.6, oneCCL 2022.0.0). Pairs with the torch+xpu
        # nightly we track in nix/torch-xpu.nix.
        #
        # oneCCL 2022.0.0 ships with the unified toolkit and supersedes the
        # standalone oneccl-bmg (2021.15.9.14) we shipped against the
        # earlier 2025.3 base toolkit:
        #   - 2021.17.2 added BMG single-process / multi-thread support
        #     for allreduce, allgatherv, reduce_scatter
        #   - 2022.0.0 added Arc Pro B-Series SPMD allreduce / allgather
        #     / alltoall / reduce_scatter / broadcast / pt2pt
        # The 2021.15.9.x line has no further patches; it also pinned
        # libsycl.so.8, which the 2026.0 toolkit's libsycl.so.9 cannot
        # satisfy — so going back to a stable torch (which links 2025.x)
        # is the only way to use the standalone oneccl-bmg.
        intel-oneapi = (pkgs.intel-oneapi-toolkit.override {
          components = [
            "intel.oneapi.lin.dpcpp-cpp-compiler"
            "intel.oneapi.lin.mkl.devel"
            "intel.oneapi.lin.dpl"
            "intel.oneapi.lin.ccl.devel"
          ];
        }).overrideAttrs (old: {
          # nixpkgs' intel-oneapi-toolkit has no `depsByComponent.ccl` entry
          # (only dpcpp-cpp-compiler / mpi / pti / vtune / mkl / etc.), so
          # adding the ccl component to the install list pulls the binaries
          # in without their native deps. autoPatchelfHook then fails on
          # libccl.so.1 -> libfabric/librdmacm/libibverbs/libpsm2/libucp/...
          # Mirror the depsByComponent.ccl set our standalone oneccl-bmg
          # derivation used.
          # TODO: upstream a depsByComponent.ccl entry to nixpkgs'
          # intel-oneapi-toolkit and drop this list.
          buildInputs = (old.buildInputs or [ ]) ++ (with pkgs; [
            rdma-core
            libpsm2
            ucx
            numactl
            libffi
            libuuid
            libfabric
          ]);
          postInstall = (old.postInstall or "") + ''
            # libccl dlopens libfabric.so when CCL_ATL_TRANSPORT=ofi: first by
            # name, then as `<dirname libccl.so>/libfabric.so`. With neither
            # resolvable, OFI init silently fails ("OFI transport was not
            # initialized, fallback to MPI transport") and libccl falls back
            # to libmpi.so.12 — which is on libtorch_xpu's RPATH via the
            # toolkit but never MPI_Init-ed, so the next allreduce segfaults
            # inside libmpi (vllm-xpu-nix #39).
            #
            # The libfabric the toolkit bundles is unusable: it's configured
            # with a hardcoded `/usr/local/lib/libfabric` provider path and
            # ships its providers as separate DSOs, so without a runtime
            # FI_PROVIDER_PATH it loads zero providers ("fi_getinfo error:
            # ret -61, providers 0"). nixpkgs' libfabric has the providers we
            # need (tcp, shm, sockets, rxm) compiled into libfabric.so
            # itself, so it works without env-var coaxing — symlink it as a
            # sibling of libccl so the relative-path dlopen succeeds.
            for cclLibDir in "$out"/ccl/*/lib; do
              if [ -d "$cclLibDir" ] && [ ! -e "$cclLibDir/libfabric.so" ]; then
                ln -s ${pkgs.libfabric}/lib/libfabric.so "$cclLibDir/libfabric.so"
              fi
            done
          '';
        });

        intel-pti = pkgs.callPackage ./nix/intel-pti.nix {
          intel-oneapi-base = intel-oneapi;
        };

        torch-xpu = pkgs.callPackage ./nix/torch-xpu.nix {
          intel-oneapi-base = intel-oneapi;
          inherit intel-pti;
          python3Packages = pkgs.python312Packages;
        };

        triton-xpu = pkgs.callPackage ./nix/triton-xpu.nix {
          intel-oneapi-base = intel-oneapi;
          inherit intel-pti;
          python3Packages = pkgs.python312Packages;
        };

        torchvision-xpu = pkgs.callPackage ./nix/torchvision-xpu.nix {
          inherit torch-xpu;
          python3Packages = pkgs.python312Packages;
        };

        # accelerate's nixpkgs definition propagates stock `torch`, which
        # collides with torch-xpu on functorch/*.pyc when both end up in a
        # python.withPackages buildEnv. Rebuild accelerate with its `torch`
        # arg pointing at torch-xpu so the buildEnv merge sees only one
        # torch. Surgical override (vs. python set-wide packageOverrides)
        # avoids re-evaluating unrelated python packages whose passthru
        # references attrs torch-xpu doesn't carry.
        # nixpkgs ships mistral-common 1.8.8; vllm 0.20.x imports
        # NamedToolChoice from mistral_common.protocol.instruct.tool_calls,
        # which only exists from 1.11+. Bump to 1.11.2 (vllm's pin).
        # overridePythonAttrs preserves the nixpkgs build recipe and just
        # swaps version+src, keeping the package self-contained against
        # whatever nixpkgs revision is in flake.lock.
        mistral-common-1_11 = pkgs.python312Packages.mistral-common.overridePythonAttrs (oldAttrs: rec {
          version = "1.11.2";
          src = pkgs.fetchFromGitHub {
            owner = "mistralai";
            repo = "mistral-common";
            rev = "v${version}";
            hash = "sha256-EXdZcBR61GNye8LqwIqRO8lP1lK6fqPJufWFO9XkkYQ=";
          };
          pythonRelaxDeps = (oldAttrs.pythonRelaxDeps or [ ]) ++ [ "numpy" ];
          doCheck = false;
        });

        python312PackagesXpu = pkgs.python312Packages // {
          accelerate = pkgs.python312Packages.accelerate.override {
            torch = torch-xpu;
          };
          mistral-common = mistral-common-1_11;
          torchvision = torchvision-xpu;
        };

        flash-linear-attention = pkgs.callPackage ./nix/flash-linear-attention.nix {
          inherit torch-xpu triton-xpu;
          python3Packages = python312PackagesXpu;
        };

        auto-round-xpu = pkgs.callPackage ./nix/auto-round-xpu.nix {
          inherit torch-xpu triton-xpu flash-linear-attention;
          python3Packages = python312PackagesXpu;
        };

        quantize = pkgs.callPackage ./nix/quantize.nix {
          inherit auto-round-xpu;
          python3Packages = python312PackagesXpu;
        };

        kl-eval = pkgs.callPackage ./nix/kl-eval.nix {
          inherit auto-round-xpu;
          python3Packages = python312PackagesXpu;
        };

        mkXpuLibFactory = { src, version, aotDevices ? [ ], useCcache ? true, kernelChunkPrefillConfig ? null, kernelPagedDecodeConfig ? null, kernelChunkPrefillExtra ? [ ], kernelPagedDecodeExtra ? [ ] }:
          let factory = pkgs.callPackage ./nix/vllm-xpu-lib.nix {
            intel-oneapi-base = intel-oneapi;
            inherit intel-pti torch-xpu;
            python3Packages = pkgs.python312Packages;
            inherit src version;
            cutlass-src = sycl-tla-src;
          };
          in { libName, featureFlags ? [ ] }:
            factory { inherit libName featureFlags aotDevices useCcache kernelChunkPrefillConfig kernelPagedDecodeConfig kernelChunkPrefillExtra kernelPagedDecodeExtra; };

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

        mkKernelLibs = { src, version, aotDevices ? [ ], useCcache ? true, kernelChunkPrefillConfig ? null, kernelPagedDecodeConfig ? null, kernelChunkPrefillExtra ? [ ], kernelPagedDecodeExtra ? [ ] }:
          let mkLib = mkXpuLibFactory { inherit src version aotDevices useCcache kernelChunkPrefillConfig kernelPagedDecodeConfig kernelChunkPrefillExtra kernelPagedDecodeExtra; }; in {
            attn-kernels-xe-2 = mkLib { libName = "attn_kernels_xe_2"; featureFlags = attnFlags; };
            gdn-attn-kernels-xe-2 = mkLib { libName = "gdn_attn_kernels_xe_2"; featureFlags = gdnAttnFlags; };
            mqa-logits-kernels-xe-2 = mkLib { libName = "mqa_logits_kernels_xe_2"; featureFlags = mqaLogitsFlags; };
            grouped-gemm-xe-2 = mkLib { libName = "grouped_gemm_xe_2"; featureFlags = groupedGemmXe2Flags; };
            grouped-gemm-xe-default = mkLib { libName = "grouped_gemm_xe_default"; featureFlags = groupedGemmXeDefaultFlags; };
          };

        # `withAotDevices` / `withJIT` / `withAOT` re-derive the closure
        # with a different SYCL AOT target list. The default is JIT:
        # kernels ship as SPIR-V and IGC specializes them at first
        # dispatch (the 256-GRF hint is preserved via
        # patches/0006-decouple-256grf-from-aot.patch so JIT codegen
        # matches AOT codegen quality, only the first-dispatch pause
        # differs). `withAOT` is a shortcut for `withAotDevices
        # [ "bmg" ]` — Battlemage being the target this project is
        # tuned for. `withAotDevices [ ... ]` for any other explicit
        # list.
        mkVllmXpuKernels = { src, version, aotDevices ? [ ], useCcache ? true, kernelChunkPrefillConfig ? null, kernelPagedDecodeConfig ? null, kernelChunkPrefillExtra ? [ ], kernelPagedDecodeExtra ? [ ] }:
          let
            kernelCfg = { inherit kernelChunkPrefillConfig kernelPagedDecodeConfig kernelChunkPrefillExtra kernelPagedDecodeExtra; };
            libs = mkKernelLibs ({ inherit src version aotDevices useCcache; } // kernelCfg);
            base = pkgs.callPackage ./nix/vllm-xpu-kernels.nix ({
              intel-oneapi-base = intel-oneapi;
              inherit intel-pti torch-xpu useCcache;
              python3Packages = pkgs.python312Packages;
              inherit src version aotDevices;
              cutlass-src = sycl-tla-src;
            } // libs);
          in
            base.overrideAttrs (old: {
              passthru = (old.passthru or {}) // {
                withAotDevices = ds: mkVllmXpuKernels ({
                  inherit src version useCcache; aotDevices = ds;
                } // kernelCfg);
                withJIT = mkVllmXpuKernels ({
                  inherit src version useCcache; aotDevices = [];
                } // kernelCfg);
                withAOT = mkVllmXpuKernels ({
                  inherit src version useCcache; aotDevices = [ "bmg" ];
                } // kernelCfg);
                withCcache = b: mkVllmXpuKernels ({
                  inherit src version aotDevices; useCcache = b;
                } // kernelCfg);
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
                withKernelConfig = { chunkPrefill ? null, pagedDecode ? null, chunkPrefillExtra ? [ ], pagedDecodeExtra ? [ ] }: mkVllmXpuKernels {
                  inherit src version aotDevices useCcache;
                  kernelChunkPrefillConfig = chunkPrefill;
                  kernelPagedDecodeConfig = pagedDecode;
                  kernelChunkPrefillExtra = chunkPrefillExtra;
                  kernelPagedDecodeExtra = pagedDecodeExtra;
                };
              };
            });

        stableLibs = mkKernelLibs {
          src = vllm-xpu-kernels-src';
          version = kernelsStableVersion;
        };
        unstableLibs = mkKernelLibs {
          src = vllm-xpu-kernels-unstable-src';
          version = kernelsUnstableVersion;
        };

        vllm-xpu-kernels = mkVllmXpuKernels {
          src = vllm-xpu-kernels-src';
          version = kernelsStableVersion;
        };
        vllm-xpu-kernels-unstable = mkVllmXpuKernels {
          src = vllm-xpu-kernels-unstable-src';
          version = kernelsUnstableVersion;
        };

        # mkVllm pairs a vllm source pin with the matching kernels build:
        # the upstream stable variant gets vllm-xpu-kernels (vllm-project),
        # the unstable variant gets vllm-xpu-kernels-unstable (jasonboukheir
        # fork). Pre-release version stamp is fine — VLLM_VERSION_OVERRIDE
        # in vllm-xpu.nix forwards to setuptools-scm's PRETEND_VERSION, so
        # setuptools-scm doesn't need a .git in the unpacked store path.
        #
        # Like mkVllmXpuKernels, the result exposes `withAotDevices` /
        # `withJIT` / `withAOT` passthrus that cascade through the kernels
        # package. Also exposes `withTorchvision` and `withAudio` passthrus
        # so consumers can opt into the +xpu torchvision wheel (for VL
        # model families) or soundfile+pyav audio decoders (for /v1/audio
        # transcription endpoints) without spelling out a full `.override`.
        # All passthrus compose:
        # `pkgs.vllm-xpu-unstable.withAOT |> .withTorchvision true |> .withAudio true`.
        #
        # `withTorchaudio` is intentionally not exposed: no consumer in
        # this project's stack needs torchaudio, so we don't carry the
        # extra +xpu wheel pin. Re-introduce if a model family that
        # depends on it lands.
        mkVllm = {
          src, version, kernels,
          withTorchvision ? false,
          withAudio ? false,
        }:
          let
            base = pkgs.callPackage ./nix/vllm-xpu.nix {
              intel-oneapi-base = intel-oneapi;
              inherit intel-pti torch-xpu triton-xpu flash-linear-attention;
              python3Packages = python312PackagesXpu;
              vllm-xpu-kernels = kernels;
              inherit src version withTorchvision withAudio;
              inherit (pkgs) level-zero intel-graphics-compiler intel-compute-runtime;
            };
          in
            base.overrideAttrs (old: {
              passthru = (old.passthru or {}) // {
                withAotDevices = ds: mkVllm {
                  inherit src version withTorchvision withAudio;
                  kernels =
                    if kernels ? withAotDevices
                    then kernels.withAotDevices ds
                    else kernels;
                };
                withJIT = mkVllm {
                  inherit src version withTorchvision withAudio;
                  kernels =
                    if kernels ? withJIT
                    then kernels.withJIT
                    else kernels;
                };
                withAOT = mkVllm {
                  inherit src version withTorchvision withAudio;
                  kernels =
                    if kernels ? withAOT
                    then kernels.withAOT
                    else kernels;
                };
                withTorchvision = b: mkVllm {
                  inherit src version kernels withAudio;
                  withTorchvision = b;
                };
                withAudio = b: mkVllm {
                  inherit src version kernels withTorchvision;
                  withAudio = b;
                };
                # Rebuild the paired kernels package against a narrowed
                # attn-kernel set. See mkVllmXpuKernels.withKernelConfig.
                withKernelConfig = cfg: mkVllm {
                  inherit src version withTorchvision withAudio;
                  kernels =
                    if kernels ? withKernelConfig
                    then kernels.withKernelConfig cfg
                    else kernels;
                };
              };
            });

        vllm-xpu = mkVllm {
          src = vllm-xpu-src;
          version = vllmStableVersion;
          kernels = vllm-xpu-kernels;
        };

        vllm-xpu-unstable = mkVllm {
          src = vllm-xpu-unstable-src;
          version = vllmUnstableVersion;
          kernels = vllm-xpu-kernels-unstable;
        };

        syclToolchainShellHook = ''
          syclHome="${intel-oneapi}/compiler/latest"
          mkdir -p .dev-bin
          ln -sf ${pkgs.intel-compute-runtime}/bin/ocloc-* .dev-bin/ocloc 2>/dev/null || true
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
          export VLLM_CUTLASS_SRC_DIR="${sycl-tla-src}"
        '';
        hfMetadata = pkgs.callPackage ./nix/hf-metadata.nix { };

        # Pinned ruff binaries matching each project's
        # .pre-commit-config.yaml, so `lint` produces the same diagnostics CI
        # would. Nixpkgs' ruff drifts ahead of the projects' pins, and
        # `uv tool run` trips on uv's bundled glibc python under NixOS, so we
        # vendor the official musl static binary per version.
        mkRuff = { version, sha256 }:
          pkgs.stdenvNoCC.mkDerivation {
            pname = "ruff";
            inherit version;
            src = pkgs.fetchurl {
              url = "https://github.com/astral-sh/ruff/releases/download/${version}/ruff-x86_64-unknown-linux-musl.tar.gz";
              inherit sha256;
            };
            sourceRoot = "ruff-x86_64-unknown-linux-musl";
            dontConfigure = true;
            dontBuild = true;
            installPhase = ''
              install -Dm755 ruff "$out/bin/ruff"
            '';
          };
        ruffVllm = mkRuff {
          version = "0.14.0";
          sha256 = "sha256-7W0bhAeh0ijcMy+xkFfobgSmzTwr6s2zJK1v8qP5Bxs=";
        };
        ruffKernels = mkRuff {
          version = "0.11.7";
          sha256 = "sha256-DK8yqww5m/ugjRixkNwarsjqRzaIZKS/Oz8RPjbo/Wg=";
        };

        # Lint local vllm / vllm-xpu-kernels checkouts with the pinned ruff
        # (or pre-commit if available). Defaults to sibling checkouts
        # (../vllm, ../vllm-xpu-kernels relative to $PWD — the ~/Projects
        # layout); override with positional args: lint [VLLM_SRC] [KERNELS_SRC].
        lint = pkgs.writeShellApplication {
          name = "lint";
          runtimeInputs = [ pkgs.git ];
          text = ''
            vllm_src="''${1:-../vllm}"
            kernels_src="''${2:-../vllm-xpu-kernels}"
            run_in() {
              local dir="$1" ruff="$2"
              if [ ! -d "$dir" ]; then
                echo "[$dir] not found — skipping"
                return 0
              fi
              echo "=== $dir ==="
              (
                cd "$dir"
                if [ -f .pre-commit-config.yaml ] && command -v pre-commit >/dev/null 2>&1; then
                  echo "[$dir] pre-commit run --all-files"
                  pre-commit run --all-files
                else
                  if [ ! -f .pre-commit-config.yaml ]; then
                    echo "[$dir] no .pre-commit-config.yaml — running ruff only"
                  else
                    echo "[$dir] pre-commit not installed — falling back to ruff only"
                  fi
                  echo "[$dir] ruff check ."
                  "$ruff" check .
                  echo "[$dir] ruff format --check ."
                  "$ruff" format --check .
                fi
              )
            }
            fail=0
            run_in "$vllm_src"    "${ruffVllm}/bin/ruff"    || fail=1
            run_in "$kernels_src" "${ruffKernels}/bin/ruff" || fail=1
            exit "$fail"
          '';
        };
      in {
        # Per-system helpers consumers reach via
        # `inputs.vllm-xpu-nix.lib.${pkgs.system}.fromHfConfig`.
        # `lib.${system}` (rather than just `lib`) is intentional — the
        # helpers wrap `pkgs.fetchurl` (system-scoped) and the mk* builders
        # close over this system's pkgs / torch-xpu / oneAPI substrate.
        lib = {
          inherit (hfMetadata)
            fetchHfConfig
            readHfConfig
            attnParamsFromConfig
            fromHfConfig
            unionKernelSet;
          # Parameterized builders, so a consumer can build a vllm (or the
          # kernels) from an arbitrary source — e.g. a local submodule
          # checkout — without overriding flake inputs. The packages.*
          # `vllm-xpu`/`vllm-xpu-unstable` outputs are just two fixed
          # instantiations of mkVllm; mkVllm { src; version; kernels; ... }
          # builds against the same pinned substrate from any src.
          inherit mkVllm mkVllmXpuKernels;
        };

        packages = {
          inherit
            intel-oneapi intel-pti
            torch-xpu triton-xpu torchvision-xpu
            flash-linear-attention
            auto-round-xpu
            vllm-xpu-kernels vllm-xpu-kernels-unstable
            vllm-xpu vllm-xpu-unstable;
          inherit (stableLibs)
            attn-kernels-xe-2
            gdn-attn-kernels-xe-2
            mqa-logits-kernels-xe-2
            grouped-gemm-xe-2
            grouped-gemm-xe-default;
          default = intel-oneapi;
          inherit quantize kl-eval lint;
        };

        apps = {
          autoround = {
            type = "app";
            program = "${auto-round-xpu}/bin/auto-round";
          };
          quantize = {
            type = "app";
            program = "${quantize}/bin/quantize";
          };
          kl-eval = {
            type = "app";
            program = "${kl-eval}/bin/kl-eval";
          };
          lint = {
            type = "app";
            program = "${lint}/bin/lint";
          };
        };

        devShells.default = pkgs.mkShell {
          name = "vllm-xpu-nix-dev";
          packages = with pkgs; [
            git
            nix-tree
            nix-diff
            nixfmt
            nil
            patchelf
            file
            skopeo
            pre-commit
            lint
          ];
          shellHook = ''
            cat <<'EOF'
            vllm-xpu-nix dev shell.

            Build packages:
              nix build .#intel-oneapi
              nix build .#torch-xpu
              nix build .#triton-xpu
              nix build .#vllm-xpu-kernels             # upstream vllm-project (stable)
              nix build .#vllm-xpu-kernels-unstable    # jasonboukheir fork (work-in-progress)
              nix build .#vllm-xpu                     # upstream vllm-project (stable)
              nix build .#vllm-xpu-unstable            # jasonboukheir fork (work-in-progress)

            Iterate against a local checkout (no flake edit needed):
              nix build .#vllm-xpu-unstable \
                --override-input vllm-xpu-unstable-src path:../vllm
              nix build .#vllm-xpu-kernels-unstable \
                --override-input vllm-xpu-kernels-unstable-src path:../vllm-xpu-kernels

            Lint local checkouts (pinned ruff matching each project's pins):
              lint                       # defaults to ../vllm and ../vllm-xpu-kernels
              lint /path/to/vllm /path/to/vllm-xpu-kernels

            Develop the kernels (editable install, edit/build/test in tree):
              cd /path/to/vllm-xpu-kernels
              nix develop /path/to/vllm-xpu-nix#kernels-dev
              pip install -e . --no-build-isolation
              pytest tests/                 # e.g. tests/gdn_attn for the GDN work

            Develop vllm (editable install, edit/run/test in tree):
              cd /path/to/vllm
              nix develop /path/to/vllm-xpu-nix#vllm-dev
              pip install -e . --no-build-isolation --no-deps
              vllm serve <model> --enforce-eager
            EOF
          '';
        };

        devShells.kernels-dev = pkgs.mkShell {
          name = "vllm-xpu-kernels-dev";
          inputsFrom = [ vllm-xpu-kernels ];
          packages = with pkgs; [
            cmake
            ninja
            git
          ] ++ (with pkgs.python312Packages; [
            pip
            setuptools
            wheel
            packaging
            jinja2
            psutil
            torch-xpu
            triton-xpu
          ]);
          shellHook = syclToolchainShellHook + ''
            export VLLM_XPU_AOT_DEVICES="''${VLLM_XPU_AOT_DEVICES:-bmg}"
            export VLLM_XPU_XE2_AOT_DEVICES="''${VLLM_XPU_XE2_AOT_DEVICES:-bmg}"
            export MAX_JOBS="''${MAX_JOBS:-2}"
            cat <<'EOF'
            vllm-xpu-nix kernels-dev shell.

            Toolchain + the full vllm-xpu-kernels closure (torch-xpu, triton-xpu,
            oneAPI MKL/SYCL, cutlass src) wired up. MAX_JOBS=2 by default — each
            SYCL-TLA template instantiation holds ~40 GiB during icpx, raise with
            care.

            Quick start (against a local kernels checkout):
              cd /path/to/vllm-xpu-kernels
              git submodule update --init --recursive
              pip install -e . --no-build-isolation

            Incremental rebuild after editing a .cpp:
              ninja -C build/temp.*/release install

            Tests:
              pytest tests/

            Tune feature flags via cmake -D… or env (BASIC_KERNELS_ENABLED,
            FA2_KERNELS_ENABLED, MOE_KERNELS_ENABLED, GDN_KERNELS_ENABLED, etc.).
            EOF
          '';
        };

        devShells.vllm-dev = pkgs.mkShell {
          name = "vllm-xpu-vllm-dev";
          inputsFrom = [ vllm-xpu ];
          packages = with pkgs; [
            git
          ] ++ (with pkgs.python312Packages; [
            pip
            setuptools
            wheel
            packaging
            jinja2
            torch-xpu
            triton-xpu
            vllm-xpu-kernels
          ]);
          shellHook = syclToolchainShellHook + ''
            export VLLM_TARGET_DEVICE=xpu
            export VLLM_VERSION_OVERRIDE="''${VLLM_VERSION_OVERRIDE:-0.0.0.dev}"

            # BMG single-card oneCCL safe defaults — match what the systemd
            # module bakes in. Override per-session to taste.
            export CCL_PROCESS_LAUNCHER="''${CCL_PROCESS_LAUNCHER:-none}"
            export CCL_ATL_TRANSPORT="''${CCL_ATL_TRANSPORT:-ofi}"
            export CCL_ZE_IPC_EXCHANGE="''${CCL_ZE_IPC_EXCHANGE:-sockets}"
            export CCL_LOG_LEVEL="''${CCL_LOG_LEVEL:-warn}"

            cat <<'EOF'
            vllm-xpu-nix vllm-dev shell.

            Toolchain + full vllm-xpu closure (torch-xpu, triton-xpu, vllm-xpu-
            kernels, runtime python deps) wired up. VLLM_TARGET_DEVICE=xpu and
            BMG single-card oneCCL env are pre-set.

            Editable install:
              cd /path/to/vllm
              pip install -e . --no-build-isolation --no-deps
              python -c 'import vllm; print(vllm.__version__)'

            Run server:
              vllm serve <model> --enforce-eager

            Tests:
              pytest tests/
            EOF
          '';
        };
      });
    in
    systemOutputs // {
      # System-independent outputs (NixOS modules, overlays).
      #
      # Two modules are exposed:
      #   - `nixosModules.vllm-xpu`: the pure option module. Reads
      #     `pkgs.vllm-xpu` etc., so the consumer must apply
      #     `overlays.default` themselves (or supply the package
      #     explicitly via `services.vllm-xpu.package`).
      #   - `nixosModules.default`: the batteries-included entry point.
      #     Applies `overlays.default` AND imports the option module, so
      #     consumers just `imports = [ inputs.vllm-xpu-nix.nixosModules.default ]`
      #     and `pkgs.vllm-xpu` / `pkgs.vllm-xpu-unstable` are visible
      #     without writing their own overlay.
      nixosModules.vllm-xpu = ./nix/modules/vllm-xpu.nix;
      nixosModules.default = { ... }: {
        imports = [ ./nix/modules/vllm-xpu.nix ];
        nixpkgs.overlays = [ self.overlays.default ];
      };

      # Overlay that injects the XPU package set into a host's pkgs.
      # Pair with the bare `nixosModules.vllm-xpu`, or just import
      # `nixosModules.default` which applies this for you.
      overlays.default = _final: prev:
        let
          pkgs = systemOutputs.packages.${prev.stdenv.hostPlatform.system} or { };
          pick = name: lib.optionalAttrs (pkgs ? ${name}) { ${name} = pkgs.${name}; };
          inherit (nixpkgs) lib;
        in
        pick "torch-xpu"
        // pick "triton-xpu"
        // pick "intel-pti"
        // pick "flash-linear-attention"
        // pick "auto-round-xpu"
        // pick "vllm-xpu-kernels"
        // pick "vllm-xpu-kernels-unstable"
        // pick "vllm-xpu"
        // pick "vllm-xpu-unstable";
    };
}
