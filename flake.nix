{
  description = "Nix-native Intel XPU substrate for vLLM (torch+xpu, triton-xpu, vllm-xpu-kernels, vllm)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";

    vllm-xpu-kernels-src = {
      type = "git";
      url = "https://github.com/vllm-project/vllm-xpu-kernels.git";
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

        intel-oneapi = pkgs.intel-oneapi.base.override {
          components = [
            "intel.oneapi.lin.dpcpp-cpp-compiler"
            "intel.oneapi.lin.mkl.devel"
            "intel.oneapi.lin.dpl"
          ];
        };

        intel-pti = pkgs.callPackage ./nix/intel-pti.nix {
          intel-oneapi-base = intel-oneapi;
        };

        oneccl-bmg = pkgs.callPackage ./nix/oneccl-bmg.nix {
          intel-oneapi-base = intel-oneapi;
        };

        torch-xpu = pkgs.callPackage ./nix/torch-xpu.nix {
          intel-oneapi-base = intel-oneapi;
          inherit oneccl-bmg intel-pti;
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

        torchaudio-xpu = pkgs.callPackage ./nix/torchaudio-xpu.nix {
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
          torchaudio = torchaudio-xpu;
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

        mkXpuLibFactory = { src, aotDevices ? [ ], useCcache ? true }:
          let factory = pkgs.callPackage ./nix/vllm-xpu-lib.nix {
            intel-oneapi-base = intel-oneapi;
            inherit intel-pti oneccl-bmg torch-xpu;
            python3Packages = pkgs.python312Packages;
            inherit src;
            cutlass-src = sycl-tla-src;
          };
          in { libName, featureFlags ? [ ] }:
            factory { inherit libName featureFlags aotDevices useCcache; };

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

        mkKernelLibs = { src, aotDevices ? [ ], useCcache ? true }:
          let mkLib = mkXpuLibFactory { inherit src aotDevices useCcache; }; in {
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
        mkVllmXpuKernels = { src, aotDevices ? [ ], useCcache ? true }:
          let
            libs = mkKernelLibs { inherit src aotDevices useCcache; };
            base = pkgs.callPackage ./nix/vllm-xpu-kernels.nix ({
              intel-oneapi-base = intel-oneapi;
              inherit intel-pti oneccl-bmg torch-xpu useCcache;
              python3Packages = pkgs.python312Packages;
              inherit src aotDevices;
              cutlass-src = sycl-tla-src;
            } // libs);
          in
            base.overrideAttrs (old: {
              passthru = (old.passthru or {}) // {
                withAotDevices = ds: mkVllmXpuKernels {
                  inherit src useCcache; aotDevices = ds;
                };
                withJIT = mkVllmXpuKernels {
                  inherit src useCcache; aotDevices = [];
                };
                withAOT = mkVllmXpuKernels {
                  inherit src useCcache; aotDevices = [ "bmg" ];
                };
                withCcache = b: mkVllmXpuKernels {
                  inherit src aotDevices; useCcache = b;
                };
              };
            });

        stableLibs = mkKernelLibs { src = vllm-xpu-kernels-src'; };
        unstableLibs = mkKernelLibs { src = vllm-xpu-kernels-unstable-src'; };

        vllm-xpu-kernels = mkVllmXpuKernels { src = vllm-xpu-kernels-src'; };
        vllm-xpu-kernels-unstable = mkVllmXpuKernels { src = vllm-xpu-kernels-unstable-src'; };

        # mkVllm pairs a vllm source pin with the matching kernels build:
        # the upstream stable variant gets vllm-xpu-kernels (vllm-project),
        # the unstable variant gets vllm-xpu-kernels-unstable (jasonboukheir
        # fork). Pre-release version stamp is fine — VLLM_VERSION_OVERRIDE
        # in vllm-xpu.nix forwards to setuptools-scm's PRETEND_VERSION, so
        # setuptools-scm doesn't need a .git in the unpacked store path.
        #
        # Like mkVllmXpuKernels, the result exposes `withAotDevices` /
        # `withJIT` / `withAOT` passthrus that cascade through the kernels
        # package. Also exposes `withTorchvision`, `withTorchaudio`, and
        # `withAudio` passthrus so consumers can opt into the +xpu wheels
        # for VL / audio model families (or audio decoders for
        # transcription endpoints) without spelling out a full `.override`.
        # All passthrus compose:
        # `pkgs.vllm-xpu-unstable.withAOT |> .withTorchvision true |> .withAudio true`.
        mkVllm = {
          src, version, kernels,
          withTorchvision ? false,
          withTorchaudio ? false,
          withAudio ? false,
        }:
          let
            base = pkgs.callPackage ./nix/vllm-xpu.nix {
              intel-oneapi-base = intel-oneapi;
              inherit intel-pti oneccl-bmg torch-xpu triton-xpu flash-linear-attention;
              python3Packages = python312PackagesXpu;
              vllm-xpu-kernels = kernels;
              inherit src version withTorchvision withTorchaudio withAudio;
              inherit (pkgs) level-zero intel-graphics-compiler intel-compute-runtime;
            };
          in
            base.overrideAttrs (old: {
              passthru = (old.passthru or {}) // {
                withAotDevices = ds: mkVllm {
                  inherit src version withTorchvision withTorchaudio withAudio;
                  kernels =
                    if kernels ? withAotDevices
                    then kernels.withAotDevices ds
                    else kernels;
                };
                withJIT = mkVllm {
                  inherit src version withTorchvision withTorchaudio withAudio;
                  kernels =
                    if kernels ? withJIT
                    then kernels.withJIT
                    else kernels;
                };
                withAOT = mkVllm {
                  inherit src version withTorchvision withTorchaudio withAudio;
                  kernels =
                    if kernels ? withAOT
                    then kernels.withAOT
                    else kernels;
                };
                withTorchvision = b: mkVllm {
                  inherit src version kernels withTorchaudio withAudio;
                  withTorchvision = b;
                };
                withTorchaudio = b: mkVllm {
                  inherit src version kernels withTorchvision withAudio;
                  withTorchaudio = b;
                };
                withAudio = b: mkVllm {
                  inherit src version kernels withTorchvision withTorchaudio;
                  withAudio = b;
                };
              };
            });

        vllm-xpu = mkVllm {
          src = vllm-xpu-src;
          version = "0.20.2.dev";
          kernels = vllm-xpu-kernels;
        };

        vllm-xpu-unstable = mkVllm {
          src = vllm-xpu-unstable-src;
          version = "0.20.2.dev0+xpu.unstable";
          kernels = vllm-xpu-kernels-unstable;
        };

        syclToolchainShellHook = ''
          syclHome="${intel-oneapi}/compiler/latest"
          mkdir -p .dev-bin
          ln -sf ${pkgs.intel-compute-runtime}/bin/ocloc-* .dev-bin/ocloc 2>/dev/null || true
          export PATH="$PWD/.dev-bin:$syclHome/bin:$PATH"
          export LD_LIBRARY_PATH="${pkgs.intel-graphics-compiler}/lib:${pkgs.intel-compute-runtime}/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
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
      in {
        # Per-system helpers consumers reach via
        # `inputs.vllm-xpu-nix.lib.${pkgs.system}.fromHfConfig`.
        # `lib.${system}` (rather than just `lib`) is intentional —
        # the helpers wrap `pkgs.fetchurl`, which is system-scoped.
        lib = {
          inherit (hfMetadata)
            fetchHfConfig
            readHfConfig
            attnParamsFromConfig
            fromHfConfig
            unionKernelSet;
        };

        packages = {
          inherit
            intel-oneapi intel-pti oneccl-bmg
            torch-xpu triton-xpu torchvision-xpu torchaudio-xpu
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
          inherit quantize kl-eval;
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

            Iterate against a local kernels checkout (no flake edit needed):
              nix build .#vllm-xpu-kernels-unstable \
                --override-input vllm-xpu-kernels-unstable-src path:/path/to/local/checkout

            Fast in-tree kernel iteration (impure, ~10s incrementals):
              nix develop .#attn-dev
              make dev-attn KERNELS_SRC=/path/to/vllm-xpu-kernels

            Editable install of all kernels:
              cd /path/to/vllm-xpu-kernels
              nix develop /path/to/vllm-xpu-nix#kernels-dev
              pip install -e . --no-build-isolation
            EOF
          '';
        };

        devShells.attn-dev = pkgs.mkShell {
          name = "vllm-xpu-attn-dev";
          inputsFrom = [ stableLibs.attn-kernels-xe-2 ];
          packages = with pkgs; [
            cmake
            ninja
            git
          ];
          shellHook = syclToolchainShellHook + ''
            cat <<'EOF'
            vllm-xpu-nix attn-dev shell.

            Toolchain: icpx, cmake, ninja, oneAPI MKL/SYCL, cutlass src all set up.

            Quick start:
              make dev-attn KERNELS_SRC=/path/to/vllm-xpu-kernels
              export VLLM_XPU_DEV_LIB_DIR=$PWD/build-dev/csrc/xpu/attn/xe_2
              python -c 'import vllm_xpu_kernels'

            Edit any kernel .cpp/.hpp and rerun 'make dev-attn' for incremental rebuild.
            Tear down with 'make dev-attn-clean'.
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
        // pick "oneccl-bmg"
        // pick "flash-linear-attention"
        // pick "auto-round-xpu"
        // pick "vllm-xpu-kernels"
        // pick "vllm-xpu-kernels-unstable"
        // pick "vllm-xpu"
        // pick "vllm-xpu-unstable";
    };
}
