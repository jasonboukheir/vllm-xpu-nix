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

        mkXpuLibFactory = src: pkgs.callPackage ./nix/vllm-xpu-lib.nix {
          intel-oneapi-base = intel-oneapi;
          inherit intel-pti oneccl-bmg torch-xpu;
          python3Packages = pkgs.python312Packages;
          inherit src;
          cutlass-src = sycl-tla-src;
        };

        # Dynamic-derivations build of attn_kernels_xe_2: per-TU compile
        # drvs + a final link drv replaying cmake's captured link command.
        # See nix/vllm-xpu-attn-dyndrv.nix for the staging.
        mkAttnDynDrv = src: pkgs.callPackage ./nix/vllm-xpu-attn-dyndrv.nix {
          intel-oneapi-base = intel-oneapi;
          inherit intel-pti oneccl-bmg torch-xpu;
          python3Packages = pkgs.python312Packages;
          inherit src;
          cutlass-src = sycl-tla-src;
        };

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

        mkKernelLibs = src:
          let mkLib = mkXpuLibFactory src; in {
            # attn_kernels_xe_2 is built via the dynamic-derivations path
            # (per-TU compile drvs + replayed cmake link). All other libs
            # still go through the cmake-builds-the-whole-target mkLib.
            attn-kernels-xe-2 = mkAttnDynDrv src;
            gdn-attn-kernels-xe-2 = mkLib { libName = "gdn_attn_kernels_xe_2"; featureFlags = gdnAttnFlags; };
            mqa-logits-kernels-xe-2 = mkLib { libName = "mqa_logits_kernels_xe_2"; featureFlags = mqaLogitsFlags; };
            grouped-gemm-xe-2 = mkLib { libName = "grouped_gemm_xe_2"; featureFlags = groupedGemmXe2Flags; };
            grouped-gemm-xe-default = mkLib { libName = "grouped_gemm_xe_default"; featureFlags = groupedGemmXeDefaultFlags; };
          };

        mkVllmXpuKernels = src:
          let libs = mkKernelLibs src; in
          pkgs.callPackage ./nix/vllm-xpu-kernels.nix ({
            intel-oneapi-base = intel-oneapi;
            inherit intel-pti oneccl-bmg torch-xpu;
            python3Packages = pkgs.python312Packages;
            inherit src;
            cutlass-src = sycl-tla-src;
          } // libs);

        stableLibs = mkKernelLibs vllm-xpu-kernels-src;
        unstableLibs = mkKernelLibs vllm-xpu-kernels-unstable-src;

        vllm-xpu-kernels = mkVllmXpuKernels vllm-xpu-kernels-src;
        vllm-xpu-kernels-unstable = mkVllmXpuKernels vllm-xpu-kernels-unstable-src;

        # mkVllm pairs a vllm source pin with the matching kernels build:
        # the upstream stable variant gets vllm-xpu-kernels (vllm-project),
        # the unstable variant gets vllm-xpu-kernels-unstable (jasonboukheir
        # fork). Pre-release version stamp is fine — VLLM_VERSION_OVERRIDE
        # in vllm-xpu.nix forwards to setuptools-scm's PRETEND_VERSION, so
        # setuptools-scm doesn't need a .git in the unpacked store path.
        mkVllm = { src, version, kernels }: pkgs.callPackage ./nix/vllm-xpu.nix {
          intel-oneapi-base = intel-oneapi;
          inherit intel-pti oneccl-bmg torch-xpu triton-xpu flash-linear-attention;
          python3Packages = python312PackagesXpu;
          vllm-xpu-kernels = kernels;
          inherit src version;
        };

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
      in {
        packages = {
          inherit
            intel-oneapi intel-pti oneccl-bmg
            torch-xpu triton-xpu
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
      nixosModules.vllm-xpu = ./nix/modules/vllm-xpu.nix;

      # Overlay that injects the XPU package set into a host's pkgs. Pair with
      # the nixosModules.vllm-xpu module so `services.vllm-xpu.package` defaults
      # to `pkgs.vllm-xpu`.
      overlays.default = final: _prev:
        let
          pkgs = systemOutputs.packages.${final.system} or { };
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
