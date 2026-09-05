{
  description = "Nix-native Intel XPU substrate for vLLM (torch+xpu, triton-xpu, vllm-xpu-kernels, vllm)";

  inputs = {
    # Keep the entire XPU build substrate reproducible for downstream users.
    # A revision in flake.lock alone can be refreshed by a consumer's broad
    # `nix flake update`, even when the vllm-xpu-nix source input itself does
    # not move. Pinning the revision in the input URL makes nixpkgs updates an
    # explicit change in this repository, preventing surprise torch rebuilds.
    nixpkgs.url = "github:NixOS/nixpkgs/0e251e24a4f24e036a084b6b4b2d2491af4167f4";
    flake-utils.url = "github:numtide/flake-utils";

    vllm-xpu-kernels-src = {
      type = "git";
      url = "https://github.com/vllm-project/vllm-xpu-kernels.git";
      ref = "release/v0.1.11";
      submodules = true;
      flake = false;
    };

    vllm-xpu-kernels-unstable-src = {
      type = "git";
      url = "ssh://forgejo@git.sunnycareboo.com:2222/jasonbk/vllm-xpu-kernels.git";
      ref = "refs/heads/experimental/kvarn-factory-round5-integration";
      submodules = true;
      flake = false;
    };

    vllm-xpu-src = {
      type = "git";
      url = "https://github.com/vllm-project/vllm.git";
      ref = "refs/heads/releases/v0.25.0";
      flake = false;
    };

    vllm-xpu-unstable-src = {
      type = "git";
      url = "ssh://forgejo@git.sunnycareboo.com:2222/jasonbk/vllm.git";
      ref = "refs/heads/experimental/kvarn-factory-round5-integration";
      flake = false;
    };

    sycl-tla-src = {
      url = "github:intel/sycl-tla/87f6850680a580654b9ea2c80dbc01aeb36ad231";
      flake = false;
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      vllm-xpu-kernels-src,
      vllm-xpu-kernels-unstable-src,
      vllm-xpu-src,
      vllm-xpu-unstable-src,
      sycl-tla-src,
    }:
    let
      systemOutputs = flake-utils.lib.eachSystem [ "x86_64-linux" ] (
        system:
        let
          pkgs = import nixpkgs {
            inherit system;
            config.allowUnfree = true;
            overlays = [ (import ./nix/python-test-workarounds-overlay.nix) ];
          };
          sourceRevision =
            source:
            if source ? rev then
              source.rev
            else if source ? dirtyRev then
              source.dirtyRev
            else
              "dirty";

          # ---- source narrowing + version stamping ----
          mkKernelsSrc = import ./nix/lib/kernels-src.nix { inherit (pkgs) lib; };
          vllm-xpu-kernels-src' = mkKernelsSrc vllm-xpu-kernels-src;
          vllm-xpu-kernels-unstable-src' = mkKernelsSrc vllm-xpu-kernels-unstable-src;

          mkInputVersion = import ./nix/lib/mk-input-version.nix { lockFile = ./flake.lock; };
          kernelsStableVersion = mkInputVersion {
            name = "vllm-xpu-kernels-src";
            input = vllm-xpu-kernels-src;
          };
          kernelsUnstableVersion = mkInputVersion {
            name = "vllm-xpu-kernels-unstable-src";
            input = vllm-xpu-kernels-unstable-src;
            # main descends from the 0.1.14.1/v0.1.14 tag commit; retain the
            # most precise upstream release line before the snapshot suffix.
            base = "0.1.14.1";
            unstable = true;
          };
          vllmStableVersion = mkInputVersion {
            name = "vllm-xpu-src";
            input = vllm-xpu-src;
          };
          vllmUnstableVersion = mkInputVersion {
            name = "vllm-xpu-unstable-src";
            input = vllm-xpu-unstable-src;
            # vLLM cuts final releases on release branches, so the tag is not
            # an ancestor of main. Track the latest final release line rather
            # than leaving post-v0.28 main snapshots labelled as v0.26.
            base = "0.28.0";
            unstable = true;
          };

          # ---- toolchain + base substrate ----
          intel-oneapi = import ./nix/intel-oneapi.nix { inherit pkgs; };

          intel-pti = pkgs.callPackage ./nix/intel-pti.nix {
            intel-oneapi-base = intel-oneapi;
          };

          triton-xpu = pkgs.callPackage ./nix/triton-xpu.nix {
            intel-oneapi-base = intel-oneapi;
            inherit intel-pti;
            python3Packages = pkgs.python312Packages;
          };

          torch-xpu = pkgs.callPackage ./nix/torch-xpu.nix {
            intel-oneapi-base = intel-oneapi;
            inherit intel-pti triton-xpu;
            python3Packages = pkgs.python312Packages;
          };

          torchvision-xpu = pkgs.callPackage ./nix/torchvision-xpu.nix {
            inherit torch-xpu;
            python3Packages = pkgs.python312Packages;
          };

          python312PackagesXpu = import ./nix/python-xpu.nix {
            inherit
              pkgs
              torch-xpu
              triton-xpu
              torchvision-xpu
              ;
          };

          flash-linear-attention = pkgs.callPackage ./nix/flash-linear-attention.nix {
            inherit torch-xpu triton-xpu;
            python3Packages = python312PackagesXpu;
          };

          auto-round-xpu = pkgs.callPackage ./nix/auto-round-xpu.nix {
            inherit torch-xpu triton-xpu flash-linear-attention;
            python3Packages = python312PackagesXpu;
          };

          llm-compressor-xpu = pkgs.callPackage ./nix/llm-compressor-xpu.nix {
            inherit torch-xpu auto-round-xpu;
            python3Packages = python312PackagesXpu;
          };

          quantize = pkgs.callPackage ./nix/quantize.nix {
            inherit auto-round-xpu llm-compressor-xpu;
            intel-oneapi-base = intel-oneapi;
            python3Packages = python312PackagesXpu;
          };

          kl-eval = pkgs.callPackage ./nix/kl-eval.nix {
            inherit auto-round-xpu;
            python3Packages = python312PackagesXpu;
          };

          # ---- kernel + vllm build factories ----
          inherit
            (import ./nix/mk-kernels.nix {
              inherit
                pkgs
                intel-oneapi
                intel-pti
                torch-xpu
                ;
              cutlass-src = sycl-tla-src;
            })
            mkKernelLibs
            mkVllmXpuKernels
            ;

          stableLibs = mkKernelLibs {
            src = vllm-xpu-kernels-src';
            version = kernelsStableVersion;
            sourceRevision = sourceRevision vllm-xpu-kernels-src;
          };
          unstableLibs = mkKernelLibs {
            src = vllm-xpu-kernels-unstable-src';
            version = kernelsUnstableVersion;
            sourceRevision = sourceRevision vllm-xpu-kernels-unstable-src;
          };

          vllm-xpu-kernels = mkVllmXpuKernels {
            src = vllm-xpu-kernels-src';
            version = kernelsStableVersion;
            sourceRevision = sourceRevision vllm-xpu-kernels-src;
          };
          vllm-xpu-kernels-unstable = mkVllmXpuKernels {
            src = vllm-xpu-kernels-unstable-src';
            version = kernelsUnstableVersion;
            sourceRevision = sourceRevision vllm-xpu-kernels-unstable-src;
          };

          mkVllm = import ./nix/mk-vllm.nix {
            inherit
              pkgs
              intel-oneapi
              intel-pti
              torch-xpu
              triton-xpu
              flash-linear-attention
              ;
            python3Packages = python312PackagesXpu;
          };

          vllm-xpu = mkVllm {
            src = vllm-xpu-src;
            version = vllmStableVersion;
            kernels = vllm-xpu-kernels;
          };

          vllm-xpu-unstable = mkVllm {
            src = vllm-xpu-unstable-src;
            version = vllmUnstableVersion;
            kernels = vllm-xpu-kernels-unstable;
            mcpPackage = python312PackagesXpu.mcp-v2;
          };

          # Kvarn optimization factory for Brutus's frozen Qwen3.8 profile.
          # The static attention sources contain every runtime-selectable
          # Kvarn candidate; these config files narrow only upstream's
          # generated FA2 Cartesian sweep.  Keep the two arms in the same
          # package so auto and Kvarn cannot accidentally benchmark different
          # builds.
          kvarnFactoryKernelConfig = {
            chunkPrefill = builtins.toFile "brutus-kvarn-chunk-prefill.conf" ''
              # auto paged prompt/chunk processing
              256,true,true,false,false,false

              # Kvarn raw first prefill and materialized cached continuation
              256,false,true,false,false,false
            '';
            pagedDecode = builtins.toFile "brutus-auto-paged-decode.conf" ''
              # Hq24/Hkv4 -> qgroup 8; qlen=1 XPU decode is non-causal
              8,256,64,false,false,false
            '';
          };

          vllm-xpu-kvarn-factory = vllm-xpu-unstable.override {
            withTorchvision = true;
            aotDevices = [ "bmg" ];
            kernelConfig = kvarnFactoryKernelConfig;
          };
          kvarnFactoryAttentionLibrary =
            vllm-xpu-kvarn-factory.kernelPackage.kernelLibraries.attn-kernels-xe-2;
          kvarnFactoryAttentionSourceProvenance = kvarnFactoryAttentionLibrary.kernelSourceProvenance;

          # Run the entire B70 candidate matrix from the same Python closure
          # as the factory package.  The host script receives the embedded
          # package as its positional artifact, so the environment and the
          # attested shared objects cannot silently come from different
          # builds.
          kvarnFactoryPython = pkgs.python312.withPackages (_: [
            vllm-xpu-kvarn-factory
            pkgs.python312Packages.pytest
          ]);

          # Service and correctness harnesses need both the vLLM executable
          # and a Python interpreter from one immutable candidate.  The
          # withPackages interpreter above has the complete Python closure,
          # but is not wrapped with vLLM's pinned Level Zero, compute-runtime,
          # IGC, oneAPI, compiler, and JIT-linker environment.  Preserve the
          # package's existing bin/vllm wrapper and apply those same runtime
          # arguments to Python.  PYTHONPATH is the sole exception: the
          # withPackages environment already owns its complete module path,
          # while vLLM's argument contains a package-output placeholder.
          kvarnFactoryRuntimeWrapperArgs = builtins.filter (
            arg: !(pkgs.lib.hasPrefix "--prefix PYTHONPATH " arg)
          ) vllm-xpu-kvarn-factory.makeWrapperArgs;
          vllm-xpu-kvarn-factory-env = pkgs.symlinkJoin {
            name = "vllm-xpu-kvarn-factory-env";
            paths = [ kvarnFactoryPython ];
            nativeBuildInputs = [ pkgs.makeWrapper ];
            postBuild = ''
              rm -f "$out/bin/python" "$out/bin/python3" "$out/bin/python3.12"
              makeWrapper \
                ${kvarnFactoryPython}/bin/python \
                "$out/bin/python" \
                ${builtins.concatStringsSep " " kvarnFactoryRuntimeWrapperArgs}
              ln -s python "$out/bin/python3"
              ln -s python "$out/bin/python3.12"
            '';
          };
          kvarn-factory-host = pkgs.writeShellApplication {
            name = "kvarn-factory-host";
            runtimeInputs = [
              pkgs.gitMinimal
              pkgs.nix
            ];
            text = ''
              # Factory selectors and model-service settings must not leak
              # into the direct primitive runner.  The runner passes every
              # candidate selector explicitly to the native operators.
              for variable in ''${!KVARN_@} ''${!VLLM_@}; do
                unset "$variable"
              done
              unset \
                PYTHONHOME PYTHONPATH PYTHONSTARTUP PYTHONUSERBASE \
                PYTEST_ADDOPTS PYTEST_PLUGINS LD_AUDIT LD_PRELOAD \
                LD_LIBRARY_PATH LIBRARY_PATH CC CXX CMPLR_ROOT \
                LEVEL_ZERO_V1_SDK_PATH ONEAPI_ROOT SYCL_HOME \
                ONEAPI_DEVICE_SELECTOR SYCL_DEVICE_FILTER ZE_AFFINITY_MASK \
                CUDA_VISIBLE_DEVICES

              export PYTHONNOUSERSITE=1
              export PYTHONDONTWRITEBYTECODE=1
              export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
              export ONEAPI_ROOT=${intel-oneapi}
              export SYCL_HOME=${intel-oneapi}/compiler/latest
              export CMPLR_ROOT=${intel-oneapi}/compiler/latest
              export LEVEL_ZERO_V1_SDK_PATH=${pkgs.level-zero}
              export LIBRARY_PATH=${pkgs.level-zero}/lib
              export LD_LIBRARY_PATH=${
                pkgs.lib.makeLibraryPath [
                  pkgs.level-zero
                  pkgs.intel-graphics-compiler
                  pkgs.intel-compute-runtime
                  pkgs.intel-compute-runtime.drivers
                ]
              }
              export CC=${pkgs.stdenv.cc}/bin/cc
              export CXX=${pkgs.stdenv.cc}/bin/c++
              export PATH=${
                pkgs.lib.makeBinPath [
                  pkgs.gitMinimal
                  pkgs.nix
                  pkgs.intel-compute-runtime
                ]
              }

              exec ${kvarnFactoryPython}/bin/python \
                ${./scripts/kvarn_factory_host.py} \
                ${vllm-xpu-kvarn-factory} \
                ${sourceRevision self} \
                ${sourceRevision vllm-xpu-unstable-src} \
                ${sourceRevision vllm-xpu-kernels-unstable-src} \
                "$@" \
                --expected-native-attention-output \
                ${kvarnFactoryAttentionLibrary} \
                --native-attention-source-scheme \
                ${pkgs.lib.escapeShellArg kvarnFactoryAttentionSourceProvenance.artifactIdentity.scheme} \
                --native-attention-source-store-hash \
                ${pkgs.lib.escapeShellArg kvarnFactoryAttentionSourceProvenance.artifactIdentity.filteredSourceStoreHash} \
                --native-attention-compatible-revision \
                ${pkgs.lib.escapeShellArg kvarnFactoryAttentionSourceProvenance.compatibilityProvenance.upstreamRevision}
            '';
          };

          kv-kernel-ab = pkgs.callPackage ./nix/kv-kernel-ab.nix { };

          # ---- shells, checks, and misc helpers ----
          hfMetadata = pkgs.callPackage ./nix/hf-metadata.nix { };

          mkQuantizationWorkspace = import ./nix/quantization-workspace.nix {
            inherit pkgs quantize;
          };

          lint = import ./nix/lint.nix { inherit pkgs; };

          testPython = pkgs.python312.withPackages (_: [
            llm-compressor-xpu
            pkgs.python312Packages.huggingface-hub
            pkgs.python312Packages.pytest
          ]);
        in
        {
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
              unionKernelSet
              ;
            # Parameterized builders, so a consumer can build a vllm (or the
            # kernels) from an arbitrary source — e.g. a local submodule
            # checkout — without overriding flake inputs. The packages.*
            # `vllm-xpu`/`vllm-xpu-unstable` outputs are just two fixed
            # instantiations of mkVllm; mkVllm { src; version; kernels; ... }
            # builds against the same pinned substrate from any src.
            inherit
              mkVllm
              mkVllmXpuKernels
              mkQuantizationWorkspace
              ;
          };

          packages = {
            inherit
              intel-oneapi
              intel-pti
              torch-xpu
              triton-xpu
              torchvision-xpu
              flash-linear-attention
              auto-round-xpu
              llm-compressor-xpu
              vllm-xpu-kernels
              vllm-xpu-kernels-unstable
              vllm-xpu
              vllm-xpu-unstable
              vllm-xpu-kvarn-factory
              vllm-xpu-kvarn-factory-env
              kvarn-factory-host
              ;
            inherit (stableLibs)
              attn-kernels-xe-2
              gdn-attn-kernels-xe-2
              mhc-kernels-xe-2
              mqa-logits-kernels-xe-2
              grouped-gemm-xe-2
              grouped-gemm-xe-default
              ;
            default = intel-oneapi;
            inherit
              quantize
              kl-eval
              kv-kernel-ab
              lint
              ;
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
            kv-kernel-ab = {
              type = "app";
              program = "${kv-kernel-ab}/bin/kv-kernel-ab";
            };
            kvarn-factory = {
              type = "app";
              program = "${kvarn-factory-host}/bin/kvarn-factory-host";
            };
            lint = {
              type = "app";
              program = "${lint}/bin/lint";
            };
          };

          checks = {
            kernel-glue-cache-identity = import ./nix/tests/kernel-glue-cache-identity.nix {
              inherit pkgs;
              kernelSource = vllm-xpu-kernels-unstable-src';
            };

            kernel-lib-cache-identity = import ./nix/tests/kernel-lib-cache-identity.nix {
              inherit pkgs;
            };

            formatting =
              pkgs.runCommand "vllm-xpu-nix-formatting"
                {
                  nativeBuildInputs = [ pkgs.nixfmt ];
                }
                ''
                  cd ${self}
                  # The historical Nix tree is not uniformly nixfmt-clean
                  # yet. Expand this list as the remaining files are
                  # normalized.
                  nixfmt --check \
                    flake.nix \
                    nix/devshells.nix \
                    nix/lib/kernel-glue-src.nix \
                    nix/mk-kernels.nix \
                    nix/tests/kernel-glue-cache-identity.nix \
                    nix/vllm-xpu-kernels-compose.nix \
                    nix/vllm-xpu-kernels.nix
                  touch $out
                '';

            unit-tests =
              pkgs.runCommand "vllm-xpu-nix-unit-tests"
                {
                  nativeBuildInputs = [
                    testPython
                    pkgs.gitMinimal
                  ];
                }
                ''
                  cd ${self}
                  export HOME=$TMPDIR
                  # Run every file through pytest so pytest-style functions
                  # are collected. Keep files in separate processes because
                  # quantization modules use global plugin registries.
                  for test_file in tests/test_*.py; do
                    python -m pytest "$test_file" -v -p no:cacheprovider
                  done
                  touch $out
                '';
          };

          devShells = import ./nix/devshells.nix {
            inherit pkgs lint;
          };
        }
      );
    in
    systemOutputs
    // {
      # System-independent outputs (NixOS modules, overlays).
      #
      # Three modules are exposed:
      #   - `nixosModules.hf-cache`: generation-aware cache roots and GC for
      #     Hugging Face models, datasets, and Spaces.
      #   - `nixosModules.vllm-xpu`: the pure option module. Reads
      #     `pkgs.vllm-xpu` etc., so the consumer must apply
      #     `overlays.default` themselves (or supply the package
      #     explicitly via `services.vllm-xpu.package`).
      #   - `nixosModules.default`: the batteries-included entry point.
      #     Applies `overlays.default` AND imports the option module, so
      #     consumers just `imports = [ inputs.vllm-xpu-nix.nixosModules.default ]`
      #     and `pkgs.vllm-xpu` / `pkgs.vllm-xpu-unstable` are visible
      #     without writing their own overlay.
      nixosModules.hf-cache = ./nix/modules/hf-cache.nix;
      nixosModules.vllm-xpu = ./nix/modules/vllm-xpu.nix;
      nixosModules.default = { ... }: {
        imports = [ ./nix/modules/vllm-xpu.nix ];
        nixpkgs.overlays = [ self.overlays.default ];
      };

      # Overlay that injects the XPU package set into a host's pkgs.
      # Pair with the bare `nixosModules.vllm-xpu`, or just import
      # `nixosModules.default` which applies this for you.
      overlays.default =
        _final: prev:
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
        // pick "vllm-xpu-unstable"
        // pick "vllm-xpu-kvarn-factory"
        // pick "vllm-xpu-kvarn-factory-env";
    };
}
