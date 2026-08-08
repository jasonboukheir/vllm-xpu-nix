{
  description = "Nix-native Intel XPU substrate for vLLM (torch+xpu, triton-xpu, vllm-xpu-kernels, vllm)";

  inputs = {
    # Keep the entire XPU build substrate reproducible for downstream users.
    # A revision in flake.lock alone can be refreshed by a consumer's broad
    # `nix flake update`, even when the vllm-xpu-nix source input itself does
    # not move. Pinning the revision in the input URL makes nixpkgs updates an
    # explicit change in this repository, preventing surprise torch rebuilds.
    nixpkgs.url = "github:NixOS/nixpkgs/b7c2ada94fe99c15b0dbcf4d11fd7850b957a436";
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
      url = "https://git.sunnycareboo.com/jasonbk/vllm-xpu-kernels.git";
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
      url = "https://git.sunnycareboo.com/jasonbk/vllm.git";
      ref = "refs/heads/experimental/kvarn-xpu-46812";
      flake = false;
    };

    sycl-tla-src = {
      url = "github:intel/sycl-tla/cd763790ad2f74d7294435ecf77682bac0062c3a";
      flake = false;
    };
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
    vllm-xpu-kernels-src,
    vllm-xpu-kernels-unstable-src,
    vllm-xpu-src,
    vllm-xpu-unstable-src,
    sycl-tla-src,
  }: let
    systemOutputs = flake-utils.lib.eachSystem ["x86_64-linux"] (system: let
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfree = true;
        overlays = [(import ./nix/python-test-workarounds-overlay.nix)];
      };

      # ---- source narrowing + version stamping ----
      mkKernelsSrc = import ./nix/lib/kernels-src.nix {inherit (pkgs) lib;};
      vllm-xpu-kernels-src' = mkKernelsSrc vllm-xpu-kernels-src;
      vllm-xpu-kernels-unstable-src' = mkKernelsSrc vllm-xpu-kernels-unstable-src;

      mkInputVersion = import ./nix/lib/mk-input-version.nix {lockFile = ./flake.lock;};
      kernelsStableVersion = mkInputVersion {
        name = "vllm-xpu-kernels-src";
        input = vllm-xpu-kernels-src;
      };
      kernelsUnstableVersion = mkInputVersion {
        name = "vllm-xpu-kernels-unstable-src";
        input = vllm-xpu-kernels-unstable-src;
        # main descends from the v0.1.12 tag (git describe -> v0.1.12-N);
        # the +unstable.<date>.g<rev> suffix marks the snapshot ahead of it.
        base = "0.1.12";
        unstable = true;
      };
      vllmStableVersion = mkInputVersion {
        name = "vllm-xpu-src";
        input = vllm-xpu-src;
      };
      vllmUnstableVersion = mkInputVersion {
        name = "vllm-xpu-unstable-src";
        input = vllm-xpu-unstable-src;
        base = "0.26.0";
        unstable = true;
      };

      # ---- toolchain + base substrate ----
      intel-oneapi = import ./nix/intel-oneapi.nix {inherit pkgs;};

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

      python312PackagesXpu = import ./nix/python-xpu.nix {
        inherit pkgs torch-xpu torchvision-xpu;
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

      # ---- kernel + vllm build factories ----
      inherit
        (import ./nix/mk-kernels.nix {
          inherit pkgs intel-oneapi intel-pti torch-xpu;
          cutlass-src = sycl-tla-src;
        })
        mkKernelLibs
        mkVllmXpuKernels
        ;

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

      mkVllm = import ./nix/mk-vllm.nix {
        inherit pkgs intel-oneapi intel-pti torch-xpu triton-xpu flash-linear-attention;
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
      };

      # Exact package deployed by Brutus's chat + embedding services. Keep the
      # host-facing build policy here so `nix build .#vllm-xpu-chat` and the
      # NixOS module consumer resolve to one derivation and one store output.
      # The kernel set covers Qwen3.6-27B's head_dim=256 full attention plus
      # the bidirectional head_dim=64 Jina embedding encoder.
      vllm-xpu-chat = vllm-xpu-unstable.override {
        withTorchvision = true;
        aotDevices = ["bmg"];
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

      chatProfile = import ./nix/chat-profile.nix;

      mkMaintenanceServeArgs = port: inst:
        pkgs.lib.concatStringsSep " " (
          [
            (pkgs.lib.escapeShellArg inst.model)
            "--host" "127.0.0.1"
            "--port" (toString port)
            "--served-model-name" (pkgs.lib.escapeShellArg inst.servedName)
            "--dtype" (pkgs.lib.escapeShellArg inst.dtype)
            "--gpu-memory-utilization" (toString inst.gpuMemoryUtilization)
          ]
          ++ pkgs.lib.optionals (inst ? runner) ["--runner" (pkgs.lib.escapeShellArg inst.runner)]
          ++ pkgs.lib.optionals (inst ? quantization) ["--quantization" (pkgs.lib.escapeShellArg inst.quantization)]
          ++ pkgs.lib.optionals (inst ? kvCacheDtype) ["--kv-cache-dtype" (pkgs.lib.escapeShellArg inst.kvCacheDtype)]
          ++ ["--max-model-len" (toString inst.maxModelLen)]
          ++ ["--max-num-seqs" (toString inst.maxNumSeqs)]
          ++ pkgs.lib.optionals (inst ? speculativeConfig) [
            "--speculative-config"
            (pkgs.lib.escapeShellArg (builtins.toJSON inst.speculativeConfig))
          ]
          ++ pkgs.lib.optionals inst.enforceEager ["--enforce-eager"]
          ++ pkgs.lib.optionals (inst ? cudagraphCaptureSizes) [
            "--compilation-config"
            (pkgs.lib.escapeShellArg (builtins.toJSON {
              cudagraph_capture_sizes = inst.cudagraphCaptureSizes;
            }))
          ]
          ++ pkgs.lib.optionals (inst ? reasoningParser) ["--reasoning-parser" inst.reasoningParser]
          ++ pkgs.lib.optionals (inst.enableAutoToolChoice or false) ["--enable-auto-tool-choice"]
          ++ pkgs.lib.optionals (inst ? toolCallParser) ["--tool-call-parser" inst.toolCallParser]
          ++ pkgs.lib.optionals (inst.languageModelOnly or false) ["--language-model-only"]
          ++ map pkgs.lib.escapeShellArg inst.extraArgs
        );

      vllm-xpu-maintenance = pkgs.writeShellApplication {
        name = "vllm-xpu-maintenance";
        runtimeInputs = [pkgs.coreutils pkgs.curl];
        text = ''
          set -m

          runtime_root="''${XDG_CACHE_HOME:-$HOME/.cache}/vllm-xpu-maintenance"
          mkdir -p "$runtime_root/chat" "$runtime_root/embedding"

          export VLLM_TARGET_DEVICE=xpu
          export HF_HOME=/var/cache/huggingface
          export CCL_PROCESS_LAUNCHER=none
          export CCL_ATL_TRANSPORT=ofi
          export CCL_ZE_IPC_EXCHANGE=sockets
          export CCL_LOG_LEVEL=warn

          cleanup() {
            trap - EXIT INT TERM
            kill "''${chat_pid:-}" "''${embedding_pid:-}" 2>/dev/null || true
            wait "''${chat_pid:-}" "''${embedding_pid:-}" 2>/dev/null || true
          }
          trap cleanup EXIT
          trap 'cleanup; exit 130' INT TERM

          HOME="$runtime_root/embedding" \
          VLLM_CACHE_ROOT="$runtime_root/embedding" \
            ${vllm-xpu-chat}/bin/vllm serve \
              ${mkMaintenanceServeArgs 8001 chatProfile.embedding} &
          embedding_pid=$!

          for _ in $(seq 1 180); do
            if curl --fail --silent --max-time 2 \
              http://127.0.0.1:8001/v1/models >/dev/null; then
              break
            fi
            if ! kill -0 "$embedding_pid" 2>/dev/null; then
              wait "$embedding_pid"
            fi
            sleep 1
          done
          curl --fail --silent --max-time 2 \
            http://127.0.0.1:8001/v1/models >/dev/null

          chat_attempt=1
          while :; do
            HOME="$runtime_root/chat" \
            VLLM_CACHE_ROOT="$runtime_root/chat" \
            VLLM_XPU_ENABLE_XPU_GRAPH=1 \
              ${vllm-xpu-chat}/bin/vllm serve \
                ${mkMaintenanceServeArgs 8000 chatProfile.chat} &
            chat_pid=$!

            if wait "$chat_pid"; then
              exit 0
            else
              chat_status=$?
            fi
            chat_pid=""

            if [ "$chat_attempt" -ge 3 ]; then
              echo "chat failed after $chat_attempt attempts" >&2
              exit "$chat_status"
            fi

            echo "chat attempt $chat_attempt failed; retrying with compiled artifacts" >&2
            chat_attempt=$((chat_attempt + 1))
            sleep 5
          done
        '';
      };

      mkKvarnEagerRunner = name: profile:
        pkgs.writeShellApplication {
          inherit name;
          text = ''
            runtime_root="''${XDG_CACHE_HOME:-$HOME/.cache}/${name}"
            mkdir -p "$runtime_root"

            export VLLM_TARGET_DEVICE=xpu
            export HF_HOME=/var/cache/huggingface
            export CCL_PROCESS_LAUNCHER=none
            export CCL_ATL_TRANSPORT=ofi
            export CCL_ZE_IPC_EXCHANGE=sockets
            export CCL_LOG_LEVEL=warn

            HOME="$runtime_root" VLLM_CACHE_ROOT="$runtime_root" \
              exec ${vllm-xpu-chat}/bin/vllm serve \
                ${mkMaintenanceServeArgs 8000 profile}
          '';
        };

      kvarn-eager-k4v4 = mkKvarnEagerRunner
        "vllm-xpu-kvarn-eager-k4v4"
        chatProfile.kvarnEagerK4V4;
      kvarn-eager-k4v2 = mkKvarnEagerRunner
        "vllm-xpu-kvarn-eager-k4v2"
        chatProfile.kvarnEagerK4V2;
      kvarn-mtp-eager-k4v4 = mkKvarnEagerRunner
        "vllm-xpu-kvarn-mtp-eager-k4v4"
        chatProfile.kvarnMtpEagerK4V4;

      # ---- shells + misc helpers ----
      syclToolchainShellHook = import ./nix/sycl-shellhook.nix {
        inherit pkgs intel-oneapi;
        cutlass-src = sycl-tla-src;
      };

      hfMetadata = pkgs.callPackage ./nix/hf-metadata.nix {};

      lint = import ./nix/lint.nix {inherit pkgs;};
    in {
      # Per-system helpers consumers reach via
      # `inputs.vllm-xpu-nix.lib.${pkgs.system}.fromHfConfig`.
      # `lib.${system}` (rather than just `lib`) is intentional — the
      # helpers wrap `pkgs.fetchurl` (system-scoped) and the mk* builders
      # close over this system's pkgs / torch-xpu / oneAPI substrate.
      lib = {
        inherit
          (hfMetadata)
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
        inherit mkVllm mkVllmXpuKernels chatProfile;
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
          vllm-xpu-kernels
          vllm-xpu-kernels-unstable
          vllm-xpu
          vllm-xpu-unstable
          vllm-xpu-chat
          ;
        inherit
          (stableLibs)
          attn-kernels-xe-2
          gdn-attn-kernels-xe-2
          mhc-kernels-xe-2
          mqa-logits-kernels-xe-2
          grouped-gemm-xe-2
          grouped-gemm-xe-default
          ;
        default = intel-oneapi;
        inherit quantize kl-eval lint;
      };

      apps = {
        vllm-xpu-chat = {
          type = "app";
          program = "${vllm-xpu-maintenance}/bin/vllm-xpu-maintenance";
        };
        vllm-xpu-kvarn-eager-k4v4 = {
          type = "app";
          program = "${kvarn-eager-k4v4}/bin/vllm-xpu-kvarn-eager-k4v4";
        };
        vllm-xpu-kvarn-eager-k4v2 = {
          type = "app";
          program = "${kvarn-eager-k4v2}/bin/vllm-xpu-kvarn-eager-k4v2";
        };
        vllm-xpu-kvarn-mtp-eager-k4v4 = {
          type = "app";
          program = "${kvarn-mtp-eager-k4v4}/bin/vllm-xpu-kvarn-mtp-eager-k4v4";
        };
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

      devShells = import ./nix/devshells.nix {
        inherit pkgs syclToolchainShellHook lint torch-xpu triton-xpu;
        # Dev shells track the unstable (fork) variant — the one actually
        # deployed — so `nix develop` never realizes the stable closure.
        kernelLibs = unstableLibs;
        vllmPkg = vllm-xpu-unstable;
        vllmKernels = vllm-xpu-kernels-unstable;
      };
    });
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
      nixosModules.default = {...}: {
        imports = [./nix/modules/vllm-xpu.nix];
        nixpkgs.overlays = [self.overlays.default];
      };

      # Overlay that injects the XPU package set into a host's pkgs.
      # Pair with the bare `nixosModules.vllm-xpu`, or just import
      # `nixosModules.default` which applies this for you.
      overlays.default = _final: prev: let
        pkgs = systemOutputs.packages.${prev.stdenv.hostPlatform.system} or {};
        pick = name: lib.optionalAttrs (pkgs ? ${name}) {${name} = pkgs.${name};};
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
        // pick "vllm-xpu-chat";
    };
}
