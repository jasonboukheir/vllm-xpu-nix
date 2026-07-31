{
  config,
  lib,
  pkgs,
  ...
}: let
  cfg = config.services.vllm-xpu;

  effectivePackage =
    if cfg.package ? withAotDevices
    then cfg.package.withAotDevices cfg.aotDevices
    else cfg.package;

  bmgCclEnv = {
    CCL_PROCESS_LAUNCHER = "none";
    CCL_ATL_TRANSPORT = "ofi";
    CCL_ZE_IPC_EXCHANGE = "sockets";
    CCL_LOG_LEVEL = "warn";
  };

  instanceModule = {name, ...}: {
    options = {
      enable = lib.mkEnableOption "this vLLM-XPU instance";

      package = lib.mkOption {
        type = lib.types.package;
        default = effectivePackage;
        defaultText = lib.literalExpression ''
          config.services.vllm-xpu.package, with .withAotDevices
          services.vllm-xpu.aotDevices applied
        '';
        description = ''
          vLLM-XPU package providing `bin/vllm`. Defaults to the
          flake-level `services.vllm-xpu.package`, automatically
          re-derived through `.withAotDevices
          services.vllm-xpu.aotDevices`. Override per instance to
          e.g. mix the stable build for chat with the unstable fork
          for an experimental embedder.
        '';
      };

      model = lib.mkOption {
        type = lib.types.str;
        description = ''
          HuggingFace model id or local path served by vLLM. Positional
          argument of `vllm serve`. With `quantization = "inc"` (Intel
          Neural Compressor dispatch on the IPEX-free build), point at
          a pre-quantized GPTQv2 sym-int4 repo. With
          `quantization = "sym_int4"` (legacy IPEX online INT4 path),
          point at the unquantized BF16 base — IPEX packs the weights
          to INT4 at load time.
        '';
        example = "palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4";
      };

      revision = lib.mkOption {
        type = lib.types.nullOr (lib.types.strMatching "[0-9a-fA-F]{40}");
        default = null;
        description = ''
          Immutable Hugging Face commit passed through as `vllm serve
          --revision <commit>`. The cache GC manifest records this revision.
          Null keeps the whole model repository rooted for backward
          compatibility.
        '';
      };

      servedName = lib.mkOption {
        type = lib.types.str;
        description = ''
          Pass `--served-model-name <id>`. The id advertised over the
          OpenAI-compatible API — what downstream consumers (LiteLLM,
          OpenWebUI) reference as `model`. Set to the same value as
          `model` if you want vLLM to expose the raw HF repo id.
        '';
        example = "qwen3.6-35b-a3b";
      };

      runner = lib.mkOption {
        type = lib.types.nullOr (lib.types.enum ["generate" "pooling" "draft"]);
        default = null;
        description = ''
          Pass `--runner <kind>`. `pooling` exposes `/v1/embeddings`
          for an embedding model and skips the autoregressive engine
          (no KV pool). `null` lets vLLM auto-detect from the model's
          config (the right default for chat checkpoints).
        '';
        example = "pooling";
      };

      host = lib.mkOption {
        type = lib.types.str;
        default = "127.0.0.1";
        description = ''
          Pass `--host <addr>`. Bind address for the OpenAI-compatible
          API server. Defaults to localhost-only — public exposure
          should go through a fronting proxy (LiteLLM, nginx, etc.).
        '';
      };

      port = lib.mkOption {
        type = lib.types.port;
        default = 8000;
        description = ''
          Pass `--port <n>`. TCP port the API server listens on. Each
          enabled instance must pick a unique port — the systemd units
          all run on the same host network namespace.
        '';
      };

      dtype = lib.mkOption {
        type = lib.types.str;
        default = "bfloat16";
        description = ''
          Pass `--dtype <type>`. Activation/compute dtype.
          `bfloat16` is the default for the IPEX-free vllm-xpu stack
          shipped by this flake; `float16` is required when running
          against the legacy intel/llm-scaler-vllm `sym_int4` path
          which IPEX builds on (IPEX rejects bf16 there).
        '';
      };

      quantization = lib.mkOption {
        type = lib.types.nullOr lib.types.str;
        default = null;
        example = "inc";
        description = ''
          Pass `--quantization <method>`, or null to skip. With this
          flake's native vllm-xpu build use:
          - `inc` — Intel Neural Compressor dispatch. Loads
            pre-quantized GPTQv2 sym-int4 weights and routes the MoE
            through `vllm-xpu-kernels`' `xpu_fused_moe(is_int4=True)`.
            Compatible with torch.compile + XPU graph capture.
          - `sym_int4` — IPEX online INT4 (GGML Q4_0) MoE path.
            Quantizes BF16 weights at load. Forces eager mode (Dynamo
            trips on IPEX C-extensions) and is only meaningful on the
            legacy llm-scaler image.
          For embedding models, leave at `null` (FP16/BF16).
        '';
      };

      attentionBackend = lib.mkOption {
        type = lib.types.nullOr lib.types.str;
        default = null;
        example = "TRITON_ATTN";
        description = ''
          Pass `--attention-backend <name>`. Overrides vLLM's per-platform
          backend default. On XPU the platform default is `FLASH_ATTN`,
          which dispatches through `torch.ops._vllm_fa2_C.varlen_fwd` — a
          CUDA-only kernel — and crashes any model whose architecture
          isn't covered by the IPEX attention path (e.g. Whisper
          cross-attention). Set to `"TRITON_ATTN"` for those cases; it's
          the portable XPU fallback the vLLM docs recommend.

          Replaces the `VLLM_ATTENTION_BACKEND` env var which vLLM 0.20+
          no longer reads (it emits "Unknown vLLM environment variable
          detected" and silently falls back to the platform default).
          Configs that still pass it through `extraEnvironment` should
          migrate to this option.
        '';
      };

      kvCacheDtype = lib.mkOption {
        type = lib.types.nullOr lib.types.str;
        default = null;
        example = "fp8";
        description = ''
          Pass `--kv-cache-dtype <type>`, or null to fall back to
          model-precision KV. Options:
          - `fp8` — universal, ~2x KV headroom for ~2-3% per-stream
            throughput cost.
          - `turboquant_k3v4_nc` (and siblings) — only on builds that
            include the TurboQuant kernels. Compresses K to 3-bit
            MSE-Lloyd-Max + V to 4-bit. KL vs FP16 KV at 4096 ctx
            measured at 0.0179 — functionally identical for greedy
            decoding.
          Tighter KV is the headroom that lets concurrent agentic
          sessions accumulate context without evicting.
        '';
      };

      maxModelLen = lib.mkOption {
        type = lib.types.nullOr lib.types.int;
        default = null;
        description = ''
          Pass `--max-model-len <n>`. Caps prompt + output tokens per
          request and shapes the KV pool — the only knob that
          meaningfully reclaims VRAM for KV at fixed
          `gpuMemoryUtilization`. Note: Qwen3.6-35B-A3B's model card
          recommends keeping at least 128k to preserve thinking
          quality; values below that are a deliberate VRAM tradeoff.
          For embedding workloads, this caps the largest chunk a
          client may submit.
        '';
      };

      maxNumSeqs = lib.mkOption {
        type = lib.types.nullOr lib.types.int;
        default = null;
        description = ''
          Pass `--max-num-seqs <n>`. Caps the engine's concurrent
          sequence count, which bounds the worst-case shape used by
          vLLM's startup memory-profile pass (max_num_seqs ×
          max_num_batched_tokens). Lower values trade concurrency
          ceiling for a smaller activation peak — the difference between
          OOM at engine init and clean startup when GPU VRAM is shared
          with other models. `null` keeps vLLM's default (256).
        '';
      };

      gpuMemoryUtilization = lib.mkOption {
        type = lib.types.float;
        default = 0.9;
        description = ''
          Pass `--gpu-memory-utilization <fraction>`. Fraction of XPU
          VRAM vLLM claims for weights, activations, KV pool, and graph
          capture buffers combined. Higher values grow the KV pool but
          starve co-resident services. On a 32 GiB B70 sharing with a
          ~2.2 GiB embedding model and ~0.7 GiB whisper, 0.85 leaves
          ~1.9 GiB headroom; 0.93 over-commits.
        '';
      };

      speculativeConfig = lib.mkOption {
        type = lib.types.nullOr lib.types.attrs;
        default = null;
        description = ''
          JSON attrset passed to `--speculative-config`. Enables
          speculative decoding (MTP / EAGLE / draft-target). Methods:
          - `mtp` — generic MTP drafter, model-agnostic plumbing.
          - `qwen3_next_mtp` — Qwen3-family-specific dispatcher; same
            drafter, model-aware fast path. Use this for
            Qwen3.6-35B-A3B (the model card's recommendation).
          - `eagle` / `eagle3` — separate draft model.
          The K value (`num_speculative_tokens`) must match
          `cudagraphCaptureSizes` because vLLM rounds capture sizes up
          to multiples of (K + 1) — verify-pass shape = 1 real + K
          spec. K=2 wants `[3]`, K=3 wants `[4]`, etc. Requires the
          GDN spec-decode dispatcher patch on XPU (`vllm-xpu-unstable`
          tracks it); otherwise the SYCL `gdn_attention` kernel
          asserts on the first verify pass.
        '';
        example = lib.literalExpression ''{ method = "qwen3_next_mtp"; num_speculative_tokens = 2; }'';
      };

      enforceEager = lib.mkOption {
        type = lib.types.bool;
        default = false;
        description = ''
          Pass `--enforce-eager`. Required on the legacy
          intel/llm-scaler-vllm image because Dynamo trips on IPEX's
          pybind11 C-extensions. The IPEX-free vllm-xpu build shipped
          by this flake makes Dynamo + Inductor + XPU graph capture
          (PIECEWISE) viable; turning eager off there lifts
          single-stream from ~20 tok/s to ~58 tok/s on
          Qwen3.6-35B-A3B (matching llama.cpp's hand-tuned SYCL
          pipeline). Pair with `enableXpuGraph` and
          `cudagraphCaptureSizes`. Embedding workloads (`runner =
          "pooling"`) typically leave this on — eager-mode throughput
          is at the kernel ceiling for pooled forward passes.
        '';
      };

      enableXpuGraph = lib.mkOption {
        type = lib.types.bool;
        default = false;
        description = ''
          Set `VLLM_XPU_ENABLE_XPU_GRAPH=1` in the unit env. This is
          what actually captures the decode loop into a Level Zero
          command list — torch.compile alone helps a little, but the
          ~3x single-stream win comes from graph replay collapsing the
          hundreds of per-kernel CPU dispatches into one submission
          per token. No-op while `enforceEager = true`.
        '';
      };

      cudagraphCaptureSizes = lib.mkOption {
        type = lib.types.nullOr (lib.types.listOf lib.types.int);
        default = null;
        description = ''
          Batch sizes to capture into PIECEWISE XPU graphs (passed via
          `--compilation-config '{"cudagraph_capture_sizes":[…]}'`).
          Defaults to `null` which lets vLLM pick — typically 19 sizes
          from 1 to 128 costing ~7 GiB of VRAM, which OOMs the KV
          cache budget on a 32 GiB B70 alongside a 20 GiB model. Set
          e.g. `[ 1 2 4 8 ]` (~1.4 GiB) to cover single-stream + light
          concurrency; beyond the largest captured size vLLM falls
          back to eager-style submission. Ineffective unless
          `enableXpuGraph = true` and `enforceEager = false`.
        '';
        example = lib.literalExpression "[ 1 4 ]";
      };

      cudagraphMode = lib.mkOption {
        type = lib.types.nullOr (lib.types.enum [
          "NONE"
          "PIECEWISE"
          "FULL"
          "FULL_AND_PIECEWISE"
          "FULL_DECODE_ONLY"
        ]);
        default = null;
        example = "PIECEWISE";
        description = ''
          Select which graph-capture strategy vLLM uses
          (`compilation_config.cudagraph_mode`). Null leaves vLLM at
          its default (currently `FULL_AND_PIECEWISE`). Override when
          the default trips an XPU-specific capture bug, e.g.
          `PIECEWISE` skips the FULL decode capture that crashes the
          FA2 varlen kernel on oneAPI 2025.3 with
          `sycl_ext_oneapi_work_group_scratch_memory feature is not
          yet available for use with the SYCL Graph extension`.
          Combined with `cudagraphCaptureSizes` into a single
          `--compilation-config` JSON payload. Ineffective unless
          `enableXpuGraph = true` and `enforceEager = false`.
        '';
      };

      reasoningParser = lib.mkOption {
        type = lib.types.nullOr lib.types.str;
        default = null;
        example = "qwen3";
        description = ''
          Pass `--reasoning-parser <value>`. Splits the model's
          `<think>...</think>` reasoning block off the user-visible
          answer and emits it on `delta.reasoning` instead of
          `delta.content`. Use the bundled `qwen3` for the standard
          path, or set this to a parser registered by
          `reasoningParserPlugin` for custom behavior (e.g. a
          parser that bypasses extraction in fast mode).
        '';
      };

      reasoningParserPlugin = lib.mkOption {
        type = lib.types.nullOr lib.types.path;
        default = null;
        description = ''
          Path to a Python file implementing a custom reasoning parser.
          Passed verbatim as `--reasoning-parser-plugin <path>`. The
          plugin must register itself via
          `ReasoningParserManager.register_module("name")`; that name
          then goes in `reasoningParser`.
        '';
      };

      enableAutoToolChoice = lib.mkOption {
        type = lib.types.bool;
        default = false;
        description = ''
          Pass `--enable-auto-tool-choice`. Required for the OpenAI
          tool-calling API (`tool_choice = "auto"`) to actually run
          the parser configured via `toolCallParser` — without this
          flag, vLLM passes the request through without parsing
          tool-call markup, so clients see the raw `<tool_call>` text
          on `message.content` instead of structured `tool_calls`.
          Enable for agentic clients (Claude Code, Aider, OpenWebUI).
        '';
      };

      toolCallParser = lib.mkOption {
        type = lib.types.nullOr lib.types.str;
        default = null;
        example = "qwen3_coder";
        description = ''
          Pass `--tool-call-parser <value>`. Selects the parser that
          extracts tool calls from the model's emitted text into the
          OpenAI `tool_calls` schema. For Qwen3.6-35B-A3B set
          `"qwen3_coder"` (per the model card's recommended vLLM
          command — the Qwen3-Coder family parser is also what this
          MoE checkpoint was tuned against). Only effective when
          `enableAutoToolChoice = true`.
        '';
      };

      toolCallParserPlugin = lib.mkOption {
        type = lib.types.nullOr lib.types.path;
        default = null;
        description = ''
          Path to a Python file implementing a custom tool-call parser.
          Passed as `--tool-parser-plugin <path>`. The plugin must
          register itself via
          `ToolParserManager.register_module("name")`; that name then
          goes in `toolCallParser`.
        '';
      };

      limitMmPerPrompt = lib.mkOption {
        type = lib.types.nullOr lib.types.attrs;
        default = null;
        description = ''
          JSON attrset passed to `--limit-mm-per-prompt`. For text-only
          use of a multimodal checkpoint (Qwen3.6-35B-A3B is VL-tagged
          even though we only want the text path), set
          `{ image = 0; video = 0; }` to skip the multimodal-budget
          init that otherwise calls
          `Qwen2VLImageProcessor.max_pixels` and crashes on newer
          transformers. Prefer `languageModelOnly = true` for the
          same effect with less typing.
        '';
        example = lib.literalExpression ''{ image = 0; video = 0; }'';
      };

      languageModelOnly = lib.mkOption {
        type = lib.types.bool;
        default = false;
        description = ''
          Pass `--language-model-only`. Sets every modality's
          `limit_per_prompt` to 0 — equivalent to
          `limitMmPerPrompt = { image = 0; video = 0; ... }` but
          sweeping every registered modality. Use for text-only
          inference on VL-tagged checkpoints. Per vLLM source
          (vllm/config/multimodal.py:315) this only changes the
          runtime limit; vision-tower weights still load if present
          in the checkpoint.
        '';
      };

      cacheDir = lib.mkOption {
        type = lib.types.path;
        default = "/var/lib/vllm-xpu/${name}";
        defaultText = lib.literalExpression "\"/var/lib/vllm-xpu/\${name}\"";
        description = ''
          Per-instance state directory and the root of `HF_HOME` (unless
          `sharedHfCache` overrides it). Each instance gets its own
          subdirectory so concurrent downloads of different models can't
          race over the same `~/.cache/huggingface/locks`.

          The compile caches (`HOME` for Triton's `~/.triton` and
          `VLLM_CACHE_ROOT` for torch.compile/inductor artefacts) live
          one level deeper, under `''${cacheDir}/build/<store-hash>`,
          keyed to the instance package's Nix store hash. A package bump
          (torch, kernels, oneAPI toolkit) changes the hash and so gets a
          fresh cache, which avoids replaying a stale Triton
          `spirv_utils.so` linked against the previous libsycl ABI.
          Keeping everything under `cacheDir` also keeps those writes
          inside the unit's `ReadWritePaths` sandbox —
          `users.users.vllm.home` is the parent `/var/lib/vllm-xpu`,
          which is read-only to the unit.
        '';
      };

      cclEnv = lib.mkOption {
        type = lib.types.attrsOf lib.types.str;
        default = bmgCclEnv;
        defaultText = lib.literalExpression ''
          {
            CCL_PROCESS_LAUNCHER = "none";
            CCL_ATL_TRANSPORT = "ofi";
            CCL_ZE_IPC_EXCHANGE = "sockets";
            CCL_LOG_LEVEL = "warn";
          }
        '';
        description = ''
          OneCCL env vars exported into the systemd unit. The default
          is the safe BMG single-card config: with the default xccl
          backend even at world_size=1 the warmup `all_reduce` hangs
          unless IPC is forced to sockets and the launcher is told
          there's no MPI ranks. Override per instance only for
          multi-card setups.
        '';
      };

      environmentFile = lib.mkOption {
        type = lib.types.nullOr lib.types.path;
        default = null;
        description = ''
          Optional `EnvironmentFile` (e.g. for `HF_TOKEN` secrets on
          gated repos). Read by systemd at start time; secrets stay
          out of the world-readable unit env.
        '';
      };

      extraArgs = lib.mkOption {
        type = lib.types.listOf lib.types.str;
        default = [];
        description = "Extra arguments appended to the `vllm serve` command line.";
      };

      extraEnvironment = lib.mkOption {
        type = lib.types.attrsOf lib.types.str;
        default = {};
        description = ''
          Extra environment variables to merge into the systemd unit
          on top of the built-in vLLM/oneCCL defaults. Useful for
          knobs that don't have a dedicated option (e.g.
          `VLLM_LOGGING_LEVEL`, `TORCHDYNAMO_VERBOSE`).
        '';
      };
    };
  };

  enabledInstances = lib.filterAttrs (_: i: i.enable) cfg.instances;

  mkServeArgs = inst:
    lib.concatStringsSep " " (
      [
        (lib.escapeShellArg inst.model)
        "--host"
        (lib.escapeShellArg inst.host)
        "--port"
        (toString inst.port)
        "--served-model-name"
        (lib.escapeShellArg inst.servedName)
        "--dtype"
        (lib.escapeShellArg inst.dtype)
        "--gpu-memory-utilization"
        (toString inst.gpuMemoryUtilization)
      ]
      ++ lib.optionals (inst.runner != null) ["--runner" (lib.escapeShellArg inst.runner)]
      ++ lib.optionals (inst.revision != null) ["--revision" (lib.escapeShellArg inst.revision)]
      ++ lib.optionals (inst.quantization != null) ["--quantization" (lib.escapeShellArg inst.quantization)]
      ++ lib.optionals (inst.attentionBackend != null) [
        "--attention-backend"
        (lib.escapeShellArg inst.attentionBackend)
      ]
      ++ lib.optionals (inst.kvCacheDtype != null) ["--kv-cache-dtype" (lib.escapeShellArg inst.kvCacheDtype)]
      ++ lib.optionals (inst.maxModelLen != null) ["--max-model-len" (toString inst.maxModelLen)]
      ++ lib.optionals (inst.maxNumSeqs != null) ["--max-num-seqs" (toString inst.maxNumSeqs)]
      ++ lib.optionals (inst.speculativeConfig != null) [
        "--speculative-config"
        (lib.escapeShellArg (builtins.toJSON inst.speculativeConfig))
      ]
      ++ lib.optionals inst.enforceEager ["--enforce-eager"]
      ++ lib.optionals
      (inst.cudagraphCaptureSizes != null || inst.cudagraphMode != null)
      [
        "--compilation-config"
        (lib.escapeShellArg (builtins.toJSON (
          lib.optionalAttrs (inst.cudagraphMode != null) {
            cudagraph_mode = inst.cudagraphMode;
          }
          // lib.optionalAttrs (inst.cudagraphCaptureSizes != null) {
            cudagraph_capture_sizes = inst.cudagraphCaptureSizes;
          }
        )))
      ]
      ++ lib.optionals (inst.reasoningParser != null) [
        "--reasoning-parser"
        (lib.escapeShellArg inst.reasoningParser)
      ]
      ++ lib.optionals (inst.reasoningParserPlugin != null) [
        "--reasoning-parser-plugin"
        (lib.escapeShellArg (toString inst.reasoningParserPlugin))
      ]
      ++ lib.optionals inst.enableAutoToolChoice ["--enable-auto-tool-choice"]
      ++ lib.optionals (inst.toolCallParser != null) [
        "--tool-call-parser"
        (lib.escapeShellArg inst.toolCallParser)
      ]
      ++ lib.optionals (inst.toolCallParserPlugin != null) [
        "--tool-parser-plugin"
        (lib.escapeShellArg (toString inst.toolCallParserPlugin))
      ]
      ++ lib.optionals (inst.limitMmPerPrompt != null) [
        "--limit-mm-per-prompt"
        (lib.escapeShellArg (builtins.toJSON inst.limitMmPerPrompt))
      ]
      ++ lib.optionals inst.languageModelOnly ["--language-model-only"]
      ++ map lib.escapeShellArg inst.extraArgs
    );

  hfHomeFor = inst:
    if cfg.sharedHfCache != null
    then cfg.sharedHfCache
    else inst.cacheDir;

  buildKeyFor = inst: builtins.substring 0 32 (builtins.baseNameOf (toString inst.package));
  buildCacheDirFor = inst: "${inst.cacheDir}/build/${buildKeyFor inst}";

  mkUnit = name: inst:
    lib.nameValuePair "vllm-xpu-${name}" {
      description = "vLLM XPU serve (${inst.servedName})";
      wantedBy = ["multi-user.target"];
      after = ["network-online.target"];
      wants = ["network-online.target"];

      environment =
        {
          VLLM_TARGET_DEVICE = "xpu";
          HOME = buildCacheDirFor inst;
          HF_HOME = hfHomeFor inst;
          VLLM_CACHE_ROOT = buildCacheDirFor inst;
        }
        // inst.cclEnv
        // lib.optionalAttrs inst.enableXpuGraph {
          VLLM_XPU_ENABLE_XPU_GRAPH = "1";
        }
        // inst.extraEnvironment;

      serviceConfig =
        {
          Type = "simple";
          User = cfg.user;
          Group = cfg.group;
          SupplementaryGroups =
            ["render" "video"]
            ++ lib.optional (cfg.sharedHfCache != null) cfg.sharedHfCacheGroup;

          ExecStartPre = "-${pkgs.findutils}/bin/find ${inst.cacheDir}/build -mindepth 1 -maxdepth 1 ! -name ${buildKeyFor inst} -exec ${pkgs.coreutils}/bin/rm -rf {} +";

          ExecStart = "${inst.package}/bin/vllm serve ${mkServeArgs inst}";

          Restart = "on-failure";
          RestartSec = 5;
          TimeoutStartSec = 0;

          DeviceAllow = ["char-drm rw"];
          PrivateDevices = false;

          WorkingDirectory = inst.cacheDir;

          ProtectSystem = "strict";
          ProtectHome = true;
          NoNewPrivileges = true;
          ReadWritePaths =
            [inst.cacheDir]
            ++ lib.optional (cfg.sharedHfCache != null) cfg.sharedHfCache;
        }
        // lib.optionalAttrs (cfg.sharedHfCache != null) {
          UMask = "0002";
        }
        // lib.optionalAttrs (inst.environmentFile != null) {
          EnvironmentFile = inst.environmentFile;
        };
    };
in {
  imports = [./hf-cache.nix];

  options.services.vllm-xpu = {
    package = lib.mkOption {
      type = lib.types.package;
      # The unstable (jasonboukheir/vllm fork) variant is the deployed /
      # primary build; defaulting to it avoids the footgun of silently
      # serving the upstream stable build when consumers forget to set
      # `package`.
      default = pkgs.vllm-xpu-unstable;
      defaultText = lib.literalExpression "pkgs.vllm-xpu-unstable";
      description = ''
        Default vLLM-XPU package used by every instance unless the
        instance overrides it. Defaults to `pkgs.vllm-xpu-unstable`
        (the jasonboukheir/vllm fork — the deployed/primary variant;
        carries patches required for `speculativeConfig` on
        Qwen3-family MTP and the GDN graph-capture fix). Set to
        `pkgs.vllm-xpu` for the upstream stable release.
      '';
    };

    user = lib.mkOption {
      type = lib.types.str;
      default = "vllm";
      description = ''
        User account every instance's systemd unit runs as. A single
        shared user simplifies `/dev/dri` group membership; isolation
        between instances comes from per-instance `cacheDir` and
        systemd unit hardening, not separate users.
      '';
    };

    group = lib.mkOption {
      type = lib.types.str;
      default = "vllm";
      description = "Group account paired with `user`.";
    };

    sharedHfCache = lib.mkOption {
      type = lib.types.nullOr lib.types.path;
      default = "/var/cache/huggingface";
      defaultText = lib.literalExpression "\"/var/cache/huggingface\"";
      description = ''
        System-wide HuggingFace cache shared across every instance
        and (via the configured group) interactive dev shells.
        Overrides each instance's `HF_HOME` so the same model pulled
        by `chat`, `embedding`, or a developer running
        `huggingface-cli download` lands on one content-addressed
        `hub/models--…/blobs/<sha>` tree instead of being duplicated
        under `''${cacheDir}/huggingface`. Set to `null` to disable
        and fall back to a per-instance `HF_HOME` rooted at
        `cacheDir`.

        The module creates the directory mode `2775` (setgid +
        group-writable) owned by `cfg.user`:`sharedHfCacheGroup` so
        files written by the service user or by humans in the group
        inherit group ownership and stay cross-readable, adds the
        path to each unit's `ReadWritePaths`, adds the group to the
        unit's `SupplementaryGroups`, and sets `UMask=0002` on the
        unit so newly-written files keep the group-writable bit.

        Per-instance compiled artefacts stay in the per-instance
        `cacheDir` — only `HF_HOME` (Hub weights, datasets,
        transformers cache) is consolidated. `VLLM_CACHE_ROOT` and
        Triton's `~/.triton` (via `HOME`) live under
        `''${cacheDir}/build/<store-hash>`, keyed to the instance
        package so a build bump gets a fresh compile cache rather than
        replaying one built against the old toolchain.

        Dev side: `export HF_HOME=/var/cache/huggingface` (or symlink
        `~/.cache/huggingface` to it) and add yourself to the group
        named by `sharedHfCacheGroup`.
      '';
    };

    sharedHfCacheGroup = lib.mkOption {
      type = lib.types.str;
      default = "huggingface";
      description = ''
        Group that owns `sharedHfCache`. The module creates the group
        if it doesn't already exist and adds `cfg.user` to it. Add
        interactive users to the same group via
        `users.users.<name>.extraGroups` so they can read and write
        the shared cache. Has no effect when `sharedHfCache = null`.
      '';
    };

    aotDevices = lib.mkOption {
      type = lib.types.listOf lib.types.str;
      default = [];
      example = lib.literalExpression ''[ "bmg" ]'';
      description = ''
        SYCL AOT target list compiled into the kernel `.so`s. Passed
        through to `cfg.package.withAotDevices`.

        - `[]` (default): JIT mode. Kernels ship as SPIR-V and IGC
          specializes them at first dispatch. The 256-GRF hint is
          preserved via patches/0006-decouple-256grf-from-aot.patch
          so JIT codegen matches AOT codegen quality; the only
          difference is a one-shot first-dispatch pause per kernel.
        - explicit list (`[ "bmg" ]`, `[ "bmg" "pvc" ]`): AOT for
          the listed devices. Each entry triggers a separate ocloc
          invocation at link time, so multi-device builds get
          expensive.

        A custom `cfg.package` that doesn't expose the `withAotDevices`
        passthru is left unchanged.
      '';
    };

    instances = lib.mkOption {
      type = lib.types.attrsOf (lib.types.submodule instanceModule);
      default = {};
      description = ''
        Named vLLM-XPU instances. Each enabled instance gets its own
        systemd unit `vllm-xpu-<name>.service`, its own listen port,
        and its own build cache under `/var/lib/vllm-xpu/`. By default they
        share the Hugging Face model cache at `/var/cache/huggingface`.
        They also share `/dev/dri`, the `vllm` user/group, and the BMG
        single-card oneCCL environment defaults.

        Typical layout: one chat instance plus one pooling-mode
        embedder co-resident on a single B70.
      '';
      example = lib.literalExpression ''
        {
          chat = {
            enable = true;
            model = "palmfuture/Qwen3.6-35B-A3B-GPTQ-Int4";
            servedName = "qwen3.6-35b-a3b";
            quantization = "inc";
            kvCacheDtype = "turboquant_k3v4_nc";
            gpuMemoryUtilization = 0.85;
          };
          embedding = {
            enable = true;
            port = 8001;
            runner = "pooling";
            model = "Qwen/Qwen3-Embedding-0.6B";
            servedName = "qwen3-embedding-0.6b";
            gpuMemoryUtilization = 0.07;
            maxNumSeqs = 8;
            enforceEager = true;
          };
        }
      '';
    };
  };

  config = lib.mkMerge [
    {
      services.hf-cache =
        {
          roots =
            lib.mapAttrsToList (name: inst: {
              source = "vllm-xpu.${name}";
              type = "model";
              repo = inst.model;
              revision = inst.revision;
            })
            enabledInstances;
        }
        // lib.optionalAttrs (cfg.sharedHfCache != null) {
          home = cfg.sharedHfCache;
        };
    }
    (lib.mkIf (enabledInstances != {}) {
      users.users.${cfg.user} = {
        isSystemUser = true;
        group = cfg.group;
        home = "/var/lib/vllm-xpu";
        createHome = false;
        extraGroups =
          ["render" "video"]
          ++ lib.optional (cfg.sharedHfCache != null) cfg.sharedHfCacheGroup;
      };
      users.groups =
        {
          ${cfg.group} = {};
        }
        // lib.optionalAttrs (cfg.sharedHfCache != null) {
          ${cfg.sharedHfCacheGroup} = {};
        };

      systemd.tmpfiles.rules =
        ["d /var/lib/vllm-xpu 0750 ${cfg.user} ${cfg.group} - -"]
        ++ lib.concatLists (lib.mapAttrsToList
          (_: inst: [
            "d ${inst.cacheDir} 0750 ${cfg.user} ${cfg.group} - -"
            "d ${inst.cacheDir}/build 0750 ${cfg.user} ${cfg.group} - -"
            "d ${buildCacheDirFor inst} 0750 ${cfg.user} ${cfg.group} - -"
          ])
          enabledInstances)
        ++ lib.optional (cfg.sharedHfCache != null)
        "d ${cfg.sharedHfCache} 2775 ${cfg.user} ${cfg.sharedHfCacheGroup} - -";

      systemd.services = lib.mapAttrs' mkUnit enabledInstances;
    })
  ];
}
