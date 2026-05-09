{ config, lib, pkgs, ... }:

let
  cfg = config.services.vllm-xpu;

  bmgCclEnv = {
    CCL_PROCESS_LAUNCHER = "none";
    CCL_ATL_TRANSPORT = "ofi";
    CCL_ZE_IPC_EXCHANGE = "sockets";
    CCL_LOG_LEVEL = "warn";
  };

  serveArgs = lib.concatStringsSep " " (
    [
      (lib.escapeShellArg cfg.model)
      "--host" (lib.escapeShellArg cfg.host)
      "--port" (toString cfg.port)
      "--served-model-name" (lib.escapeShellArg cfg.servedName)
      "--dtype" (lib.escapeShellArg cfg.dtype)
      "--gpu-memory-utilization" (toString cfg.gpuMemoryUtilization)
    ]
    ++ lib.optionals (cfg.quantization != null) [ "--quantization" (lib.escapeShellArg cfg.quantization) ]
    ++ lib.optionals (cfg.kvCacheDtype != null) [ "--kv-cache-dtype" (lib.escapeShellArg cfg.kvCacheDtype) ]
    ++ lib.optionals (cfg.maxModelLen != null) [ "--max-model-len" (toString cfg.maxModelLen) ]
    ++ lib.optionals cfg.enforceEager [ "--enforce-eager" ]
    ++ map lib.escapeShellArg cfg.extraArgs
  );
in
{
  options.services.vllm-xpu = {
    enable = lib.mkEnableOption "native vLLM XPU serve daemon";

    package = lib.mkOption {
      type = lib.types.package;
      description = "vLLM-XPU package providing bin/vllm.";
    };

    model = lib.mkOption {
      type = lib.types.str;
      description = "HuggingFace model id or local path served by vllm.";
    };

    servedName = lib.mkOption {
      type = lib.types.str;
      description = "Name advertised over the OpenAI-compatible API.";
    };

    host = lib.mkOption {
      type = lib.types.str;
      default = "127.0.0.1";
      description = "Address vllm binds the API server to.";
    };

    port = lib.mkOption {
      type = lib.types.port;
      default = 8000;
      description = "TCP port for the OpenAI-compatible API server.";
    };

    dtype = lib.mkOption {
      type = lib.types.str;
      default = "bfloat16";
      description = "Inference dtype passed to vllm serve.";
    };

    quantization = lib.mkOption {
      type = lib.types.nullOr lib.types.str;
      default = null;
      example = "inc";
      description = "Optional --quantization argument.";
    };

    kvCacheDtype = lib.mkOption {
      type = lib.types.nullOr lib.types.str;
      default = null;
      example = "fp8_e5m2";
      description = "Optional --kv-cache-dtype argument.";
    };

    maxModelLen = lib.mkOption {
      type = lib.types.nullOr lib.types.int;
      default = null;
      description = "Optional --max-model-len argument.";
    };

    gpuMemoryUtilization = lib.mkOption {
      type = lib.types.float;
      default = 0.9;
      description = "Fraction of XPU memory vllm may use.";
    };

    enforceEager = lib.mkOption {
      type = lib.types.bool;
      default = false;
      description = "Pass --enforce-eager (skip torch.compile graph capture).";
    };

    cacheDir = lib.mkOption {
      type = lib.types.path;
      default = "/var/lib/vllm";
      description = "State directory used as HF_HOME and VLLM_CACHE_ROOT.";
    };

    user = lib.mkOption {
      type = lib.types.str;
      default = "vllm";
      description = "Service unit user.";
    };

    group = lib.mkOption {
      type = lib.types.str;
      default = "vllm";
      description = "Service unit group.";
    };

    cclEnv = lib.mkOption {
      type = lib.types.attrsOf lib.types.str;
      default = bmgCclEnv;
      description = "oneCCL env vars exported into the systemd unit. The default is the safe BMG single-card config.";
    };

    environmentFile = lib.mkOption {
      type = lib.types.nullOr lib.types.path;
      default = null;
      description = "Optional EnvironmentFile (e.g. for HF_TOKEN secrets).";
    };

    extraArgs = lib.mkOption {
      type = lib.types.listOf lib.types.str;
      default = [];
      description = "Extra arguments appended after the built-in flags on the vllm serve command line.";
    };
  };

  config = lib.mkIf cfg.enable {
    users.users.${cfg.user} = {
      isSystemUser = true;
      group = cfg.group;
      home = cfg.cacheDir;
      createHome = false;
      extraGroups = [ "render" "video" ];
    };
    users.groups.${cfg.group} = { };

    systemd.tmpfiles.rules = [
      "d ${cfg.cacheDir} 0750 ${cfg.user} ${cfg.group} - -"
    ];

    systemd.services.vllm-xpu = {
      description = "vLLM XPU serve (${cfg.servedName})";
      wantedBy = [ "multi-user.target" ];
      after = [ "network-online.target" ];
      wants = [ "network-online.target" ];

      environment = {
        VLLM_TARGET_DEVICE = "xpu";
        HF_HOME = cfg.cacheDir;
        VLLM_CACHE_ROOT = cfg.cacheDir;
      } // cfg.cclEnv;

      serviceConfig = {
        Type = "simple";
        User = cfg.user;
        Group = cfg.group;
        SupplementaryGroups = [ "render" "video" ];

        ExecStart = "${cfg.package}/bin/vllm serve ${serveArgs}";

        Restart = "on-failure";
        RestartSec = 5;

        # /dev/dri access for Intel render nodes; vllm needs read+write on
        # the render node and read on the card node for clinfo-style probes.
        DeviceAllow = [
          "char-drm rw"
        ];
        PrivateDevices = false;

        StateDirectory = "vllm";
        WorkingDirectory = cfg.cacheDir;

        ProtectSystem = "strict";
        ProtectHome = true;
        NoNewPrivileges = true;
        ReadWritePaths = [ cfg.cacheDir ];
      } // lib.optionalAttrs (cfg.environmentFile != null) {
        EnvironmentFile = cfg.environmentFile;
      };
    };
  };
}
