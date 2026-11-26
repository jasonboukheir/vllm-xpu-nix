# Realize only the store paths referenced by the selected unit and manifests.
{
  trialPath,
  trialRev,
  hostPath,
  hostRev,
  phasedDecode ? null,
}:
let
  trial = builtins.getFlake ("git+file://" + trialPath + "?rev=" + trialRev);
  hostFlake = builtins.getFlake ("git+file://" + hostPath + "?rev=" + hostRev);
  host = hostFlake.nixosConfigurations.brutus;
  pkgs = host.pkgs;
  inherit (pkgs) lib;
  runtime = trial.packages.x86_64-linux.vllm-xpu-inkling2-kvarn;
  recipePath = trial.outPath + "/experiments/inkling-b70/brutus-inkling2.nix";
  recipe = import recipePath (
    { inherit runtime; } // lib.optionalAttrs (phasedDecode != null) { inherit phasedDecode; }
  );
  normal = trial.inputs.nixpkgs.lib.nixosSystem {
    system = "x86_64-linux";
    modules = [
      trial.nixosModules.vllm-xpu
      {
        system.stateVersion = host.config.system.stateVersion;
        services.vllm-xpu.instances.chat = recipe.instance;
      }
    ];
  };
  preview = host.extendModules { modules = [ recipe.brutusModule ]; };
  cfg = preview.config;
  manifest = config: config.environment.etc."huggingface/cache-roots.json".source;
  command = config: config.systemd.services.vllm-xpu-chat.serviceConfig.ExecStart;
  catalog = cfg.services.local-inference.models.chat;
in
assert command normal.config == command cfg;
pkgs.writeText "brutus-inkling-recipe-preview.json" (
  builtins.toJSON {
    runtime = toString runtime;
    unit = "${cfg.systemd.units."vllm-xpu-chat.service".unit}/vllm-xpu-chat.service";
    inherit (recipe.instance)
      model
      revision
      servedName
      maxModelLen
      gpuMemoryUtilization
      ;
    environment = builtins.intersectAttrs (
      recipe.instance.extraEnvironment
      // {
        HF_HOME = null;
        VLLM_CACHE_ROOT = null;
      }
    ) cfg.systemd.services.vllm-xpu-chat.environment;
    command = command cfg;
    manifests = {
      normal = manifest normal.config;
      brutus = manifest cfg;
    };
    currentHostModel = host.config.services.vllm-xpu.instances.chat.model;
    catalog = {
      inherit (catalog)
        maxInputTokens
        maxOutputTokens
        chatTemplate
        servedName
        supportedReasoningEfforts
        ;
    };
    hostRevision = hostFlake.rev;
    trialRevision = trial.rev;
    recipeSha256 = builtins.hashFile "sha256" recipePath;
  }
)
