{
  config,
  lib,
  pkgs,
  ...
}: let
  cfg = config.services.hf-cache;
  rootModule = lib.types.submodule {
    options = {
      type = lib.mkOption {
        type = lib.types.enum ["model" "dataset" "space"];
        default = "model";
        description = "Hugging Face repository type.";
      };
      repo = lib.mkOption {
        type = lib.types.str;
        description = "Hugging Face repository id, such as `owner/name`.";
      };
      revision = lib.mkOption {
        type = lib.types.nullOr (lib.types.strMatching "[0-9a-fA-F]{40}");
        default = null;
        description = ''
          Immutable Hugging Face commit to retain. Null roots every cached
          revision of the repository.
        '';
      };
      source = lib.mkOption {
        type = lib.types.nullOr lib.types.str;
        default = null;
        description = "Optional description of the service contributing this root.";
      };
    };
  };
  manifest = {
    version = 1;
    roots = cfg.roots;
  };
  manifestFile = pkgs.writeText "hf-cache-roots.json" (builtins.toJSON manifest);
  gcPython = pkgs.python3.withPackages (ps: [ps.huggingface-hub]);
  gc = pkgs.writeShellApplication {
    name = "hf-cache-gc";
    runtimeInputs = [gcPython];
    text = ''
      export HF_CACHE_GC_CACHE_DIR=${lib.escapeShellArg "${cfg.home}/hub"}
      exec python ${../hf-cache-gc.py} "$@"
    '';
  };
in {
  options.services.hf-cache = {
    home = lib.mkOption {
      type = lib.types.path;
      default = "/var/cache/huggingface";
      description = "Shared `HF_HOME` whose `hub` cache is managed.";
    };
    roots = lib.mkOption {
      type = lib.types.listOf rootModule;
      default = [];
      description = ''
        Hugging Face repositories and revisions rooted by this NixOS
        generation. Definitions from multiple services are merged.
      '';
    };
  };

  config = {
    environment.etc."huggingface/cache-roots.json".source = manifestFile;
    environment.systemPackages = [gc];
  };
}
