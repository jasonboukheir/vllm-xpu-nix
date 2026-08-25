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
        type = lib.types.nullOr (lib.types.strMatching "[0-9a-f]{40}");
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
  accessAcl = lib.concatStringsSep "," [
    # Repair migrated descendants even when their owning group is stale. `X`
    # adds traversal only to directories and files already executable.
    "g:${cfg.group}:rwX"
    "m::rwX"
    # Make future descendants writable by the shared group regardless of the
    # creating process's umask.
    "d:g:${cfg.group}:rwx"
    "d:m::rwx"
  ];
  accessPolicy = pkgs.writeText "hf-cache-access-policy.json" (builtins.toJSON {
    inherit (cfg) group owner;
    home = toString cfg.home;
    inherit accessAcl;
  });
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
    enable = lib.mkEnableOption "authoritative Hugging Face cache management";
    home = lib.mkOption {
      type = lib.types.path;
      default = "/var/cache/huggingface";
      description = "Shared `HF_HOME` whose `hub` cache is managed.";
    };
    owner = lib.mkOption {
      type = lib.types.str;
      default = "root";
      description = "Owner of the shared cache root.";
    };
    group = lib.mkOption {
      type = lib.types.str;
      default = "huggingface";
      description = ''
        Group granted inherited read/write access to the shared cache.
        The module creates this group.
      '';
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

  config = lib.mkIf cfg.enable {
    environment.etc."huggingface/cache-roots.json".source = manifestFile;
    environment.systemPackages = [gc];

    users.groups.${cfg.group} = {};

    # `d` establishes the root and setgid ownership. The recursive named-group
    # ACL repairs existing cache content and its default entries make the
    # policy independent of each writer's umask.
    systemd.tmpfiles.settings."10-hf-cache".${toString cfg.home} = {
      d = {
        mode = "2775";
        user = cfg.owner;
        group = cfg.group;
      };
      "A+".argument = accessAcl;
    };

    # Run the cache-specific policy only after the real backing mount is
    # available. This avoids repairing an underlying root-filesystem directory
    # when `home` is an automounted bind path.
    systemd.services.hf-cache-prepare = {
      description = "Prepare the shared Hugging Face cache";
      wantedBy = ["multi-user.target"];
      unitConfig.RequiresMountsFor = [cfg.home];
      restartTriggers = [accessPolicy];
      serviceConfig = {
        Type = "oneshot";
        ExecStart = "${pkgs.systemd}/bin/systemd-tmpfiles --create /etc/tmpfiles.d/10-hf-cache.conf";
      };
    };
  };
}
