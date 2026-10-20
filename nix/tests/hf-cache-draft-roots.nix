{ pkgs, nixpkgs }:
let
  pinned = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
  instance = model: revision: {
    enable = true;
    model = "example/target";
    speculativeConfig = {
      inherit model revision;
    };
  };
  evaluated = import (nixpkgs + "/nixos/lib/eval-config.nix") {
    system = pkgs.stdenv.hostPlatform.system;
    modules = [
      ../modules/vllm-xpu.nix
      {
        services.vllm-xpu.package = pkgs.emptyDirectory;
        services.vllm-xpu.instances = {
          pinned = instance "example/pinned" pinned;
          branch = instance "example/branch" "main";
          tag = instance "example/tag" "v2";
          single = instance "gpt2" null;
          relative = instance "models/dflash" null;
          absolute = instance "/srv/models/dflash" null;
          dot = instance "./dflash" null;
          parent = instance "../dflash" null;
          disabled = (instance "unused/draft" null) // {
            enable = false;
          };
        };
      }
    ];
  };
  manifest = pkgs.writeText "evaluated-draft-roots.json" (
    builtins.toJSON {
      roots = evaluated.config.services.hf-cache.roots;
      serving = pkgs.lib.mapAttrs (
        _: inst: inst.speculativeConfig
      ) evaluated.config.services.vllm-xpu.instances;
    }
  );
  python = pkgs.python312.withPackages (ps: [ ps.huggingface-hub ]);
in
pkgs.runCommand "hf-cache-draft-roots"
  {
    nativeBuildInputs = [ python ];
    HF_CACHE_TEST_ROOTS = manifest;
  }
  ''
    cd ${../..}
    python -m unittest discover -s tests -p test_hf_cache_gc.py -v
    cp ${manifest} $out
  ''
