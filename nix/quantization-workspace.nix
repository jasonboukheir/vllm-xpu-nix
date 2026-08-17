{
  pkgs,
  quantize,
}: {workspace ? null}: let
  python = pkgs.python312.withPackages (ps: [
    ps.huggingface-hub
    ps.hf-xet
  ]);
  shell = pkgs.mkShell {
    packages = [
      quantize
      python
      pkgs.git
      pkgs.git-lfs
      pkgs.jq
    ];
    shellHook = ''
      export HF_HOME="''${HF_HOME:-/var/cache/huggingface}"
      export HF_TOKEN_PATH="''${HF_TOKEN_PATH:-$HOME/.config/huggingface/token}"
      export QUANTIZATION_WORKSPACE="''${QUANTIZATION_WORKSPACE:-$PWD}"
    '';
  };
  exportApp = pkgs.writeShellApplication {
    name = "quantize-export";
    runtimeInputs = [quantize];
    text = ''exec quantize export "$@"'';
  };
in {
  packages.x86_64-linux.default = quantize;
  apps.x86_64-linux = {
    quantize = {
      type = "app";
      program = "${quantize}/bin/quantize";
    };
    export = {
      type = "app";
      program = "${exportApp}/bin/quantize-export";
    };
  };
  devShells.x86_64-linux.default = shell;
}
