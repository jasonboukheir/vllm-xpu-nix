{
  description = "Nix-native Intel XPU substrate for vLLM (torch+xpu, triton-xpu, vllm-xpu-kernels, vllm)";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachSystem [ "x86_64-linux" ] (system:
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

        oneccl-bmg = pkgs.callPackage ./nix/oneccl-bmg.nix {
          intel-oneapi-base = intel-oneapi;
        };
      in {
        packages = {
          inherit intel-oneapi oneccl-bmg;
          default = intel-oneapi;
        };

        devShells.default = pkgs.mkShell {
          name = "vllm-xpu-nix-dev";
          packages = with pkgs; [
            git
            nix-tree
            nix-diff
            nixfmt-rfc-style
            nil
            patchelf
            file
            skopeo
          ];
          shellHook = ''
            echo "vllm-xpu-nix dev shell. Try: nix build .#intel-oneapi --no-link --print-out-paths"
          '';
        };
      });
}
