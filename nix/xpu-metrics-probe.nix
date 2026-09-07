{
  pkgs ? (builtins.getFlake (toString ../.)).inputs.nixpkgs.legacyPackages.${builtins.currentSystem},
}:
pkgs.stdenv.mkDerivation {
  pname = "kvarn-xpu-metrics-probe";
  version = "1";
  src = ../scripts/kvarn_xpu_metrics_probe.cpp;
  dontUnpack = true;
  buildInputs = [ pkgs.level-zero ];
  buildPhase = ''
    $CXX -std=c++17 -Wall -Wextra -Werror -O2 "$src" -lze_loader -o metrics-probe
  '';
  installPhase = ''
    mkdir -p "$out/bin"
    cp metrics-probe "$out/bin/metrics-probe"
  '';
}
