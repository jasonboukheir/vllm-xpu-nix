# Wrap an existing immutable environment without rebuilding its model/kernel packages.
{
  candidateEnv,
  ptiDir,
  enableMetrics ? false,
  pkgs ? (builtins.getFlake (toString ../.)).inputs.nixpkgs.legacyPackages.${builtins.currentSystem},
}:
let
  baseline = builtins.storePath candidateEnv;
  pti = builtins.storePath ptiDir;
  metrics = import ./xpu-metrics-libs.nix { inherit pkgs; };
  loaderPath = pkgs.lib.concatStringsSep ":" (
    [ "${pti}/lib" ]
    ++ pkgs.lib.optionals enableMetrics [ "${metrics.library}/lib" "${metrics.discovery}/lib" ]
  );
in
pkgs.runCommand "vllm-xpu-profile-env" {
  nativeBuildInputs = [ pkgs.makeWrapper ];
} ''
  mkdir -p "$out/bin"
  for name in vllm python python3 python3.12; do
    if [ -x "${baseline}/bin/$name" ]; then
      makeWrapper "${baseline}/bin/$name" "$out/bin/$name" \
        --prefix LD_LIBRARY_PATH : "${loaderPath}"
    fi
  done
  test -x "$out/bin/vllm"
  test -x "$out/bin/python"
''
