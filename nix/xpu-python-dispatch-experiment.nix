# Build matched service environments while changing only vLLM Python sources.
# Pass the control and candidate worktrees in separate invocations. Both must
# use the same baselinePackage, which owns all dependency and native DSO pins.
# XPU vLLM has no native ext_modules; nix/vllm-xpu.nix installs its Python wheel.
{
  vllmSrc ? null,
  baselinePackage ? null,
  version ? null,
}:
let
  flake = builtins.getFlake (toString ../.);
  system = builtins.currentSystem;
  pkgs = import flake.inputs.nixpkgs {
    inherit system;
    config.allowUnfree = true;
    overlays = [ (import ./python-test-workarounds-overlay.nix) ];
  };
  inherit (pkgs) lib;
  baseline =
    if baselinePackage == null then
      flake.packages.${system}.vllm-xpu-kvarn-validation
    else
      baselinePackage;
  sourceOverrides = lib.optionalAttrs (vllmSrc != null) {
    src = lib.cleanSourceWith {
      name = "vllm-python-dispatch-src";
      src = builtins.toPath vllmSrc;
      filter =
        path: type:
        lib.cleanSourceFilter path type
        && !(builtins.elem (builtins.baseNameOf path) [
          ".venv"
          ".pytest_cache"
          ".ruff_cache"
          ".mypy_cache"
          "__pycache__"
        ]);
    };
  };
  vllmPackage = baseline.override (
    sourceOverrides // lib.optionalAttrs (version != null) { inherit version; }
  );
  pythonEnv = pkgs.python312.withPackages (_: [
    vllmPackage
    pkgs.python312Packages.pytest
  ]);
  # Match the packaged validation environment: Python and its spawned model
  # inspection/worker processes need the same modules and device runtime as
  # bin/vllm. withPackages owns PYTHONPATH; the package argument uses an output
  # placeholder that must not be expanded against this enclosing environment.
  runtimeWrapperArgs = builtins.filter (
    arg: !(lib.hasPrefix "--prefix PYTHONPATH " arg)
  ) vllmPackage.makeWrapperArgs;
in
assert toString vllmPackage.kernelPackage == toString baseline.kernelPackage;
pkgs.symlinkJoin {
  name = "vllm-xpu-python-dispatch-env";
  paths = [ pythonEnv ];
  nativeBuildInputs = [ pkgs.makeWrapper ];
  postBuild = ''
    rm -f "$out/bin/python" "$out/bin/python3" "$out/bin/python3.12"
    makeWrapper \
      ${pythonEnv}/bin/python \
      "$out/bin/python" \
      ${builtins.concatStringsSep " " runtimeWrapperArgs}
    ln -s python "$out/bin/python3"
    ln -s python "$out/bin/python3.12"
  '';
  passthru = {
    inherit vllmPackage;
    baselinePackage = baseline;
    kernelPackage = vllmPackage.kernelPackage;
  };
}
