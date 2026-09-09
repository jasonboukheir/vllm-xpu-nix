# Build only the experimental Xe2 grouped GEMM DSO against this flake's
# pinned substrate. Its unchanged Python binding can load it through
# LD_LIBRARY_PATH in an isolated process; this does not change serving.
{
  kernelsSrc,
  aotDevices ? [ "bmg" ],
}:
let
  flake = builtins.getFlake (toString ../.);
  inherit (flake.inputs.nixpkgs) lib;
  control =
    (flake.packages.${builtins.currentSystem}.vllm-xpu-kernels-unstable.override {
      inherit aotDevices;
    }).kernelLibraries.grouped-gemm-xe-2;
  narrow = import ./lib/kernels-src.nix { inherit lib; };
  project = import ./lib/kernel-lib-src.nix { inherit lib; };
in
control.overrideAttrs {
  src = project {
    src = narrow (builtins.toPath kernelsSrc);
    libName = "grouped_gemm_xe_2";
  };
  version = "0.1.14.1+dense-int4-policies";
}
