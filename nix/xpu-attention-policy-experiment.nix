# Replace only the Xe2 attention DSO in a frozen profiling process.
{
  kernelsSrc ? null,
  screenOnly ? true,
  aotDevices ? [ "bmg" ],
}:
let
  flake = builtins.getFlake (toString ../.);
  inherit (flake.inputs.nixpkgs) lib;
  control =
    (flake.packages.${builtins.currentSystem}.vllm-xpu-kernels-unstable.override {
      inherit aotDevices;
    }).kernelLibraries.attn-kernels-xe-2;
  narrow = import ./lib/kernels-src.nix { inherit lib; };
  project = import ./lib/kernel-lib-src.nix { inherit lib; };
  prefillConfig = builtins.toFile "issue11-prefill.conf" ''
    256,false,true,false,false,false
    256,true,true,false,false,false
  '';
  decodeConfig = builtins.toFile "issue11-decode.conf" ''
    8,256,64,true,false,false
    8,256,64,false,false,false
  '';
in
control.overrideAttrs (old: {
  src =
    if kernelsSrc == null then
      old.src
    else
      project {
        src = narrow (builtins.toPath kernelsSrc);
        libName = "attn_kernels_xe_2";
      };
  cmakeFlags =
    if !screenOnly then
      old.cmakeFlags
    else
      (lib.filter (
        flag:
        !(
          lib.hasPrefix "-DVLLM_CHUNK_PREFILL_CONFIG=" flag
          || lib.hasPrefix "-DVLLM_PAGED_DECODE_CONFIG=" flag
        )
      ) old.cmakeFlags)
      ++ [
        "-DVLLM_CHUNK_PREFILL_CONFIG=${prefillConfig}"
        "-DVLLM_PAGED_DECODE_CONFIG=${decodeConfig}"
      ];
  preBuild = "export NIX_BUILD_CORES=4";
  version = "0.1.14.1+d256-policy-experiment";
})
