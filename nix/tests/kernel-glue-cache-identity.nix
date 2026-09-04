{
  pkgs,
  kernelSource,
}:
let
  inherit (pkgs) lib;
  mkKernelsSrc = import ../lib/kernels-src.nix { inherit lib; };
  mkKernelGlueSrc = import ../lib/kernel-glue-src.nix { inherit lib; };

  paths =
    fixture:
    let
      src = mkKernelsSrc fixture;
    in
    {
      base = toString (mkKernelGlueSrc {
        inherit src;
        component = "base";
      });
      fa2 = toString (mkKernelGlueSrc {
        inherit src;
        component = "fa2";
      });
    };
  changed =
    left: right:
    lib.filter (component: left.${component} != right.${component}) [
      "base"
      "fa2"
    ];

  baseline = paths ./fixtures/kernel-lib-cache/baseline;
  testOnly = paths ./fixtures/kernel-lib-cache/test-only;
  attentionImplementation = paths ./fixtures/kernel-lib-cache/attention-only;
  sharedCmake = paths ./fixtures/kernel-lib-cache/shared-cmake;
  nonAttentionImplementation = paths ./fixtures/kernel-lib-cache/gdn-only;

  productionFa2 = mkKernelGlueSrc {
    src = kernelSource;
    component = "fa2";
  };
  productionBase = mkKernelGlueSrc {
    src = kernelSource;
    component = "base";
  };

  observations = {
    testOnlyChanged = changed baseline testOnly;
    attentionImplementationChanged = changed baseline attentionImplementation;
    sharedCmakeChanged = changed baseline sharedCmake;
    nonAttentionImplementationChanged = changed baseline nonAttentionImplementation;
    fa2ContainsBinding = builtins.pathExists "${productionFa2}/csrc/flash_attn/flash_api.cpp";
    fa2ContainsPublicAbi = builtins.pathExists "${productionFa2}/csrc/xpu/attn/xe_2/kvarn_decode_xe2.h";
    fa2ExcludesPrivateImplementation =
      !(builtins.pathExists "${productionFa2}/csrc/xpu/attn/xe_2/kvarn_decode.hpp");
    baseExcludesFa2 = !(builtins.pathExists "${productionBase}/csrc/flash_attn/flash_api.cpp");
  };
in
assert observations.testOnlyChanged == [ ];
assert observations.attentionImplementationChanged == [ ];
assert
  observations.sharedCmakeChanged == [
    "base"
    "fa2"
  ];
assert observations.nonAttentionImplementationChanged == [ "base" ];
assert observations.fa2ContainsBinding;
assert observations.fa2ContainsPublicAbi;
assert observations.fa2ExcludesPrivateImplementation;
assert observations.baseExcludesFa2;
pkgs.writeText "kernel-glue-cache-identity.json" (builtins.toJSON observations)
