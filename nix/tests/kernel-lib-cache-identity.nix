{ pkgs }:
let
  inherit (pkgs) lib;
  mkKernelsSrc = import ../lib/kernels-src.nix { inherit lib; };
  mkKernelLibSrc = import ../lib/kernel-lib-src.nix { inherit lib; };

  libraryNames = [
    "gdn_attn_kernels_xe_2"
    "grouped_gemm_xe_default"
    "grouped_gemm_xe_2"
    "mhc_kernels_xe_2"
    "mqa_logits_kernels_xe_2"
    "attn_kernels_xe_2"
  ];

  # Exercise the production source projections directly. This does not model
  # the build dependency graph or claim that sibling builds are independent.
  paths =
    fixture:
    let
      narrowedSource = mkKernelsSrc fixture;
    in
    lib.genAttrs libraryNames (
      libName:
      toString (mkKernelLibSrc {
        src = narrowedSource;
        inherit libName;
      })
    );

  changed = left: right: lib.filter (name: left.${name} != right.${name}) libraryNames;

  baseline = paths ./fixtures/kernel-lib-cache/baseline;
  testOnly = paths ./fixtures/kernel-lib-cache/test-only;
  # This fixture changes only the attention target's own CMakeLists.txt. Shared
  # helpers under cmake/ are intentionally covered by the separate common case.
  attentionOnly = paths ./fixtures/kernel-lib-cache/attention-only;
  sharedCmake = paths ./fixtures/kernel-lib-cache/shared-cmake;
  gdnOnly = paths ./fixtures/kernel-lib-cache/gdn-only;

  observations = {
    testOnlyChanged = changed baseline testOnly;
    attentionOnlyChanged = changed baseline attentionOnly;
    sharedCmakeChanged = changed baseline sharedCmake;
    gdnOnlyChanged = changed baseline gdnOnly;
  };
in
assert observations.testOnlyChanged == [ ];
assert observations.attentionOnlyChanged == [ "attn_kernels_xe_2" ];
assert observations.sharedCmakeChanged == libraryNames;
assert observations.gdnOnlyChanged == [ "gdn_attn_kernels_xe_2" ];
pkgs.writeText "kernel-lib-cache-identity.json" (builtins.toJSON observations)
