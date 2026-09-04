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

  # Use the production source filters, then put their results into tiny dummy
  # derivations with the same ordering graph as mkKernelLibs.  Comparing the
  # resulting drvPaths exercises both source projection and the directional
  # buildDependencies invalidation without compiling or executing any kernel.
  mkGraph =
    fixture:
    let
      narrowedSource = mkKernelsSrc fixture;
      mkNode =
        libName: orderingDependencies:
        pkgs.runCommandLocal "kernel-cache-identity-${lib.replaceStrings [ "_" ] [ "-" ] libName}"
          {
            projectedSource = mkKernelLibSrc {
              src = narrowedSource;
              inherit libName;
            };
            nativeBuildInputs = orderingDependencies;
            disallowedReferences = orderingDependencies;
          }
          ''
            test -d "$projectedSource"
            touch "$out"
          '';

      gdn = mkNode "gdn_attn_kernels_xe_2" [ ];
      groupedDefault = mkNode "grouped_gemm_xe_default" [ gdn ];
      groupedXe2 = mkNode "grouped_gemm_xe_2" [ groupedDefault ];
      mhc = mkNode "mhc_kernels_xe_2" [ groupedXe2 ];
      mqa = mkNode "mqa_logits_kernels_xe_2" [ mhc ];
      attn = mkNode "attn_kernels_xe_2" [ mqa ];
    in
    {
      gdn_attn_kernels_xe_2 = gdn;
      grouped_gemm_xe_default = groupedDefault;
      grouped_gemm_xe_2 = groupedXe2;
      mhc_kernels_xe_2 = mhc;
      mqa_logits_kernels_xe_2 = mqa;
      attn_kernels_xe_2 = attn;
    };

  paths = graph: lib.mapAttrs (_: drv: drv.drvPath) graph;
  changed = left: right: lib.filter (name: left.${name} != right.${name}) libraryNames;

  baseline = paths (mkGraph ./fixtures/kernel-lib-cache/baseline);
  testOnly = paths (mkGraph ./fixtures/kernel-lib-cache/test-only);
  # This fixture changes only the attention target's own CMakeLists.txt. Shared
  # helpers under cmake/ are intentionally covered by the separate common case.
  attentionOnly = paths (mkGraph ./fixtures/kernel-lib-cache/attention-only);
  sharedCmake = paths (mkGraph ./fixtures/kernel-lib-cache/shared-cmake);
  gdnOnly = paths (mkGraph ./fixtures/kernel-lib-cache/gdn-only);

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
# GDN is the first ordering dependency, so its source change invalidates the
# complete successor chain even though the five sibling projections are stable.
assert observations.gdnOnlyChanged == libraryNames;
pkgs.writeText "kernel-lib-cache-identity.json" (builtins.toJSON observations)
