# Give each split kernel library only the target-specific source subtree it can
# compile, plus the repository-wide build/common sources.  The caller already
# narrows the raw checkout with kernels-src.nix; this second filter removes
# sibling kernel families so an attention-only change does not invalidate GDN,
# MQA, MHC, or grouped-GEMM derivations.
{ lib }:
let
  targetRoots = {
    attn_kernels_xe_2 = "csrc/xpu/attn";
    gdn_attn_kernels_xe_2 = "csrc/xpu/gdn_attn";
    mqa_logits_kernels_xe_2 = "csrc/xpu/mqa_logits";
    mhc_kernels_xe_2 = "csrc/xpu/mhc";
    # Both grouped-GEMM targets consume common files under collective/.
    grouped_gemm_xe_2 = "csrc/xpu/grouped_gemm";
    grouped_gemm_xe_default = "csrc/xpu/grouped_gemm";
  };

  isUnder = root: path: lib.hasSuffix "/${root}" path || lib.hasInfix "/${root}/" path;
in
{
  src,
  libName,
}:
let
  selectedRoot =
    targetRoots.${libName} or (throw "kernel-lib-src.nix: unknown split kernel library ${libName}");
  excludedRoots = lib.filter (root: root != selectedRoot) (lib.unique (lib.attrValues targetRoots));
in
lib.cleanSourceWith {
  name = "vllm-xpu-kernels-${lib.replaceStrings [ "_" ] [ "-" ] libName}-source";
  inherit src;
  filter =
    path: _type:
    let
      pathString = toString path;
    in
    !lib.any (root: isUnder root pathString) excludedRoots;
}
