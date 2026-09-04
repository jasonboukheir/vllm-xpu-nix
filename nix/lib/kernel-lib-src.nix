# Give each split kernel library only its target-specific source subtree plus
# the repository-wide sources left by kernels-src.nix.  This is deliberately a
# narrow cache contract: a change confined to one target subtree does not alter
# sibling source projections.  Shared CMake files, common headers, setup files,
# and every other source retained by kernels-src.nix remain in every projection
# and therefore invalidate every affected split library when they change.
#
# Do not describe this as general per-feature independence.  In addition to the
# shared-source rule above, mk-kernels.nix chains the memory-heavy derivations in
# one direction to keep the daemon from compiling them concurrently.  A changed
# predecessor invalidates its successors even when their projected source is
# unchanged; attention is intentionally last because it is the active iteration
# target.
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
