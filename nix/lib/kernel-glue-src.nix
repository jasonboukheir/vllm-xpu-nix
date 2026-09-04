# Source projections for the native Python-extension layer that links the
# independently-built SYCL-TLA libraries.
#
# `base` deliberately excludes FlashAttention's binding and implementation
# sources.  An attention-only implementation change can therefore reuse the
# expensive `_C`, `_moe_C`, `_xpu_C`, and allocator extensions.
#
# `fa2` contains only the `_vllm_fa2_C` binding sources and the public attention
# headers they compile against.  Private Xe2 `.hpp`/`.cpp` implementation
# changes leave this source identity stable, but the changing prebuilt attention
# DSO remains a direct derivation input and conservatively forces a small relink.
{ lib }:
{
  src,
  component,
}:
if component == "base" then
  lib.cleanSourceWith {
    name = "vllm-xpu-kernels-base-glue-source";
    inherit src;
    filter =
      path: _type:
      let
        pathString = toString path;
        isUnder = root: lib.hasSuffix "/${root}" pathString || lib.hasInfix "/${root}/" pathString;
      in
      !(isUnder "csrc/flash_attn" || isUnder "csrc/xpu/attn");
  }
else if component == "fa2" then
  lib.sources.sourceByRegex src [
    "^CMakeLists\\.txt$"
    "^cmake(/.*)?$"
    "^setup\\.py$"
    "^pyproject\\.toml$"
    "^tools(/.*)?$"
    "^vllm_xpu_kernels(/.*)?$"
    # sourceByRegex must retain the directory spine before it can inspect the
    # selected files below.
    "^csrc$"
    "^csrc/sycl_first\\.h$"
    "^csrc/utils\\.h$"
    "^csrc/core(/.*)?$"
    "^csrc/flash_attn(/.*)?$"
    "^csrc/xpu$"
    "^csrc/xpu/attn$"
    "^csrc/xpu/attn/attn_interface\\.(cpp|h)$"
    "^csrc/xpu/attn/paged_kv_utils\\.h$"
    # The binding includes only the declaration headers in xe_2. Private
    # implementation headers use .hpp and belong solely to the split DSO.
    "^csrc/xpu/attn/xe_2$"
    "^csrc/xpu/attn/xe_2/[^/]+\\.h$"
  ]
else
  throw "kernel-glue-src.nix: unknown component ${component}"
