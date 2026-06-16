# Narrow the vllm-xpu-kernels checkout to just the files the build actually
# consumes. Without this, the whole repo (docs/, tests/, benchmark/,
# README.md, Dockerfile.xpu, tools/) feeds configureDrv's input hash, so
# upstream README/test/CI edits bump configureDrv's store path and cascade
# through every per-TU drv.
#
# `lib.fileset.toSource` would be the idiomatic API for this kind of
# narrowing, but its `root` must be an eval-time path (it rejects store-path
# strings under pure-eval), and flake inputs arrive as store paths.
# `lib.sources.sourceByRegex` filters via `builtins.filterSource`, which
# accepts store paths and produces an equivalent narrowed source.
#
# Patterns cover every file referenced by the cmake graph or read by
# `pip install`, including everything touched by `nix/patches/000*-*.patch`
# (CMakeLists.txt, cmake/utils.cmake, setup.py,
# vllm_xpu_kernels/__init__.py, csrc/xpu/attn/xe_2/*). tools/ is required
# because setup.py loads tools/envs.py to read VLLM_TARGET_DEVICE.
# third_party/ stays because cmake/Modules/FindoneDNN.cmake reads
# third_party/oneDNN (the project's own Find module, not oneAPI).
{ lib }:
rawSrc:
  lib.sources.sourceByRegex rawSrc [
    "^CMakeLists\\.txt$"
    "^cmake(/.*)?$"
    "^csrc(/.*)?$"
    "^setup\\.py$"
    "^pyproject\\.toml$"
    "^tools(/.*)?$"
    "^vllm_xpu_kernels(/.*)?$"
    "^third_party(/.*)?$"
  ]
