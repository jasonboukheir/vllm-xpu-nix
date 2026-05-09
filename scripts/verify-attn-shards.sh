#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<EOF
verify-attn-shards.sh — sanity check libattn_kernels_xe_2.so after a sharded build.

Usage:
  $0 [path-to-libattn_kernels_xe_2.so]      (default: ./result/lib/libattn_kernels_xe_2.so)
  $0 --reference <baseline.so> <new.so>     compare two builds (e.g. pre-split vs sharded)
EOF
  exit 1
}

count_static_initializers() {
  nm --defined-only "$1" 2>/dev/null | grep -c '_GLOBAL__sub_I_' || true
}

# oneAPI 2025.3 final-link layout: the AOT'd SPIR-V binary lives in
# OFFLOAD_DEVICE_CODE, with .tgtimg as the device-image header and .tgtsym
# as the offload symbol table. Earlier compilers used __CLANG_OFFLOAD_BUNDLE__.
count_sycl_offload_sections() {
  readelf -W -S "$1" 2>/dev/null \
    | grep -cE 'OFFLOAD_DEVICE_CODE|__CLANG_OFFLOAD_BUNDLE__|\.tgtimg|\.tgtsym|sycl_offloading' \
    || true
}

device_code_size() {
  readelf -W -S "$1" 2>/dev/null \
    | awk '/OFFLOAD_DEVICE_CODE/ { print "0x" $6 }' \
    | head -1 \
    | xargs -I{} printf '%d\n' {} 2>/dev/null \
    | numfmt --to=iec 2>/dev/null \
    || echo "absent"
}

has_sycl_register_lib() {
  local n
  n=$(nm "$1" 2>/dev/null | grep -c '__sycl_register_lib' || true)
  if [[ "$n" -gt 0 ]]; then echo "yes"; else echo "no"; fi
}

so_size_human() {
  stat -c '%s' "$1" | numfmt --to=iec
}

list_dt_needed() {
  readelf -d "$1" 2>/dev/null | awk '/NEEDED/ {gsub(/\[|\]/,"",$NF); print $NF}'
}

check_runtime_closure_no_shards() {
  local so="$1"
  local store_path
  store_path="$(realpath "$so")"
  if [[ ! "$store_path" =~ /nix/store/ ]]; then
    echo "  (skipped: $so not inside the Nix store, runtime closure check needs a store path)"
    return 0
  fi
  local closure_drv="${store_path%%/lib/*}"
  local shard_refs
  shard_refs=$(nix-store --query --requisites "$closure_drv" 2>/dev/null | grep -c 'vllm-xpu-attn-shard-' || true)
  if [[ "$shard_refs" -gt 0 ]]; then
    echo "  WARN: runtime closure of $closure_drv includes $shard_refs shard derivations"
    echo "  (shards should be build-time-only — investigate with: nix-store --query --references $closure_drv)"
  else
    echo "  OK: no shard derivations in runtime closure of $closure_drv"
  fi
}

inspect() {
  local so="$1"
  echo "=== $so ==="
  if [[ ! -f "$so" ]]; then
    echo "  ERROR: file does not exist"
    return 1
  fi
  echo "  size:                  $(so_size_human "$so")"
  echo "  static initializers:   $(count_static_initializers "$so")  (legacy metric — empty in oneAPI 2025+, see __sycl_register_lib)"
  echo "  SYCL/offload sections: $(count_sycl_offload_sections "$so")"
  echo "  device code section:   $(device_code_size "$so")  (OFFLOAD_DEVICE_CODE — AOT'd SPIR-V binary)"
  echo "  __sycl_register_lib:   $(has_sycl_register_lib "$so")"
  echo "  DT_NEEDED:"
  list_dt_needed "$so" | sed 's/^/    /'
  check_runtime_closure_no_shards "$so"
  echo
}

if [[ "${1:-}" == "--reference" ]]; then
  shift
  [[ $# -eq 2 ]] || usage
  inspect "$1"
  inspect "$2"
  echo "=== diff ==="
  echo "  static initializers: $(count_static_initializers "$1") -> $(count_static_initializers "$2")"
  echo "  SYCL sections:       $(count_sycl_offload_sections "$1") -> $(count_sycl_offload_sections "$2")"
  echo "  size:                $(so_size_human "$1") -> $(so_size_human "$2")"
else
  inspect "${1:-./result/lib/libattn_kernels_xe_2.so}"
fi
