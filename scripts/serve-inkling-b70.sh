#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 {eager|graph|dspark|cutlass|sparse} [--vision]" >&2
  exit 2
}

[[ $# -ge 1 && $# -le 2 ]] || usage
profile=$1
modality=("--language-model-only")
if [[ $# -eq 2 ]]; then
  [[ $2 == --vision ]] || usage
  modality=()
fi

package=vllm-xpu-inkling2
graphs=("-cc.mode=none" "-cc.cudagraph_mode=FULL_DECODE_ONLY" "--cudagraph-metrics")
draft=()
case "$profile" in
  eager) graphs=("--enforce-eager") ;;
  graph) ;;
  dspark|cutlass|sparse)
    draft=("--speculative-config" '{"method":"dspark","num_speculative_tokens":7}')
    if [[ $profile != dspark ]]; then
      package=vllm-xpu-inkling2-$profile
    fi
    ;;
  *) usage ;;
esac

repo=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
export HF_HOME=${HF_HOME:-/var/cache/huggingface}
snapshot="$HF_HOME/hub/models--thinkingmachines--Inkling-2-Mini/snapshots/ed40e7da625f594fc9c4990f5f66be24873c22d7"
for file in config.json tokenizer.json model-00001-of-00001.safetensors; do
  if [[ ! -s "$snapshot/$file" ]]; then
    echo "Incomplete pinned Inkling snapshot: $snapshot/$file" >&2
    exit 1
  fi
done
if [[ ${#draft[@]} -gt 0 && ! -s "$snapshot/mtp.safetensors" ]]; then
  echo "Bundled drafter missing: $snapshot/mtp.safetensors" >&2
  exit 1
fi

runtime=$(nix build --no-link --print-out-paths "$repo#$package")
export HF_HUB_OFFLINE=1
export VLLM_CACHE_ROOT="${XDG_CACHE_HOME:-$HOME/.cache}/vllm-inkling-b70/$profile"
export INK2_CUTLASS_MOE=0 INK2_SPARSE_CONV_HISTORY=0
if [[ $profile == cutlass || $profile == sparse ]]; then export INK2_CUTLASS_MOE=1; fi
if [[ $profile == sparse ]]; then
  export INK2_SPARSE_CONV_HISTORY=1
  INKLING_KV_DTYPE=${INKLING_KV_DTYPE:-fp8}
fi
exec "$runtime/bin/vllm" serve "$snapshot" \
  --host 127.0.0.1 --port "${INKLING_PORT:-18001}" --served-model-name inkling-trial \
  --dtype bfloat16 --kv-cache-dtype "${INKLING_KV_DTYPE:-auto}" --max-model-len "${INKLING_CONTEXT:-8192}" \
  --max-num-batched-tokens "${INKLING_BATCH_TOKENS:-2048}" --max-num-seqs 1 --block-size 64 \
  --gpu-memory-utilization "${INKLING_GPU_FRACTION:-0.93}" --no-enable-prefix-caching \
  --reasoning-parser inkling --enable-auto-tool-choice --tool-call-parser inkling \
  "${modality[@]}" "${graphs[@]}" "${draft[@]}"
