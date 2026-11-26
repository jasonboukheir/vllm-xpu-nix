#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 {eager|graph|dspark|cutlass|sparse|kvarn} [--vision]" >&2
  exit 2
}

[[ $# -ge 1 && $# -le 2 ]] || usage
profile=$1
modality=("--language-model-only")
if [[ $# -eq 2 ]]; then
  [[ $2 == --vision ]] || usage
  modality=("--limit-mm-per-prompt" '{"image":2}')
fi

package=vllm-xpu-inkling2
graphs=("-cc.mode=none" "-cc.cudagraph_mode=FULL_DECODE_ONLY" "--cudagraph-metrics")
draft=()
scheduling=()
case "$profile" in
  eager) graphs=("--enforce-eager") ;;
  graph) ;;
  dspark|cutlass|sparse|kvarn)
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

if [[ -n ${INKLING_RUNTIME:-} ]]; then
  runtime=$INKLING_RUNTIME
else
  runtime=$(nix build --no-link --print-out-paths "$repo#$package")
fi
export HF_HUB_OFFLINE=1
export VLLM_CACHE_ROOT="${XDG_CACHE_HOME:-$HOME/.cache}/vllm-inkling-b70/$profile"
export INK2_CUTLASS_MOE=0 INK2_SPARSE_CONV_HISTORY=0
if [[ $profile == cutlass || $profile == sparse || $profile == kvarn ]]; then export INK2_CUTLASS_MOE=1; fi
if [[ $profile == sparse || $profile == kvarn ]]; then
  export INK2_SPARSE_CONV_HISTORY=1
  scheduling=("--no-async-scheduling")
  draft=("--speculative-config" '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}')
  INKLING_KV_DTYPE=${INKLING_KV_DTYPE:-fp8}
fi
cache_options=("--no-enable-prefix-caching")
block_size=64
if [[ $profile == kvarn ]]; then
  export VLLM_KV_CACHE_LAYOUT=BLNHC
  block_size=128
  ram_bytes=${INKLING_RAM_BYTES:-8589934592}
  [[ $ram_bytes =~ ^[0-9]+$ ]] || { echo "INKLING_RAM_BYTES must be bytes" >&2; exit 2; }
  if [[ $ram_bytes != 0 ]]; then
    connector='{"kv_connector":"OffloadingConnector","kv_role":"kv_both","kv_connector_extra_config":{"spec_name":"CPUOffloadingSpec","cpu_bytes_to_use":'
    connector+="$ram_bytes"
    connector+=',"gpu_cache_policy":"active_only","offload_prompt_only":false}}'
    cache_options=("--enable-prefix-caching" "--kv-transfer-config" "$connector")
  fi
  if [[ ${INKLING_EXTENSIBLE:-0} == 1 ]]; then
    cache_options+=("--enable-extensible-kv-cache")
  fi
fi
exec "$runtime/bin/vllm" serve "$snapshot" \
  --host 127.0.0.1 --port "${INKLING_PORT:-18001}" --served-model-name inkling-trial \
  --dtype bfloat16 --kv-cache-dtype "${INKLING_KV_DTYPE:-auto}" --max-model-len "${INKLING_CONTEXT:-8192}" \
  --max-num-batched-tokens "${INKLING_BATCH_TOKENS:-2048}" --max-num-seqs "${INKLING_SEQUENCES:-1}" --block-size "$block_size" \
  --gpu-memory-utilization "${INKLING_GPU_FRACTION:-0.93}" "${cache_options[@]}" \
  --reasoning-parser inkling --enable-auto-tool-choice --tool-call-parser inkling \
  "${modality[@]}" "${graphs[@]}" "${scheduling[@]}" "${draft[@]}"
