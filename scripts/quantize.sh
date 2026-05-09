#!/usr/bin/env bash
# General-purpose AutoRound wrapper with B70-tuned defaults.
# Wrapped by nix/quantize.nix as `apps.quantize`; the auto-round-xpu env
# puts the recipe binaries (auto-round, auto-round-light, auto-round-best)
# on PATH.
set -euo pipefail

RECIPE="${AUTOROUND_QUANTIZE_RECIPE:-default}"
SEQLEN="${AUTOROUND_QUANTIZE_SEQLEN:-2048}"
BATCH_SIZE="${AUTOROUND_QUANTIZE_BS:-4}"
GRAD_ACCUM="${AUTOROUND_QUANTIZE_GA:-2}"
FORMAT="${AUTOROUND_QUANTIZE_FORMAT:-auto_round}"
OUTPUT_DIR="${AUTOROUND_OUTPUT_DIR:-$PWD/output/auto-round}"

usage() {
    cat <<'EOF'
quantize — B70-tuned AutoRound wrapper

Usage:
  quantize <model> <type> [extra auto-round flags...]
  quantize help

Arguments:
  <model>   HuggingFace model id or local path
  <type>    Quantization scheme. Maps to auto-round --scheme:
              int4    -> W4A16   (4-bit sym weights, fp16 act, group_size=128)  default
              int8    -> W8A16   (8-bit sym weights, fp16 act)
              int2    -> W2A16   (2-bit sym weights, fp16 act — quality risk)
              w4a16   -> W4A16   (alias)
              w8a16   -> W8A16   (alias)
              mxfp4   -> MXFP4   (microscaling fp4)
              nvfp4   -> NVFP4   (NVIDIA fp4)
              gguf:q4_k_m -> GGUF Q4_K_M (llama.cpp)
              gguf:q5_k_m -> GGUF Q5_K_M

Tuned defaults (Battlemage B70 30 GB, dense + hybrid-MoE friendly):
  recipe              default              200 iters, full quality
  batch_size          4
  gradient_accumulate 2                    bs=4 ga=2 = 1.51x faster than bs=1 ga=8 at matched tokens
  seqlen              2048
  low_gpu_mem_usage   OFF                  +1.30x speed at +53% VRAM (still fits dense <=35B)
  torch_compile       OFF                  measured 1.57x SLOWER on dense
  format              auto_round           vLLM/SGLang-compatible
  device              0                    first XPU
  output_dir          /output (container)  -> $AUTOROUND_OUTPUT_DIR on host

Examples:
  quantize AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-BF16 int4
  AUTOROUND_QUANTIZE_RECIPE=overnight quantize <model> int4
  quantize Qwen/Qwen3.6-35B-A3B int4 --to_quant_block_names 'model\.language_model\.layers\.0$'

Recipes (set via AUTOROUND_QUANTIZE_RECIPE=<name>):
  default     200 iters, nsamples=128, no early stop                 ~4.4h on 27B-VLM B70
              empirical: KL/token ~0.052, top-1 ~91.7% on Qwen3.6-27B
  light       50 iters, nsamples=128                                 ~1.7h on 35B-MoE B70
              ~1.3-2x higher per-block MSE; iteration / smoke runs only
  overnight   400 iters, nsamples=256, --dynamic_max_gap 100         ~8-14h on 27B-VLM B70
              early-terminates per-block when no improvement for 100 iters;
              targets the gap between default and best with patience-based
              cutoff. Use when quality matters but `best` is too slow.
              Expected ~0.02-0.04 KL/token (vs 0.052 default).
  best        1000 iters, nsamples=512                               ~3-4 days on 27B-VLM B70
              no early stopping by default; rarely worth it past `overnight`.

Other environment overrides:
  AUTOROUND_QUANTIZE_SEQLEN   calibration seqlen   (default: 2048)
  AUTOROUND_QUANTIZE_BS       per-step batch size  (default: 4)
  AUTOROUND_QUANTIZE_GA       gradient accumulate  (default: 2)
  AUTOROUND_QUANTIZE_FORMAT   auto_round | auto_gptq | auto_awq | gguf:* | llm_compressor   (default: auto_round)
  AUTOROUND_OUTPUT_DIR        host output dir base (default: $PWD/output)

Why these defaults: see `auto-round-qwen-3-6-35b-a3b help` for the full per-knob
decomposition (Qwen3-0.6B dense smokes + Qwen3.6-35B-A3B verification on B70).

EOF
}

resolve_recipe() {
    local r="$1"
    case "$r" in
        default|auto-round)
            RECIPE_BIN="auto-round"
            RECIPE_FLAGS=()
            ;;
        light|auto-round-light)
            RECIPE_BIN="auto-round-light"
            RECIPE_FLAGS=()
            ;;
        best|auto-round-best)
            RECIPE_BIN="auto-round-best"
            RECIPE_FLAGS=()
            ;;
        overnight)
            RECIPE_BIN="auto-round"
            RECIPE_FLAGS=(--iters 400 --nsamples 256 --dynamic_max_gap 100)
            ;;
        *)
            echo "error: unknown recipe '$r'" >&2
            echo "       supported: default | light | overnight | best" >&2
            exit 1
            ;;
    esac
}

scheme_for_type() {
    local raw="$1"
    local lower
    lower="$(printf '%s' "$raw" | tr '[:upper:]' '[:lower:]')"
    case "$lower" in
        int4|w4a16)  echo "W4A16" ;;
        int8|w8a16)  echo "W8A16" ;;
        int2|w2a16)  echo "W2A16" ;;
        int3|w3a16)  echo "W3A16" ;;
        mxfp4)       echo "MXFP4" ;;
        mxfp8)       echo "MXFP8" ;;
        nvfp4)       echo "NVFP4" ;;
        fp8|fp8_static) echo "FP8_STATIC" ;;
        gguf:*)
            local gguf_type="${lower#gguf:}"
            printf 'GGUF:%s' "$(printf '%s' "$gguf_type" | tr '[:lower:]' '[:upper:]')"
            ;;
        *) return 1 ;;
    esac
}

format_for_scheme() {
    local scheme="$1"
    case "$scheme" in
        GGUF:*) printf 'gguf:%s' "$(printf '%s' "${scheme#GGUF:}" | tr '[:upper:]' '[:lower:]')" ;;
        *)      echo "$FORMAT" ;;
    esac
}

main() {
    local first="${1:-}"
    case "$first" in
        ""|-h|--help|help) usage; exit 0 ;;
    esac

    if [[ $# -lt 2 ]]; then
        echo "error: missing arguments. Need <model> and <type>." >&2
        echo "" >&2
        usage >&2
        exit 1
    fi

    local model="$1"; shift
    local type_arg="$1"; shift

    local scheme
    scheme="$(scheme_for_type "$type_arg" || true)"
    if [[ -z "$scheme" ]]; then
        echo "error: unknown quantization type '$type_arg'" >&2
        echo "       supported: int4 int8 int2 int3 w4a16 w8a16 mxfp4 mxfp8 nvfp4 fp8 gguf:q4_k_m gguf:q5_k_m ..." >&2
        exit 1
    fi

    local fmt
    fmt="$(format_for_scheme "$scheme")"

    local RECIPE_BIN
    local -a RECIPE_FLAGS
    resolve_recipe "$RECIPE"

    echo ">>> quantize $model -> $scheme (format=$fmt)"
    echo ">>> recipe=$RECIPE ($RECIPE_BIN ${RECIPE_FLAGS[*]:-}) bs=$BATCH_SIZE ga=$GRAD_ACCUM seqlen=$SEQLEN low_gpu_mem=OFF compile=OFF"
    echo ""

    mkdir -p "$OUTPUT_DIR"

    exec "$RECIPE_BIN" \
        --model "$model" \
        --scheme "$scheme" \
        --format "$fmt" \
        --device 0 \
        --batch_size "$BATCH_SIZE" \
        --gradient_accumulate_steps "$GRAD_ACCUM" \
        --seqlen "$SEQLEN" \
        --output_dir "$OUTPUT_DIR" \
        "${RECIPE_FLAGS[@]}" \
        "$@"
}

main "$@"
