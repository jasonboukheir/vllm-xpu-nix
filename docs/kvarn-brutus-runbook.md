# Run Kvarn acceptance on Brutus

This runbook realizes the frozen profile in
[`kvarn-validation.md`](kvarn-validation.md) from the three local feature
branches without changing the deployed NixOS configuration. Run it locally on
Brutus and keep every result under `benchmark-results/kvarn/`; `/tmp` is not a
durable evidence directory.

The downstream foreground apps were introduced at `ff564d51`, but that commit
alone did not reliably override the realized vLLM instance. `475643e2` fixed
the override target and `4d977ffe` explicitly disabled hybrid-model prefix
caching. `4d977ffe` is therefore the minimum launcher revision for this
runbook.

Those launchers force K4V4 compact, 65,536 context, B1 or B4, eager, text-only,
no MTP, no prefix caching, no XPU graph, and `KVARN_NATIVE_XPU=0`. The first
service is the non-native B1 app. Native B1 is a later A/B only. Temperature
zero is enforced by each gate request rather than by a global server default.

## Preflight and exact local build

Stop the deployed workers before compiling or allocating the GPU. Preserve
whether chat was running so it can be restored at the end.

```bash
test "$(hostname -s)" = brutus
nix shell \
  nixpkgs#curl nixpkgs#jq nixpkgs#ripgrep nixpkgs#procps \
  nixpkgs#gnutar nixpkgs#zstd --command bash
```

Run the remaining commands in that tooling shell:

```bash
set -o pipefail
packaging_repo=/home/jasonbk/Projects/vllm-xpu-nix
vllm_repo=/home/jasonbk/Projects/vllm
kernels_repo=/home/jasonbk/Projects/vllm-xpu-kernels
config_repo=/home/jasonbk/.config/nix
model=jasonboukheir/Qwen3.8-27B-AEON-Ultimate-Uncensored-BF16-W4A16-AutoRound
model_revision=6b0622f4354481d5d04577d48ba0db844efc1330
served_model=sunny-chat
run_stamp=$(date -u +%Y%m%dT%H%M%SZ)
run_root="$packaging_repo/benchmark-results/kvarn/$run_stamp"
mkdir -p "$run_root"

# Reuse one explicit warmed compilation cache across paired and restarted
# phases, outside the immutable per-run evidence tree. Use the production
# Hugging Face cache so the exact pinned model revision is available offline.
runtime_cache_parent="$packaging_repo/benchmark-results/kvarn-runtime-cache"
export HF_HOME=/var/cache/huggingface
export CCL_ATL_TRANSPORT=ofi
export CCL_LOG_LEVEL=warn
export CCL_PROCESS_LAUNCHER=none
export CCL_ZE_IPC_EXCHANGE=sockets
export VLLM_TARGET_DEVICE=xpu
mkdir -p "$runtime_cache_parent/vllm-xpu-brutus-kvarn"

# Make the frozen profile independent of inherited tuning/debug overrides.
# KVARN_NATIVE_XPU is the only Kvarn switch deliberately set for the first
# candidate; the native A/B overrides it for that process later.
unset KVARN_DBG_LAYERS KVARN_DUMP_TILES KVARN_FAST_FLUSH
unset KVARN_FUSED_DECODE KVARN_FUSED_VERIFY KVARN_FUSED_VERIFY_MAXQ
unset KVARN_FUSED_VERIFY_MIN_BLOCKS KVARN_NATIVE_XPU_CHUNK_PREFILL
unset KVARN_NATIVE_XPU_DECODE KVARN_NATIVE_XPU_DPAS_LAYOUT
unset KVARN_NATIVE_XPU_HADAMARD_SCATTER KVARN_NATIVE_XPU_LAYER
unset KVARN_NATIVE_XPU_MATERIALIZE
unset KVARN_NATIVE_XPU_PERSISTENT_SCRATCH KVARN_NATIVE_XPU_SPLITS
unset KVARN_NUM_KV_SPLITS KVARN_POOL_MEM_FRAC KVARN_POOL_SLOTS
unset KVARN_QUANT_SLIDING KVARN_RTN_QUANTILE KVARN_SHARED_VERIFY
unset KVARN_SINKHORN_ITERS KVARN_SINK_TOKENS KVARN_SPLIT_K
unset VLLM_XPU_ENABLE_XPU_GRAPH
export KVARN_NATIVE_XPU=0

# The candidate wrappers supply the stack's pinned XPU runtime. Do not let a
# host driver path or an unrelated oneAPI shell take precedence.
unset LD_LIBRARY_PATH LIBRARY_PATH ONEAPI_ROOT SYCL_HOME CMPLR_ROOT
unset LEVEL_ZERO_V1_SDK_PATH CC CXX

git -C "$config_repo" merge-base --is-ancestor \
  4d977ffe96dcaad147718c91770374afab20fa68 HEAD
git -C "$packaging_repo" merge-base --is-ancestor \
  6945377cf94dd71f427351c4e6cdec2b24ee7e86 HEAD
git -C "$vllm_repo" merge-base --is-ancestor \
  bc05215e85ffdd11a29b06abd2c5c81a8078b76c HEAD
git -C "$kernels_repo" merge-base --is-ancestor \
  cd7fc7a1561fe188c4f73da4dc5d837244aedd3f HEAD
git -C "$packaging_repo" status --short --branch
git -C "$vllm_repo" status --short --branch
git -C "$kernels_repo" status --short --branch
git -C "$kernels_repo" submodule status --recursive

chat_was_active=$(systemctl is-active vllm-xpu-chat.service || true)
embedding_was_active=$(systemctl is-active vllm-xpu-embedding.service || true)
sudo systemctl stop vllm-xpu-chat.service vllm-xpu-embedding.service
test "$(systemctl is-active vllm-xpu-chat.service)" = inactive
test "$(systemctl is-active vllm-xpu-embedding.service)" = inactive
```

This expression uses the packaging working tree for the substrate and builder
logic, the vLLM working tree for Python sources, and the kernels working tree
for native sources. It then applies Brutus's torchvision, BMG AOT, and narrowed
kernel specialization. The result contains both `bin/vllm` for the launcher
and `bin/python` for the logit tools.

```bash
nix build --impure \
  --max-jobs 1 --cores 4 \
  --out-link "$run_root/candidate-env" \
  --expr '
    let
      stack = builtins.getFlake "path:/home/jasonbk/Projects/vllm-xpu-nix";
      pkgs = stack.inputs.nixpkgs.legacyPackages.x86_64-linux;
      builders = stack.lib.x86_64-linux;
      localSource = name: path: builtins.path {
        inherit name path;
        filter = sourcePath: _sourceType:
          let
            root = toString path;
            source = toString sourcePath;
            base = builtins.baseNameOf sourcePath;
            excludedRoots = map (entry: root + "/" + entry) [
              ".deps"
              ".dev-bin"
              ".git"
              ".pytest_cache"
              ".ruff_cache"
              ".venv"
              "build"
            ];
          in !(builtins.elem source excludedRoots)
            && base != "__pycache__"
            && base != "_version.py"
            && !(pkgs.lib.hasSuffix ".pyc" base)
            && !(pkgs.lib.hasSuffix ".so" base);
      };
      kernels = builders.mkVllmXpuKernels {
        src = localSource "vllm-xpu-kernels-kvarn"
          /home/jasonbk/Projects/vllm-xpu-kernels;
        version = "0.1.13.1+kvarn.local";
      };
      vllm = builders.mkVllm {
        src = localSource "vllm-kvarn" /home/jasonbk/Projects/vllm;
        version = "0.28.0+kvarn.local";
        inherit kernels;
      };
      package = import
        /home/jasonbk/.config/nix/hosts/brutus/services/vllm-xpu/package.nix {
          vllm-xpu-unstable = vllm;
        };
      pythonEnv = package.pythonModule.withPackages (_: [ package ]);
      # The vLLM executable is wrapped with the pinned stack Level Zero,
      # compute-runtime, IGC, oneAPI, compiler, and JIT-linker environment.
      # A plain withPackages Python lacks that wrapper and can accidentally
      # load /run/opengl-driver instead. Reuse every runtime argument except
      # the package-local PYTHONPATH; pythonEnv already supplies its complete
      # Python module closure.
      runtimeWrapperArgs = builtins.filter
        (arg: !(pkgs.lib.hasPrefix "--prefix PYTHONPATH " arg))
        package.makeWrapperArgs;
    in pkgs.symlinkJoin {
      name = "vllm-kvarn-brutus-candidate-env";
      paths = [ pythonEnv ];
      nativeBuildInputs = [ pkgs.makeWrapper ];
      postBuild =
        "rm -f \"$out/bin/python\" \"$out/bin/python3\" "
        + "\"$out/bin/python3.12\"\n"
        + "makeWrapper ${pythonEnv}/bin/python \"$out/bin/python\" "
        + builtins.concatStringsSep " " runtimeWrapperArgs
        + "\nln -s python \"$out/bin/python3\""
        + "\nln -s python \"$out/bin/python3.12\"";
    }
  ' 2>&1 | tee "$run_root/build.log"

candidate_env=$(readlink -f "$run_root/candidate-env")
test -x "$candidate_env/bin/vllm"
test -x "$candidate_env/bin/python"
for variable in \
  LD_LIBRARY_PATH ONEAPI_ROOT SYCL_HOME CMPLR_ROOT \
  LEVEL_ZERO_V1_SDK_PATH LIBRARY_PATH CC CXX; do
  rg -q "$variable" "$candidate_env/bin/python"
done
if rg -q '/run/opengl-driver' "$candidate_env/bin/python"; then
  echo "candidate Python wrapper references the mutable host driver" >&2
  false
fi
# The wrapper prefixes its pinned libraries rather than discarding an
# inherited LD_LIBRARY_PATH. The explicit unset above keeps this run
# hermetic; the wrapper ordering also keeps the pins first.
printf '%s\n' "$candidate_env" > "$run_root/candidate-env.txt"
nix path-info --json -S "$candidate_env" > "$run_root/candidate-env-path-info.json"

export XDG_CACHE_HOME="$runtime_cache_parent"
export HOME="$runtime_cache_parent/vllm-xpu-brutus-kvarn"
export VLLM_CACHE_ROOT="$HOME"

"$candidate_env/bin/python" - <<'PY' \
  2>&1 | tee "$run_root/python-xpu-preflight.log"
import json
import os

import torch

runtime = {
    name: os.environ.get(name)
    for name in (
        "LD_LIBRARY_PATH",
        "ONEAPI_ROOT",
        "SYCL_HOME",
        "CMPLR_ROOT",
        "LEVEL_ZERO_V1_SDK_PATH",
        "LIBRARY_PATH",
        "CC",
        "CXX",
    )
}
assert "/run/opengl-driver" not in (runtime["LD_LIBRARY_PATH"] or "")
assert torch.xpu.is_available()
runtime["xpu_device_count"] = torch.xpu.device_count()
runtime["xpu_device_name"] = torch.xpu.get_device_name(0)
print(json.dumps(runtime, indent=2, sort_keys=True))
PY
```

A cold build or first start can retain enough compiler/driver memory to make
the initial KV profile fail. Record that attempt, stop it, and retry once using
the warmed runtime cache before classifying a capacity failure.

## Persistent paired logits

Run these offline with no service holding the GPU. Use vLLM
`tools/kvarn_forced_decode.py` and `tools/kvarn_compare_logits.py` from a commit
containing `5af80d7634`. Each pair must use identical prompt and forced-token
JSON. Generate the forced sequence once with BF16 KV, retain the input files,
and never regenerate them between reference and candidate.

The six cases below cover every fixture category and exactly 4,608 scored
decode positions without doing a long decode at any context length:

| Case | Prompt tokens | Decode positions | Coverage |
| --- | ---: | ---: | --- |
| `dialogue-127` | 127 | 1,024 | cache state at 127; scores positions 128 and 129 |
| `adversarial-128` | 128 | 768 | alternate boundary-adjacent start |
| `code-4095` | 4,095 | 768 | crosses 4K |
| `math-16383` | 16,383 | 768 | crosses 16K |
| `reasoning-32767` | 32,767 | 768 | crosses 32K |
| `reasoning-65023` | 65,023 | 512 | ends at 65,535, one below max |

The preparation process tokenizes the category fixture, extends it with
deterministic nonperiodic category-tagged records, and truncates token IDs to
the exact requested length. It loads BF16 once and greedily generates the
forced sequence for each case. It also writes a compact `service-fixture.json`
for token-ID service replay. The paired runners then consume those frozen IDs
unchanged. `--case reasoning-65023` may be used to prepare only the near-maximum
case when the original five cases already have frozen inputs.

```bash
logits_root="$run_root/logits"
mkdir -p "$logits_root"
set -o pipefail

env -u VLLM_XPU_ENABLE_XPU_GRAPH KVARN_NATIVE_XPU=0 \
  "$candidate_env/bin/python" \
  "$packaging_repo/scripts/kvarn_prepare_forced_decode.py" \
  --model "$model" \
  --revision "$model_revision" \
  --fixtures "$packaging_repo/fixtures/kvarn-long-generation.json" \
  --output-dir "$logits_root" \
  2>&1 | tee "$logits_root/prepare.log"

jq -n --arg revision "$model_revision" '{
  revision: $revision,
  dtype: "bfloat16",
  quantization: "compressed-tensors",
  kv_cache_dtype: "auto",
  max_model_len: 65536,
  max_num_seqs: 1,
  gpu_memory_utilization: 0.95,
  enforce_eager: true,
  enable_prefix_caching: false,
  language_model_only: true
}' > "$logits_root/bf16-engine.json"
jq '.kv_cache_dtype = "kvarn_k4v4_g128_compact"' \
  "$logits_root/bf16-engine.json" > "$logits_root/kvarn-engine.json"
```

The two forced-decode processes below are the pair. Native Kvarn stays off for
the first candidate. `--top-k 50` bounds artifact size; use `--full-logits`
only when the multi-gigabyte output is intentional.

```bash
for case_name in \
  dialogue-127 adversarial-128 code-4095 math-16383 reasoning-32767 \
  reasoning-65023; do
  case_dir="$logits_root/$case_name"
  cp "$logits_root/bf16-engine.json" "$case_dir/bf16-engine.json"
  cp "$logits_root/kvarn-engine.json" "$case_dir/kvarn-engine.json"

  env -u VLLM_XPU_ENABLE_XPU_GRAPH KVARN_NATIVE_XPU=0 \
    "$candidate_env/bin/python" "$vllm_repo/tools/kvarn_forced_decode.py" \
    --model "$model" \
    --prompt-token-ids "$case_dir/prompt-token-ids.json" \
    --forced-token-ids "$case_dir/forced-token-ids.json" \
    --engine-kwargs "$case_dir/bf16-engine.json" \
    --top-k 50 --output "$case_dir/bf16.npz" \
    2>&1 | tee "$case_dir/bf16.log"

  env -u VLLM_XPU_ENABLE_XPU_GRAPH KVARN_NATIVE_XPU=0 \
    "$candidate_env/bin/python" "$vllm_repo/tools/kvarn_forced_decode.py" \
    --model "$model" \
    --prompt-token-ids "$case_dir/prompt-token-ids.json" \
    --forced-token-ids "$case_dir/forced-token-ids.json" \
    --engine-kwargs "$case_dir/kvarn-engine.json" \
    --top-k 50 --output "$case_dir/kvarn.npz" \
    2>&1 | tee "$case_dir/kvarn.log"

  "$candidate_env/bin/python" "$vllm_repo/tools/kvarn_compare_logits.py" \
    --reference "$case_dir/bf16.npz" \
    --candidate "$case_dir/kvarn.npz" \
    --context-boundary 128 \
    --context-boundary 4096 \
    --context-boundary 16384 \
    --context-boundary 32768 \
    --context-boundary 65536 \
    --output "$case_dir/comparison.json" \
    > "$case_dir/comparison.stdout.json"
done
```

Record the paired command and the deliberately small environment allowlist,
then hash the complete logit phase:

```bash
for case_name in \
  dialogue-127 adversarial-128 code-4095 math-16383 reasoning-32767 \
  reasoning-65023; do
  case_dir="$logits_root/$case_name"
  "$candidate_env/bin/python" -c '
import json, sys
with open(sys.argv[1], "w", encoding="utf-8") as stream:
    json.dump(sys.argv[2:], stream, indent=2)
    stream.write("\n")
' "$case_dir/argv.json" \
    "$candidate_env/bin/python" "$vllm_repo/tools/kvarn_forced_decode.py" \
    --model "$model" \
    --prompt-token-ids "$case_dir/prompt-token-ids.json" \
    --forced-token-ids "$case_dir/forced-token-ids.json" \
    --engine-kwargs "$case_dir/kvarn-engine.json" \
    --top-k 50 --output "$case_dir/kvarn.npz"
  jq -n \
    --arg hf_home "$HF_HOME" \
    --arg home "$HOME" \
    --arg cache "$VLLM_CACHE_ROOT" \
    --arg xdg "$XDG_CACHE_HOME" \
    '{
      CCL_ATL_TRANSPORT: "ofi",
      CCL_LOG_LEVEL: "warn",
      CCL_PROCESS_LAUNCHER: "none",
      CCL_ZE_IPC_EXCHANGE: "sockets",
      HF_HOME: $hf_home,
      HOME: $home,
      KVARN_NATIVE_XPU: "0",
      VLLM_CACHE_ROOT: $cache,
      VLLM_TARGET_DEVICE: "xpu",
      VLLM_XPU_ENABLE_XPU_GRAPH: null,
      XDG_CACHE_HOME: $xdg
    }' \
    > "$case_dir/environment.json"

  cd "$packaging_repo"
  ./scripts/kvarn_provenance.py \
    --output-dir "$case_dir" \
    --model "$model" \
    --model-revision "$model_revision" \
    --fixtures fixtures/kvarn-long-generation.json \
    --argv-file "$case_dir/argv.json" \
    --environment-file "$case_dir/environment.json"
done
```

Only after the non-native service gates pass, an optional native paired A/B
uses the same token files and Kvarn engine JSON:

```bash
for case_name in \
  dialogue-127 adversarial-128 code-4095 math-16383 reasoning-32767 \
  reasoning-65023; do
  case_dir="$logits_root/$case_name"
  env -u VLLM_XPU_ENABLE_XPU_GRAPH KVARN_NATIVE_XPU=1 \
    "$candidate_env/bin/python" "$vllm_repo/tools/kvarn_forced_decode.py" \
    --model "$model" \
    --prompt-token-ids "$case_dir/prompt-token-ids.json" \
    --forced-token-ids "$case_dir/forced-token-ids.json" \
    --engine-kwargs "$case_dir/kvarn-engine.json" \
    --top-k 50 --output "$case_dir/kvarn-native.npz" \
    2>&1 | tee "$case_dir/kvarn-native.log"

  "$candidate_env/bin/python" "$vllm_repo/tools/kvarn_compare_logits.py" \
    --reference "$case_dir/bf16.npz" \
    --candidate "$case_dir/kvarn-native.npz" \
    --context-boundary 128 \
    --context-boundary 4096 \
    --context-boundary 16384 \
    --context-boundary 32768 \
    --context-boundary 65536 \
    --output "$case_dir/comparison-native.json" \
    > "$case_dir/comparison-native.stdout.json"
done
```

After the optional native loop, retain the original non-native manifest and
add a native manifest that hashes the new artifacts:

```bash
for case_name in \
  dialogue-127 adversarial-128 code-4095 math-16383 reasoning-32767 \
  reasoning-65023; do
  case_dir="$logits_root/$case_name"
  "$candidate_env/bin/python" -c '
import json, sys
with open(sys.argv[1], "w", encoding="utf-8") as stream:
    json.dump(sys.argv[2:], stream, indent=2)
    stream.write("\n")
' "$case_dir/argv-native.json" \
    "$candidate_env/bin/python" "$vllm_repo/tools/kvarn_forced_decode.py" \
    --model "$model" \
    --prompt-token-ids "$case_dir/prompt-token-ids.json" \
    --forced-token-ids "$case_dir/forced-token-ids.json" \
    --engine-kwargs "$case_dir/kvarn-engine.json" \
    --top-k 50 --output "$case_dir/kvarn-native.npz"
  jq -n \
    --arg hf_home "$HF_HOME" \
    --arg home "$HOME" \
    --arg cache "$VLLM_CACHE_ROOT" \
    --arg xdg "$XDG_CACHE_HOME" \
    '{
      CCL_ATL_TRANSPORT: "ofi",
      CCL_LOG_LEVEL: "warn",
      CCL_PROCESS_LAUNCHER: "none",
      CCL_ZE_IPC_EXCHANGE: "sockets",
      HF_HOME: $hf_home,
      HOME: $home,
      KVARN_NATIVE_XPU: "1",
      VLLM_CACHE_ROOT: $cache,
      VLLM_TARGET_DEVICE: "xpu",
      VLLM_XPU_ENABLE_XPU_GRAPH: null,
      XDG_CACHE_HOME: $xdg
    }' \
    > "$case_dir/environment-native.json"

  cd "$packaging_repo"
  ./scripts/kvarn_provenance.py \
    --output-dir "$case_dir" \
    --manifest-name provenance-native.json \
    --model "$model" \
    --model-revision "$model_revision" \
    --fixtures fixtures/kvarn-long-generation.json \
    --argv-file "$case_dir/argv-native.json" \
    --environment-file "$case_dir/environment-native.json"
done
```

## Foreground service gates

Use two terminals. Terminal A owns the foreground process and durable engine
log. Terminal B waits for readiness, captures the actual process argv and a
non-secret environment allowlist, then runs the gate.

For the first B1 start, in terminal A:

```bash
phase_dir="$run_root/b1-first"
mkdir -p "$phase_dir"
cd "$config_repo"
set -o pipefail
unset LD_LIBRARY_PATH LIBRARY_PATH ONEAPI_ROOT SYCL_HOME CMPLR_ROOT
unset LEVEL_ZERO_V1_SDK_PATH CC CXX
runtime_cache_parent="$packaging_repo/benchmark-results/kvarn-runtime-cache"
export XDG_CACHE_HOME="$runtime_cache_parent"
export HOME=/home/jasonbk
unset VLLM_CACHE_ROOT
nix run .#vllm-xpu-brutus-kvarn-b1 -- "$candidate_env" \
  2>&1 | tee "$phase_dir/engine.log"
```

In terminal B, export the same `packaging_repo`, `model`, `model_revision`,
`served_model`, `run_root`, and `phase_dir` values. Read `candidate_env` from
the durable session file and run:

```bash
set -o pipefail
candidate_env=$(<"$run_root/candidate-env.txt")
expected_max_num_seqs=1
expected_native=0
for _attempt in $(seq 1 900); do
  if curl -fsS http://127.0.0.1:8000/health >/dev/null; then
    break
  fi
  sleep 2
done
curl -fsS http://127.0.0.1:8000/health >/dev/null
curl -fsS http://127.0.0.1:8000/v1/models \
  | tee "$phase_dir/models.json" \
  | jq -e --arg model "$served_model" '.data | any(.id == $model)'

engine_pid=$(pgrep -u "$(id -u)" -f \
  '[v]llm.*serve .*--port 8000 .*--served-model-name sunny-chat' | head -n 1)
test -r "/proc/$engine_pid/cmdline"
tr '\0' '\n' < "/proc/$engine_pid/cmdline" \
  | jq -Rsc 'split("\n") | map(select(length > 0))' \
  > "$phase_dir/argv.json"
tr '\0' '\n' < "/proc/$engine_pid/environ" \
  | rg '^(CCL_ATL_TRANSPORT|CCL_LOG_LEVEL|CCL_PROCESS_LAUNCHER|CCL_ZE_IPC_EXCHANGE|HF_HOME|HOME|KVARN_[A-Z0-9_]+|VLLM_CACHE_ROOT|VLLM_TARGET_DEVICE|VLLM_XPU_ENABLE_XPU_GRAPH|XDG_CACHE_HOME)=' \
  | jq -Rsc '
      split("\n")
      | map(select(length > 0)
          | capture("^(?<key>[^=]+)=(?<value>.*)$"))
      | map({key: .key, value: .value})
      | from_entries
    ' > "$phase_dir/environment.json"

process_package=$(jq -er '
  .[]
  | select(endswith("/bin/.vllm-wrapped"))
  | split("/bin/")[0]
' "$phase_dir/argv.json")
nix-store -qR "$candidate_env" | rg -Fx "$process_package"
jq -e \
  --arg package_prefix "$process_package/" \
  --arg model "$model" \
  --arg revision "$model_revision" \
  --arg served_model "$served_model" \
  --arg max_num_seqs "$expected_max_num_seqs" \
  '
    def arg($name): index($name) as $i
      | if $i == null then null else .[$i + 1] end;
    index("serve") as $serve
    | any(.[]; startswith($package_prefix))
      and .[$serve + 1] == $model
      and arg("--served-model-name") == $served_model
      and arg("--revision") == $revision
      and arg("--dtype") == "bfloat16"
      and arg("--quantization") == "compressed-tensors"
      and arg("--kv-cache-dtype") == "kvarn_k4v4_g128_compact"
      and arg("--max-model-len") == "65536"
      and arg("--max-num-seqs") == $max_num_seqs
      and (arg("--gpu-memory-utilization") | tonumber) == 0.95
      and index("--enforce-eager") != null
      and index("--language-model-only") != null
      and index("--no-enable-prefix-caching") != null
      and index("--speculative-config") == null
      and index("--compilation-config") == null
  ' "$phase_dir/argv.json"
expected_xdg="$packaging_repo/benchmark-results/kvarn-runtime-cache"
expected_cache="$expected_xdg/vllm-xpu-brutus-kvarn"
jq -e \
  --arg cache "$expected_cache" \
  --arg native "$expected_native" \
  --arg xdg "$expected_xdg" \
  '.CCL_ATL_TRANSPORT == "ofi"
    and .CCL_LOG_LEVEL == "warn"
    and .CCL_PROCESS_LAUNCHER == "none"
    and .CCL_ZE_IPC_EXCHANGE == "sockets"
    and .HF_HOME == "/var/cache/huggingface"
    and .HOME == $cache
    and .KVARN_NATIVE_XPU == $native
    and .VLLM_CACHE_ROOT == $cache
    and .VLLM_TARGET_DEVICE == "xpu"
    and .VLLM_XPU_ENABLE_XPU_GRAPH == null
    and .XDG_CACHE_HOME == $xdg' \
  "$phase_dir/environment.json"

curl -fsS http://127.0.0.1:8000/metrics > "$phase_dir/metrics-before.txt"
cd "$packaging_repo"
./scripts/kvarn_service_gate.py \
  --base-url http://127.0.0.1:8000 \
  --model "$served_model" \
  --fixtures fixtures/kvarn-long-generation.json \
  --concurrency 1 \
  --output "$phase_dir/service-gate.json" \
  > "$phase_dir/service-gate.stdout.json"

# Exercise the same lifecycle one token below the configured context maximum.
near_fixture="$logits_root/reasoning-65023/service-fixture.json"
test -r "$near_fixture"
./scripts/kvarn_service_gate.py \
  --base-url http://127.0.0.1:8000 \
  --model "$served_model" \
  --fixtures "$near_fixture" \
  --max-tokens 512 \
  --minimum-output-tokens 512 \
  --concurrency 1 \
  --cancel-after-events 257 \
  --output "$phase_dir/near-65535-service-gate.json" \
  > "$phase_dir/near-65535-service-gate.stdout.json"
curl -fsS http://127.0.0.1:8000/metrics > "$phase_dir/metrics-after.txt"
```

The gate performs two greedy generations per fixture in the same process,
concurrent isolation at the selected width, cancellation followed by
replacement, corruption checks, and final idle-metric checks. The cancellation
request retains the fixture's full 2,048-token budget and closes after 257
stream events, crossing both 128-token cache tiles while leaving substantial
generation work outstanding. The token-ID gate separately starts at 65,023
tokens and generates 512 positions, ending at 65,535. Stop terminal A with
Ctrl-C only after both gates and final metrics finish. Then finalize the phase
after the engine log stops changing:

```bash
test ! -e "/proc/$engine_pid"
rg -n -i \
  'GPU KV cache size|Maximum concurrency|KV cache|physical blocks|usable blocks|token capacity' \
  "$phase_dir/engine.log" > "$phase_dir/capacity-lines.txt" || true
if rg -n -i \
  '(^|[^[:alpha:]])nan([^[:alpha:]]|$)|device (lost|fault)|segmentation fault|traceback|out of memory' \
  "$phase_dir/engine.log" > "$phase_dir/failure-lines.txt"; then
  echo "fatal engine-log signature; do not accept this phase" >&2
  false
fi

cd "$packaging_repo"
./scripts/kvarn_provenance.py \
  --output-dir "$phase_dir" \
  --model "$model" \
  --model-revision "$model_revision" \
  --fixtures fixtures/kvarn-long-generation.json \
  --argv-file "$phase_dir/argv.json" \
  --environment-file "$phase_dir/environment.json"
```

Restart the same non-native B1 app with the same `candidate_env` and runtime
cache, but set `phase_dir="$run_root/b1-restart"` in both terminals. Repeat
every launch, capture, standard gate, near-65K gate, stop, scan, and provenance
command. Verify cross-process greedy identity:

```bash
phase_dir="$run_root/b1-restart"
mkdir -p "$phase_dir"
cd "$config_repo"
nix run .#vllm-xpu-brutus-kvarn-b1 -- "$candidate_env" \
  2>&1 | tee "$phase_dir/engine.log"
```

After capturing, gating, stopping, and finalizing that second process:

```bash
jq -e -s '
  (.[0].isolated_first | map(.token_ids))
  == (.[1].isolated_first | map(.token_ids))
' "$run_root/b1-first/service-gate.json" \
  "$run_root/b1-restart/service-gate.json"

jq -e -s '
  .[0].isolated_first[0].token_ids == .[1].isolated_first[0].token_ids
' "$run_root/b1-first/near-65535-service-gate.json" \
  "$run_root/b1-restart/near-65535-service-gate.json"
```

After both B1 starts pass, repeat with `phase_dir="$run_root/b4"`, the B4 app,
`expected_max_num_seqs=4` during process validation, and gate concurrency four:

In terminal A:

```bash
phase_dir="$run_root/b4"
mkdir -p "$phase_dir"
cd "$config_repo"
nix run .#vllm-xpu-brutus-kvarn-b4 -- "$candidate_env" \
  2>&1 | tee "$phase_dir/engine.log"
```

In terminal B, set `phase_dir="$run_root/b4"`,
`expected_max_num_seqs=4`, and `expected_native=0`, then repeat the complete
readiness, process capture, argv, and environment validation block. Run:

```bash
cd "$packaging_repo"
./scripts/kvarn_service_gate.py \
  --base-url http://127.0.0.1:8000 \
  --model "$served_model" \
  --fixtures fixtures/kvarn-long-generation.json \
  --concurrency 4 \
  --output "$phase_dir/service-gate.json" \
  > "$phase_dir/service-gate.stdout.json"
```

At B4 the gate submits every fixture in width-four waves, including a final
partial wave, rather than checking only the first four fixtures. It polls
`/metrics` every 100 ms until each wave either completes or reaches its required
overlap. Do not accept B4 unless every `concurrent_waves` entry records
`required_overlap_observed: true` and `peak_running >= required_running`; this
distinguishes four simultaneously resident requests from four clients that the
engine silently serialized.

Do not run native first. Once non-native B1, restarted B1, and B4 pass, use
`phase_dir="$run_root/native-b1"` and the native app. Capture its process before
checking the environment:

```bash
phase_dir="$run_root/native-b1"
mkdir -p "$phase_dir"
cd "$config_repo"
nix run .#vllm-xpu-brutus-kvarn-native-b1 -- "$candidate_env" \
  2>&1 | tee "$phase_dir/engine.log"
```

In terminal B, use the same readiness and process capture, then verify
with `phase_dir="$run_root/native-b1"`, `expected_max_num_seqs=1`, and
`expected_native=1`. Run the concurrency-one gate, stop, log scan, and
provenance commands for the native phase.

## Seal artifacts and restore service

Run checksums only after all writers stop and every phase manifest has been
regenerated. The manifests record the three source revisions and dirty states,
model revision, exact argv, allowlisted environment, prompt hashes, timestamps,
and artifact hashes.

```bash
find "$run_root" -type f ! -name SHA256SUMS -print0 \
  | sort -z \
  | xargs -0 sha256sum > "$run_root/SHA256SUMS"
sha256sum -c "$run_root/SHA256SUMS"

archive_path="$run_root.tar.zst"
tar --zstd -cf "$archive_path" -C "$(dirname "$run_root")" \
  "$(basename "$run_root")"
sha256sum "$archive_path" > "$archive_path.sha256"
```

The foreground apps do not mutate the deployed configuration. Restore the
previous worker after the foreground process exits and the GPU is free:

```bash
if test "$chat_was_active" = active; then
  sudo systemctl restart vllm-xpu-chat.service
fi
if test "$embedding_was_active" = active; then
  sudo systemctl restart vllm-xpu-embedding.service
fi
systemctl is-active vllm-xpu-chat.service vllm-xpu-embedding.service || true
journalctl -u vllm-xpu-chat.service -n 200 --no-pager
```

Do not promote Kvarn from a kernel-only or logit-only pass. The uncached,
non-native B1 restart and B4 service gates are the first service acceptance;
native results remain a separate A/B until they meet the same gates.
