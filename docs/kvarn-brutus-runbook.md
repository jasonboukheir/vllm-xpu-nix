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
      builders = stack.lib.x86_64-linux;
      localSource = name: path: builtins.path {
        inherit name path;
        filter = sourcePath: _sourceType:
          let base = builtins.baseNameOf sourcePath;
          in !(builtins.elem base [
            ".git"
            ".dev-bin"
            ".venv"
            "__pycache__"
          ]);
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
    in package.pythonModule.withPackages (_: [ package ])
  ' 2>&1 | tee "$run_root/build.log"

candidate_env=$(readlink -f "$run_root/candidate-env")
test -x "$candidate_env/bin/vllm"
test -x "$candidate_env/bin/python"
printf '%s\n' "$candidate_env" > "$run_root/candidate-env.txt"
nix path-info --json -S "$candidate_env" > "$run_root/candidate-env-path-info.json"
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

The five cases below cover every fixture category and exactly 4,096 scored
decode positions without doing a 32K decode:

| Case | Prompt tokens | Decode positions | Coverage |
| --- | ---: | ---: | --- |
| `dialogue-127` | 127 | 1,024 | cache state at 127; scores positions 128 and 129 |
| `adversarial-128` | 128 | 768 | alternate boundary-adjacent start |
| `code-4095` | 4,095 | 768 | crosses 4K |
| `math-16383` | 16,383 | 768 | crosses 16K |
| `reasoning-32767` | 32,767 | 768 | crosses 32K |

The preparation process tokenizes the category fixture, extends it with
deterministic nonperiodic category-tagged records, and truncates token IDs to
the exact requested length. It loads BF16 once and greedily generates the
forced sequence for each case. The paired runners then consume those frozen
IDs unchanged.

```bash
logits_root="$run_root/logits"
mkdir -p "$logits_root"

env -u VLLM_XPU_ENABLE_XPU_GRAPH KVARN_NATIVE_XPU=0 \
  "$candidate_env/bin/python" - \
  "$model" "$model_revision" \
  "$packaging_repo/fixtures/kvarn-long-generation.json" \
  "$logits_root" <<'PY' 2>&1 | tee "$logits_root/prepare.log"
import hashlib
import json
import sys
from pathlib import Path

from vllm import LLM, SamplingParams, TokensPrompt

model, revision, fixture_path, output = sys.argv[1:]
output_dir = Path(output)
fixtures = {
    item["category"]: item
    for item in json.loads(Path(fixture_path).read_text(encoding="utf-8"))
}
specs = [
    ("dialogue-127", "dialogue", 127, 1024),
    ("adversarial-128", "adversarial", 128, 768),
    ("code-4095", "code", 4095, 768),
    ("math-16383", "math", 16383, 768),
    ("reasoning-32767", "reasoning", 32767, 768),
]
llm = LLM(
    model=model,
    revision=revision,
    dtype="bfloat16",
    quantization="compressed-tensors",
    kv_cache_dtype="auto",
    max_model_len=65536,
    max_num_seqs=1,
    gpu_memory_utilization=0.95,
    enforce_eager=True,
    enable_prefix_caching=False,
    language_model_only=True,
)
tokenizer = llm.get_tokenizer()


def exact_prompt_ids(category: str, target: int) -> list[int]:
    ids = tokenizer.encode(fixtures[category]["prompt"])
    counter = 0
    while len(ids) < target:
        digest = hashlib.sha256(f"{category}:{counter}".encode()).hexdigest()
        record = (
            f"\nCategory {category} evidence record {counter}; "
            f"stable digest {digest}; retain its distinct facts and order."
        )
        ids.extend(tokenizer.encode(record, add_special_tokens=False))
        counter += 1
    return ids[:target]


manifest = []
for name, category, prompt_tokens, decode_steps in specs:
    case_dir = output_dir / name
    case_dir.mkdir(parents=True, exist_ok=True)
    prompt_ids = exact_prompt_ids(category, prompt_tokens)
    params = SamplingParams(
        temperature=0.0,
        max_tokens=decode_steps,
        min_tokens=decode_steps,
        ignore_eos=True,
        detokenize=False,
    )
    request = llm.generate(
        [TokensPrompt(prompt_token_ids=prompt_ids)], params, use_tqdm=False
    )[0]
    forced_ids = list(request.outputs[0].token_ids)
    assert len(prompt_ids) == prompt_tokens
    assert len(forced_ids) == decode_steps
    prompt_json = json.dumps(prompt_ids, separators=(",", ":")) + "\n"
    forced_json = json.dumps(forced_ids, separators=(",", ":")) + "\n"
    (case_dir / "prompt-token-ids.json").write_text(
        prompt_json, encoding="utf-8"
    )
    (case_dir / "forced-token-ids.json").write_text(
        forced_json, encoding="utf-8"
    )
    manifest.append(
        {
            "name": name,
            "category": category,
            "prompt_tokens": prompt_tokens,
            "decode_steps": decode_steps,
            "prompt_token_ids_sha256": hashlib.sha256(
                prompt_json.encode()
            ).hexdigest(),
            "forced_token_ids_sha256": hashlib.sha256(
                forced_json.encode()
            ).hexdigest(),
        }
    )

assert sum(case["decode_steps"] for case in manifest) >= 4096
(output_dir / "cases.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
print(json.dumps(manifest, indent=2))
PY

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
  dialogue-127 adversarial-128 code-4095 math-16383 reasoning-32767; do
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
    --output "$case_dir/comparison.json" \
    > "$case_dir/comparison.stdout.json"
done
```

Record the paired command and the deliberately small environment allowlist,
then hash the complete logit phase:

```bash
for case_name in \
  dialogue-127 adversarial-128 code-4095 math-16383 reasoning-32767; do
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
  jq -n '{KVARN_NATIVE_XPU: "0", VLLM_XPU_ENABLE_XPU_GRAPH: null}' \
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
  dialogue-127 adversarial-128 code-4095 math-16383 reasoning-32767; do
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
    --output "$case_dir/comparison-native.json" \
    > "$case_dir/comparison-native.stdout.json"
done
```

After the optional native loop, retain the original non-native manifest and
add a native manifest that hashes the new artifacts:

```bash
for case_name in \
  dialogue-127 adversarial-128 code-4095 math-16383 reasoning-32767; do
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
  jq -n '{KVARN_NATIVE_XPU: "1", VLLM_XPU_ENABLE_XPU_GRAPH: null}' \
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
nix run .#vllm-xpu-brutus-kvarn-b1 -- "$candidate_env" \
  2>&1 | tee "$phase_dir/engine.log"
```

In terminal B, export the same `packaging_repo`, `model`, `model_revision`,
`served_model`, `run_root`, and `phase_dir` values. Read `candidate_env` from
the durable session file and run:

```bash
candidate_env=$(<"$run_root/candidate-env.txt")
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
  | rg '^(CCL_ATL_TRANSPORT|CCL_LOG_LEVEL|CCL_PROCESS_LAUNCHER|CCL_ZE_IPC_EXCHANGE|HF_HOME|KVARN_[A-Z0-9_]+|VLLM_CACHE_ROOT|VLLM_TARGET_DEVICE|VLLM_XPU_ENABLE_XPU_GRAPH)=' \
  | jq -Rsc '
      split("\n")
      | map(select(length > 0)
          | capture("^(?<key>[^=]+)=(?<value>.*)$"))
      | map({key: .key, value: .value})
      | from_entries
    ' > "$phase_dir/environment.json"

jq -e 'index("--kv-cache-dtype") as $i
  | .[$i + 1] == "kvarn_k4v4_g128_compact"
    and index("--enforce-eager") != null
    and index("--language-model-only") != null
    and index("--no-enable-prefix-caching") != null
    and index("--speculative-config") == null
    and index("--compilation-config") == null' "$phase_dir/argv.json"
jq -e '.KVARN_NATIVE_XPU == "0"
  and .VLLM_XPU_ENABLE_XPU_GRAPH == null' "$phase_dir/environment.json"

curl -fsS http://127.0.0.1:8000/metrics > "$phase_dir/metrics-before.txt"
cd "$packaging_repo"
./scripts/kvarn_service_gate.py \
  --base-url http://127.0.0.1:8000 \
  --model "$served_model" \
  --fixtures fixtures/kvarn-long-generation.json \
  --concurrency 1 \
  --output "$phase_dir/service-gate.json" \
  > "$phase_dir/service-gate.stdout.json"
curl -fsS http://127.0.0.1:8000/metrics > "$phase_dir/metrics-after.txt"
```

The gate performs two greedy generations per fixture in the same process,
concurrent isolation at the selected width, cancellation followed by
replacement, corruption checks, and final idle-metric checks. Stop terminal A
with Ctrl-C only after the gate and final metrics finish. Then finalize the
phase after the engine log stops changing:

```bash
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
cache, but set `phase_dir="$run_root/b1-restart"`. Repeat every launch,
capture, gate, stop, scan, and provenance command. Verify cross-process greedy
identity:

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
```

After both B1 starts pass, repeat with `phase_dir="$run_root/b4"`, the B4 app,
and gate concurrency four:

```bash
phase_dir="$run_root/b4"
mkdir -p "$phase_dir"
cd "$config_repo"
nix run .#vllm-xpu-brutus-kvarn-b4 -- "$candidate_env" \
  2>&1 | tee "$phase_dir/engine.log"

cd "$packaging_repo"
./scripts/kvarn_service_gate.py \
  --base-url http://127.0.0.1:8000 \
  --model "$served_model" \
  --fixtures fixtures/kvarn-long-generation.json \
  --concurrency 4 \
  --output "$phase_dir/service-gate.json" \
  > "$phase_dir/service-gate.stdout.json"
```

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
`jq -e '.KVARN_NATIVE_XPU == "1"' "$phase_dir/environment.json"`. Run the
concurrency-one gate, stop, log scan, and provenance commands for the native
phase.

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
