# Quantization workspaces

The `quantize` app manages one reproducible workspace per source model. Model
bytes remain runtime data: Nix pins the tools and recipe but never copies weights
into the Nix store.

```bash
# OWNER/MODEL and huggingface.co URLs are both accepted. This creates
# /home/jasonbk/Projects/quantized_models/Qwen/Qwen3-8B by default.
nix run .#quantize -- init Qwen/Qwen3-8B

cd /home/jasonbk/Projects/quantized_models/Qwen/Qwen3-8B
nix flake lock
nix run .#quantize

# Repositories are public by default; preview before publishing.
nix run .#quantize -- export --dry-run
nix run .#quantize -- export
```

`w4a16` and the `default` AutoRound recipe are the defaults. The source
revision is resolved through the Hub API and its immutable commit SHA is passed
to AutoRound and recorded in `runs/<timestamp>/manifest.json`. Successful
outputs are moved to `artifacts/<model>-<format>-AutoRound`; an existing artifact
is never overwritten.

## Configure the model before running

`quantize init` deliberately creates a generic recipe. Inspect each model's
architecture and edit `quantization.json` before spending hours on calibration.
The workspace is the source of truth for model-specific full-precision rules:

```json
{
  "quantization": {
    "format": "w4a16",
    "recipe": "best",
    "kv_cache": "fp8",
    "ignore": [
      "lm_head",
      "re:.*embed_tokens$",
      "re:.*linear_attn\\.in_proj_[ab]$"
    ],
    "calibration": {}
  }
}
```

Ignore entries use llm-compressor selectors: an exact module name, a module
class such as `Linear`, or a `re:` regular expression. The preflight resolves
every selector against the loaded model and fails if any selector matches
nothing. Resolved module names are recorded in the run manifest. Do not copy
the Qwen example blindly to another architecture; identify its output head,
embeddings, gates/routers, recurrent or linear-attention projections, and any
MTP/speculator modules first.

Some architectures store MTP tensors in the checkpoint without instantiating
them in the Transformers model. llm-compressor 0.13 copies those tensors
verbatim into the compressed checkpoint, preserving BF16 automatically. Do not
add such unloaded tensors (for example Qwen3.8 `mtp.fc`) to `ignore`; selector
validation correctly reports that they are not model modules.

All weight quantization is orchestrated by llm-compressor. Its
`AutoRoundModifier` invokes Intel AutoRound as the optimization algorithm and
llm-compressor writes compressed-tensors output for vLLM. With `kv_cache=fp8`,
a separate `QuantizationModifier` calibrates static KV scales first in the same
oneshot lifecycle. AutoRound remains the final modifier so its compressed
weight metadata controls serialization. The runner preserves the KV scales
across AutoRound's weight cleanup and writes them as an indexed safetensors
shard.

Static FP8 KV-cache calibration uses an accumulating min/max observer. K and V
ranges are collected separately per full-attention layer across the complete
calibration corpus instead of being replaced by the final minibatch.

The `best` profile keeps AutoRound's quality-oriented 1,000 tuning iterations
and uses 512 samples, sequence length 2,048, and batch size 4. Retained
AutoRound captures are published into a versioned disk activation store after
capture and exposed through lazy exact-range reads. A manifest is published
only after every immutable extent is checksummed and fsynced. The initial
baseline deliberately uses one record per extent, one synchronous reader, the
kernel page cache, and no userspace LRU; telemetry must justify later
coalescing or cache changes.

New workspaces contain the complete `calibration.storage` and
`calibration.resources` configuration from issue #4. The scratch path defaults
to `<workspace>/.cache/autoround`. Preflight accounts for the model/optimizer
estimate, live frontier and minibatch, all bounded queues, pinned allowance,
safety reserve, checkpoint generations, filesystem reserve, cgroup headroom,
and `MemAvailable`. It fails before quantization when the aggregate does not
fit. `auto` pageable LRU currently resolves to zero.

Runs use a transient user systemd scope by default (`--no-isolate` opts out)
with the configured memory limits, CPU/IO weights, nice level, and OOM score.

Run the end-to-end preflight before the full job:

```bash
nix run .#quantize -- doctor
nix run .#quantize -- test
```

It uses the real model, ignore selectors, dataset, recipe batch size and
sequence length, optional FP8 KV modifier, packing, and checkpoint validation.
By default it uses 32 calibration samples so the test remains practical while
retaining per-step XPU memory pressure. Pass `--full-calibration` to exercise
the recipe's complete host-side activation cache. Uncached 5- and 20-iteration
passes separate fixed loading/calibration/saving costs from tuning time and
scale the measured fixed cost to the target sample count. Its disposable
checkpoint stays under `runs/<timestamp>/test-output`; it never becomes an
artifact or Hub upload. The fitted full-run duration is directional rather than
a guarantee because per-layer tuning and compilation can be nonlinear.

## Power-loss-safe block checkpoints

Full `quantize run` jobs checkpoint after every completed decoder block by
default. This is not an upstream llm-compressor feature: the CLI extends
`AutoRoundModifier` at its public sequential-block lifecycle boundary. Resume
with the identical workspace recipe:

```bash
nix run .#quantize -- run --resume
```

Checkpoints form a content-addressed chain under
`checkpoints/<run-hash>/nodes/`. The run hash covers the immutable source SHA,
exact tokenized calibration data, seed and ordering, resolved ignore modules,
all algorithm settings, package versions, quantizer script, Nix Python closure,
and workspace `flake.lock`. A changed recipe, dataset, tool closure, or source
commit selects a different DAG instead of loading incompatible state.

Each immutable node contains only one completed block's state and the hash of
its parent. The DAG also alternates between two atomically replaced snapshots
of AutoRound's propagated quantized activation and RNG state. Those snapshots
are required for accuracy-equivalent continuation; restoring weights alone
would lose the quantization error AutoRound intentionally carries into the next
block. Writes are flushed and atomically renamed before `head.json` advances.
On restart the complete chain and payload hashes are verified before any block
is skipped. At most the currently tuning block is repeated after a power loss.

The storage cost is approximately one model's block states plus two calibration
activation snapshots, rather than one full model per checkpoint. Checkpointing
is disabled for `quantize test` by default to keep the two timing passes honest;
use `--checkpoint` explicitly when testing the checkpoint mechanism. Disable it
for a disposable full run with `--no-checkpoint`.

MXFP4/MXFP8/FP8 are intentionally experimental on Intel XPU and require
`--allow-experimental`. To produce W4A16 weights and calibrated static FP8
KV-cache scales in one llm-compressor pass:

```bash
nix run .#quantize -- run --kv-cache fp8 --calibration-samples 128
```

This uses the same aligned calibration dataset for AutoRound and KV observers,
then saves compressed-tensors metadata that vLLM consumes with
`--kv-cache-dtype fp8`. The manifest records the immutable source SHA, sequence
length, and sample count.

## Diagnostics and export gates

Reports are written incrementally under `runs/<id>/eval/` and copied into the
artifact. `telemetry.jsonl` is append-and-fsync crash-safe. It records corpus
roles and bytes, exact-range reads, checksums, cache behavior, block times,
propagated activation ranges and NaN/Inf counts, preflight accounting, and
component disposition. KV reports keep K and V separate per attention layer;
values unavailable from the pinned observer are explicit `null` fields.

Serialization is not an export gate. `artifact-integrity.json` compares source
and output safetensors inventories, protects MTP, vision, embeddings and
`lm_head`, checks ignored dtype/shape, processor assets, KV scale mapping, and
index/shard agreement. `export-gates.json` additionally requires fixed-seed
BF16/quantized KL and top-token agreement, generation/repetition checks,
context-length sweeps, and an intended-vLLM-loader reload. GPU-dependent gates
start as `pending`, so `quantize export` refuses the artifact. A reviewed
exception requires `--waive-gates --waiver-reason TEXT`; failed gates,
timestamp, and reason are recorded before upload.

Real-model verification remains a 512×2048, 1,000-iteration Qwen3.8 block,
then two blocks and a restart. Populate the pending evaluation gates and compare
telemetry against the requested RAM/cache/lookahead baselines before a full run.

The generated development shell includes `huggingface_hub`, `hf-xet`, Git LFS,
Git, and jq. Authentication is runtime state outside Nix:

```bash
mkdir -p ~/.config/huggingface
chmod 700 ~/.config/huggingface
nix develop
hf auth login
```

`HF_HOME` defaults to `/var/cache/huggingface` for disposable downloads and
datasets. `HF_TOKEN_PATH` defaults to `~/.config/huggingface/token`, keeping the
credential private rather than placing it in the shared cache or Nix store.
New workspaces publish publicly to
`jasonboukheir/<model>-W4A16-AutoRound` by default. Override the destination
with `--hf-repo` or `HF_QUANTIZATION_OWNER`, or use `export --private` when a
candidate needs private review.
