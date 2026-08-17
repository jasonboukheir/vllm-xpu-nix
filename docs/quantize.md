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

MXFP4/MXFP8/FP8 are intentionally experimental on Intel XPU and require
`--allow-experimental`. To produce W4A16 weights and calibrated static FP8
KV-cache scales in one llm-compressor pass:

```bash
nix run .#quantize -- --kv-cache fp8 --calibration-samples 128
```

This uses the same aligned calibration dataset for AutoRound and KV observers,
then saves compressed-tensors metadata that vLLM consumes with
`--kv-cache-dtype fp8`. The manifest records the immutable source SHA, sequence
length, and sample count.

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
