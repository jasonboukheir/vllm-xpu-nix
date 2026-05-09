# Quantize / eval

```bash
# Quantize a model to W4A16 with the default (200-iter) recipe:
nix run .#quantize -- AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-BF16 int4

# KL-divergence eval of a quantized model vs its BF16 reference:
nix run .#kl-eval -- \
  --bf16-model AEON-7/Qwen3.6-27B-AEON-Ultimate-Uncensored-BF16 \
  --quant-model output/auto-round/Qwen3.6-27B-W4A16
```

Recipe overrides: `AUTOROUND_QUANTIZE_RECIPE=overnight`,
`AUTOROUND_QUANTIZE_BS=8`, etc. — see `scripts/quantize.sh` for the full
list. Output dir defaults to `$PWD/output/auto-round`.
