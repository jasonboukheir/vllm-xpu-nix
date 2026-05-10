{ lib, fetchurl }:

# HuggingFace model-metadata helpers, sized for build-time use (eval-time
# IFD on small text files like config.json).
#
# Pattern stolen from `flox/package-hf-models` and `mitchty/nix`'s
# `fetchhf` helper. Hash-pinning the fetch keeps the eval pure and
# binary-cache-shareable; pinning `rev` to a 40-char commit SHA keeps
# the build reproducible across upstream model updates.
#
# Usage from a NixOS module:
#
#   services.vllm-xpu.modelMetadata."Qwen/Qwen3-Embedding-0.6B" =
#     hfLib.fromHfConfig {
#       repo = "Qwen/Qwen3-Embedding-0.6B";
#       rev  = "<40-char-commit-sha>";
#       hash = "sha256-...";
#     };
#
# Or fill in the fields by hand if you'd rather skip the fetch:
#
#   services.vllm-xpu.modelMetadata."Qwen/Qwen3-Embedding-0.6B" = {
#     headDim = 128;
#     dtype   = "bf16";
#   };
#
# To bootstrap a new entry:
#
#   nix-prefetch-url --type sha256 \
#     "https://huggingface.co/$repo/resolve/$rev/config.json"

rec {
  fetchHfConfig =
    { repo, rev, hash, file ? "config.json" }:
    fetchurl {
      url = "https://huggingface.co/${repo}/resolve/${rev}/${file}";
      inherit hash;
      name = "${builtins.replaceStrings [ "/" ] [ "-" ] repo}-${file}";
    };

  readHfConfig = args: builtins.fromJSON (builtins.readFile (fetchHfConfig args));

  # Project an HF config.json down to the FA2-relevant params.
  # Handles both flat (Qwen3, Llama) and nested-text_config (Qwen3.5-MoE,
  # Gemma 3) shapes, plus the `dtype`/`torch_dtype` rename.
  attnParamsFromConfig = cfg:
    let
      tc = cfg.text_config or cfg;
      rawDtype = cfg.dtype or tc.torch_dtype or "bfloat16";
      dtypeMap = { bfloat16 = "bf16"; float16 = "fp16"; float32 = "fp32"; };
    in {
      headDim = tc.head_dim or null;
      dtype = dtypeMap.${rawDtype} or null;
    };

  # Convenience: fetch + parse + project in one call.
  fromHfConfig = args: attnParamsFromConfig (readHfConfig args);

  # Union a list of metadata entries into a kernelSet attrset.
  # Drops nulls so a metadata entry with `dtype = null` doesn't poison
  # the union — it just contributes nothing on that axis.
  unionKernelSet = entries:
    let
      collect = key: lib.unique (lib.filter (x: x != null) (map (e: e.${key} or null) entries));
    in {
      headDims = collect "headDim";
      dtypes = collect "dtype";
    };
}
