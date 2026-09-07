# KVarN on Intel Arc Pro B70

KVarN is opt-in compressed KV storage with fixed runtime defaults.

## Enable it

```nix
kvCacheDtype = "kvarn_k4v4_g128_compact";
speculativeConfig = { method = "mtp"; num_speculative_tokens = 2; };
```

The cache dtype selects the native Xe2 K4V4/G128 reader and Sinkhorn writer.
No `KVARN_*` tuning overrides are needed; historical experiment selectors are
rejected at startup. Cache layout is fixed for the engine lifetime.
MTP is optional; eligible verification selects the native reader automatically.

## Supported serving envelope

- Model: `jasonboukheir/Qwen3.8-27B-AEON-Ultimate-Uncensored-BF16-W4A16-AutoRound`,
  revision `6b0622f4354481d5d04577d48ba0db844efc1330` (Qwen3.5 architecture).
- BF16 compute, compressed-tensors W4A16 weights.
- One request at a time, TP1/PP1, V1 runner, eager execution.
- Maximum combined input/output context follows the model limit and available
  KV-cache capacity; there is no separate 128K KVarN MTP ceiling.
- Prefill token budget: 2,048; GPU memory utilization: 0.90.
- At most two 448×448 images; no video.
- No prefix caching or XPU graphs.
- Bundled MTP: one or two draft tokens; two recommended.

The two options above do not configure all these serving limits. Broader
concurrency, larger images, other models/cache presets, DFlash, video/audio,
prefix caching and graph/V2 serving are outside this supported combination.
The bounded recent-FP16 cache window consumes additional fixed memory.

## Rollback

Set `speculativeConfig = null` to disable MTP, or `kvCacheDtype = "auto"`
to return to automatic cache storage.
