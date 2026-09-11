# KVarN on Intel Arc Pro B70

KVarN is opt-in compressed KV storage with fixed runtime defaults.

## Enable it

```nix
kvCacheDtype = "kvarn_k4v4_g128_compact";
speculativeConfig = { method = "mtp"; num_speculative_tokens = 2; };
maxNumSeqs = 4;
```

The cache dtype selects the native Xe2 K4V4/G128 reader and Sinkhorn writer.
No `KVARN_*` tuning overrides are needed; historical experiment selectors are
rejected at startup. Cache layout is fixed for the engine lifetime.
MTP is optional; eligible verification selects the native reader automatically.

## Serving profile

- Model: `RedHatAI/Qwen3.8-27B-INT4`, revision
  `bf08f3dbd9a324e53956920aad378a1f1b6dd24a` (Qwen3.5 architecture).
- BF16 compute, compressed-tensors W4A16 weights.
- Up to four active requests, TP1/PP1, V1 runner, eager execution.
- Maximum combined input/output context follows the model limit and available
  KV-cache capacity. This profile configures 262,144 tokens; there is no separate
  128K KVarN MTP ceiling. Concurrent requests share the available KV pool.
- Prefill token budget: 2,048; GPU memory utilization: 0.96, with the other
  vLLM workers disabled. See the release notes for the measured memory budget.
- At most two 448×448 images; no video.
- No prefix caching or XPU graphs.
- Bundled MTP: one or two draft tokens; two recommended.

The options above do not configure all these serving limits. The scheduler's
`maxNumSeqs` controls concurrency; KVarN MTP does not impose an additional
single-request limit. The bounded recent-FP16 cache window consumes additional
fixed memory.

See [xpu-v1.8.0 qualification](releases/xpu-v1.8.0.md) for the exact tested
package and evidence. Configuration acceptance alone does not establish GPU
correctness, quality or performance. Higher concurrency, larger images and
other models need their own qualification. DFlash, video/audio, prefix caching
and graph/V2 serving remain outside this combination.

## Rollback

Set `speculativeConfig = null` to disable MTP. Set `kvCacheDtype = "bfloat16"`
for an explicit BF16 KV control, with a context limit that fits its larger
cache. `kvCacheDtype = "auto"` follows the checkpoint's cache metadata; this
RedHat checkpoint embeds an FP8 KV default.
