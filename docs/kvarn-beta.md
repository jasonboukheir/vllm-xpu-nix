# KVarN on Intel Arc Pro B70

KVarN is opt-in compressed KV storage with fixed runtime defaults.

## Enable it

```nix
kvCacheDtype = "kvarn_k4v4_g128_compact";
speculativeConfig = { method = "mtp"; num_speculative_tokens = 2; };
maxNumSeqs = 4;
```

Set `kvCacheDtype = "kvarn_k4v2_g128_compact"` to select compact K4V2 instead.
The cache dtype selects the native Xe2 reader and Sinkhorn writer for its format.
No `KVARN_*` tuning overrides are needed; historical experiment selectors are
rejected at startup. Cache layout is fixed for the engine lifetime.
MTP is optional; eligible verification selects the native reader automatically.

To use compact K4V2 for both the target and the MTP draft cache:

```nix
kvCacheDtype = "kvarn_k4v2_g128_compact";
speculativeConfig = {
  method = "mtp";
  num_speculative_tokens = 2;
  kv_cache_dtype = "kvarn_k4v2_g128_compact";
};
```

Omitting the explicit draft setting also makes it follow the target format.
The earlier xpu-v1.9.0 profile instead set
`kv_cache_dtype = "kvarn_k4v4_g128_compact"`. See the
[draft-cache comparison](releases/xpu-v1.10.0.md) for measured acceptance,
performance and memory with the K4V2 target held fixed. The MTP layer has its
own attention weights and KV contents; choosing the same format does not make
the target's cache reusable by the drafter.

## Choose the cache format

Both formats keep four-bit keys. K4V2 reduces values to two bits. At head
dimension 256, each 128-token record includes payload and quantization metadata:

| Format | Bytes per block/head | Bytes per layer with four KV heads |
| --- | ---: | ---: |
| Compact K4V4 | 35,072 | 140,288 |
| Compact K4V2 | 26,880 | 107,520 |

K4V2 uses 23.36% less paged attention storage, allowing about 30.48% more
attention tokens at the same attention-cache budget. At 262,144 logical tokens
across the 17 target-plus-MTP attention layers, the difference is 1.0625 GiB.
Weights, recurrent state, null pages and FP16 pool allocations are separate.
This storage calculation is not a measured reduction in total device residency.
With sixteen K4V2 target layers and one K4V4 draft layer, the corresponding
262K logical saving is 1.0 GiB.

The Brutus release measures about 1 GiB less attention storage at 262,144
allocated tokens with a K4V4 draft. At the same 0.96 memory-utilization budget,
its shared attention capacity increases from 265,856 to 340,608 tokens (+28.12%).
Two concurrent 151,039-token prompts each complete 512 output tokens, with
302,976 cache tokens occupied concurrently and no preemptions. The per-request
limit remains 262,144 combined tokens. See the
[xpu-v1.9.0 qualification](releases/xpu-v1.9.0.md) for the exact profile and completed
long-context checks.

Changing only that draft cache from K4V4 to K4V2 saves a further 64.03125 MiB
of measured attention tensor storage at 262,144 usable tokens, including the
null page. In the follow-up's same-budget capacity checks, shared attention
capacity increases from 340,608 to 346,880 tokens (+1.84%). These are separate
storage and capacity measurements; they do not imply lower total VRAM use at
the same memory-utilization budget.

With the same `gpuMemoryUtilization`, vLLM uses the available budget for more
cache pages. Realizing the storage saving as lower VRAM use also requires a
smaller allocation budget. The release measures capacity at the same budget
and VRAM use at the same attention-block count. The cache option preserves the configured
per-request context limit; concurrent requests share any additional capacity.

K4V2 retains at least 1,024 recent full-history tokens in FP16 during decode,
with a sixteen-block high-water mark and an eight-block low-water mark. K4V4
retains its four/zero policy. Both fit the existing sixteen-block prefill
reservation, so K4V2 does not allocate a larger FP16 pool. Its extra FP16 reads
and changed flush frequency can affect speed; use matched release measurements
to assess that cost.

Both formats are lossy. Logits and generated answers can change even when a
task benchmark reports matching accuracy. The padded `kvarn_k4v2_g128` preset
does not provide the compact format's savings. Compact K4V2 requires the paired
vLLM and native kernel release; an older native library cannot read its records.

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

See [xpu-v1.8.0 qualification](releases/xpu-v1.8.0.md) for the released K4V4
profile, [xpu-v1.9.0 qualification](releases/xpu-v1.9.0.md) for the K4V2
target release, and [xpu-v1.10.0 qualification](releases/xpu-v1.10.0.md) for the
MTP draft-cache comparison and CPU draft-metadata fix. Configuration acceptance alone does not establish GPU
correctness, quality or performance. Higher concurrency, larger images and
other models need their own qualification. DFlash, video/audio, prefix caching
and graph/V2 serving remain outside this combination.

## Rollback

Set `speculativeConfig = null` to disable MTP. Set `kvCacheDtype = "bfloat16"`
for an explicit BF16 KV control, with a context limit that fits its larger
cache. `kvCacheDtype = "auto"` follows the checkpoint's cache metadata; this
RedHat checkpoint embeds an FP8 KV default.
