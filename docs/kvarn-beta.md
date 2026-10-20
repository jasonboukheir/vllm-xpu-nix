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

## Store reusable prefixes in system RAM

The `xpu-v1.12.2` RAM prefix-cache backend uses `OffloadingConnector` with
`IndependentCPUOffloadingSpec` to retain both compressed attention pages and
recurrent checkpoints in system RAM. Active requests continue using GPU KV
storage. The `active_only` GPU cache policy releases idle local cache entries;
the next request restores its reusable prefix from RAM.

The selected 8 GiB pinned host arena shares its byte budget across attention
pages and recurrent checkpoints, without a fixed checkpoint-slot count. Four
unrelated 131,072-token histories need approximately 7.6 GiB with one recurrent
checkpoint and a full FP16 sink snapshot each. Four 262,144-token histories
need approximately 14.6 GiB, for which 16 GiB is an alternative. Branches and
extra retained checkpoints can use more; these estimates do not guarantee
retention.
Attention stays KVarN-compressed; recurrent states keep their existing
precision, and FP16 attention sinks are preserved separately. Other cached
history uses the packed representation from normal idle-prefix reclamation.
This configuration removes the four extra GPU checkpoint reservations. The
installed 8 GiB configuration measured 306,688 shared attention tokens, retaining the
262,144-token request limit. The GPU-resident four-extra-checkpoint policy
measured 263,040 tokens. Four active requests share the GPU token capacity;
storing four unrelated full idle histories in RAM does not allow all four to
run at once. Active requests can share physical pages for a common prefix.

See the [exact Nix arguments and installed measurements](kvarn-prefix-cache.md#ram-prefix-storage).
The xpu-v1.12.2 qualification uses 8 GiB with BF16 Qwen3.5, compact K4V2 for target and draft,
TP1/PP1, eager V1 and MTP2. Weights and native kernels are unchanged.

V2 DFlash2 uses the same bounded RAM arena, adding the independent BF16 draft
window at the recurrent replay boundary. Packed committed history and FP16 sinks
are retained; the recent non-sink FP16 window is rebuilt as tokens are computed.
Free-form cold and restored generations are not guaranteed to be bitwise equal.
Use the [xpu-v1.13.0 manifest](releases/xpu-v1.13.0.json) for this profile's
capacity and qualification results.

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
- Up to four active requests, TP1/PP1, V2 runner, eager execution.
- Maximum combined input/output context follows the model limit and available
  KV-cache capacity. This profile configures 262,144 tokens; there is no separate
  128K KVarN MTP ceiling. Concurrent requests share the available KV pool.
- Prefill token budget: 2,048; competing-prefill threshold: 1,024.
  GPU memory utilization: 0.98, with the other vLLM workers disabled.
- At most two 448×448 images; no video.
- Local prefix caching is supported for this BF16 Qwen3.5 compact K4V2
  TP1/PP1 profile. See [prefix-cache sizing](kvarn-prefix-cache.md).
- RAM prefix storage is supported for this profile with DFlash2;
  see [the RAM-cache configuration](kvarn-prefix-cache.md#ram-prefix-storage).
- No XPU graphs.
- External DFlash2: two proposals with a separate BF16 sliding draft cache.

Set the target cache to compact K4V2 and configure the pinned drafter:

```nix
extraEnvironment.VLLM_USE_V2_MODEL_RUNNER = "1";
gpuMemoryUtilization = 0.98;
extraArgs = [ "--long-prefill-token-threshold" "1024" ];
speculativeConfig = {
  method = "dflash";
  num_speculative_tokens = 2;
  model = "syvai/Qwen3.8-27B-DFlash2-W4A16";
  revision = "4d30ec736ffc6b8688dc2ae2b502d9b48bdec279";
  kv_cache_dtype = "bfloat16";
};
```

The measured shared attention capacity is 270,976 tokens. Four request slots
share that capacity; they do not allow four maximum-length contexts.
This is 11.6% less shared attention capacity than the preceding profile's
306,688 tokens. The final two-proposal profile also varies across isolated
cold repeats at temperature zero with a fixed seed; deterministic text replay
is not a supported guarantee. See the
[post-release review](updates/release-v032-v2-20261019/post-release-review.md).
See the [release manifest](releases/xpu-v1.13.0.json) and
[matched performance comparison](updates/release-v032-v2-20261019/finish-performance-review.json).
The preceding xpu-v1.12.2 release retains the V1 MTP2 recipe at utilization 0.96.

The options above do not configure all these serving limits. The scheduler's
`maxNumSeqs` controls concurrency; KVarN MTP does not impose an additional
single-request limit. The bounded recent-FP16 cache window consumes additional
fixed memory.

See [xpu-v1.8.0 qualification](releases/xpu-v1.8.0.md) for the released K4V4
profile, [xpu-v1.9.0 qualification](releases/xpu-v1.9.0.md) for the K4V2
target release, and [xpu-v1.10.0 qualification](releases/xpu-v1.10.0.md) for the
MTP draft-cache comparison and CPU draft-metadata fix. Configuration acceptance alone does not establish GPU
correctness, quality or performance. Higher concurrency, larger images and
other models need their own qualification. Video/audio, distributed
KV connectors/events and graph execution remain outside this combination.

## Rollback

Set `speculativeConfig = null` to disable speculation. Set `kvCacheDtype = "bfloat16"`
for an explicit BF16 KV control, with a context limit that fits its larger
cache. `kvCacheDtype = "auto"` follows the checkpoint's cache metadata; this
RedHat checkpoint embeds an FP8 KV default.
