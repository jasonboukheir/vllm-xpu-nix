# KVarN on Intel Arc Pro B70

KVarN is opt-in compressed KV storage. xpu-v1.7 promotes the qualified
performance, image and bundled-MTP implementation, with fixed release defaults
instead of runtime experiment selectors. Its supported model envelope remains
bounded; the release does not claim universal service parity or vision quality.

## Enable cache compression

```nix
kvCacheDtype = "kvarn_k4v4_g128_compact";
```

This selects the Xe2 DPAS K4V4/G128 cache ABI, ID18 native decoder with adaptive
splits, qualified Sinkhorn writer, and request-stable model operations.
No `KVARN_*` tuning overrides are required. Historical experiment selectors
are rejected at startup; remove them rather than copying old factory runbooks.
Layout/reader/writer are fixed for the engine lifetime.

## Images and recommended two-token MTP

The qualified checkpoint is
`jasonboukheir/Qwen3.8-27B-AEON-Ultimate-Uncensored-BF16-W4A16-AutoRound`,
revision `6b0622f4354481d5d04577d48ba0db844efc1330` (Qwen3.5 architecture).
Use BF16 compute, compressed-tensors W4A16, one request at a time, TP1/PP1,
V1/eager, no prefix cache, maximum context8192, prefill budget2048, GPU memory
budget0.90, at most two448x448 images, and video disabled.

MTP stays opt-in. Within that envelope, two bundled draft tokens are recommended:

```nix
speculativeConfig = { method = "mtp"; num_speculative_tokens = 2; };
```

Equivalent CLI:

```sh
--kv-cache-dtype kvarn_k4v4_g128_compact \
--speculative-config '{"method":"mtp","num_speculative_tokens":2}'
```

Keep the other serving limits above; these two options alone do not configure
the entire envelope. One draft token also passes the correctness gate.
Eligible verification uses the native causal packed-cache reader automatically;
there is no materialized-versus-native trial selector. Necessary non-XPU and
other-query-shape correctness fallbacks are not alternative B70 MTP experiments.

## Results and limits

The fixed warmed short-image/text4096/image-history6143 matrix measures final
two-token KVarN at46.20/44.95/43.73 decode tok/s:40–49% above KVarN without
MTP,9–16% above one draft, and91–95% of auto with two drafts. These are
workload-specific decode results, not equal TTFT gains or a parity guarantee.
Both counts pass the16-fixture exact-token image/text gate against KVarN-off.
See [release notes](releases/xpu-v1.7.md) for profiling, tests and evidence.

Sampled resident VRAM27.98GiB includes preallocated cache/scratch and is not
an instantaneous allocator peak. The bounded recent-FP16 cache window trades
fixed memory for correctness. Broader concurrency, contexts, image sizes,
models, DFlash, video/audio, prefix caching and graph/V2 serving remain
unqualified for this MTP combination. Other cache presets are not covered.

## Rollback and historical evidence

Set `speculativeConfig = null` to disable MTP, or `kvCacheDtype = "auto"` to
return to automatic cache storage. Old xpu-v1.5/v1.6 tags are preserved.
Historical experiment scripts/artifacts are diagnostic records, not release
configuration choices. The public optimization-factory surface remains removed.

[Megaissue: measurements and deferred work](https://git.sunnycareboo.com/jasonbk/vllm-xpu-nix/issues/5).
