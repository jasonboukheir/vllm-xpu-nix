# KVarN XPU beta

KVarN is an opt-in beta for the Brutus Intel Arc Pro B70 service. It provides a
3.74x smaller raw full-attention cache page and has remained coherent at long
context where the previously tested FP8 KV cache did not. It is not yet a
performance replacement for the native `auto` KV cache.

## Enable it

Change only the cache dtype:

```nix
kvCacheDtype = "kvarn_k4v4_g128_compact";
```

No `KVARN_*` environment variables are required. On XPU, the backend attempts
the validated native Xe2 qlen=1 reader automatically and falls back when its
narrow problem contract is not satisfied. The native fast path currently
targets Hq24/Hkv4/D256, K4V4/G128, eager execution, batch sizes through 12, and
no sliding-window attention. `KVARN_NATIVE_XPU=0` remains a diagnostic rollback
override; it is not part of the normal service configuration.

The beta keeps the most recent 16 non-sink KVarN blocks in fp16 while building
later prompt chunks. Older blocks continue to flush to K4V4, so retained memory
is bounded independently of total context. Tail-pool sizing and the scheduler's
concurrency cap account for this window.

## Validated envelope

- Model: `jasonboukheir/Qwen3.8-27B-AEON-Ultimate-Uncensored-BF16-W4A16-AutoRound`
- Revision: `6b0622f4354481d5d04577d48ba0db844efc1330`
- `kvarn_k4v4_g128_compact`, BF16 compute, `compressed-tensors` weights
- Eager and text-only
- No speculative decoding, prefix caching, or XPU graphs
- `max_num_batched_tokens=2048`
- B1 and B4 native decoder primitives, including ragged sequences through
  262,144 tokens and split counts 1, 2, 4, 8, 16, 17, 24, and 32
- Service lifecycle coverage includes replay, concurrent isolation,
  cancellation/replacement, teardown, and a 65,023-token prompt on the
  correctness-first candidate

The current branch also carries a bounded continuation-prefill guardrail for a
4,095-token fixture whose normal K4V4 history caused greedy repetition. Keeping
recent prompt history in fp16 makes that failure disappear without retaining
the entire prompt.

## Known limitations

- `auto` remains faster. The latest exploratory B4/4K result was about 74% of
  auto output throughput; it was not the sealed ABBA parity gate.
- MTP, prefix caching, XPU graphs, multimodal serving, and production graph
  capture are outside the beta contract.
- Native decode is specialized. Unsupported model shapes use the slower
  fallback path.
- The bounded recent-fp16 policy spends additional fixed pool memory and may
  reduce the automatically supported concurrency on memory-constrained models.
- K4V2 and non-compact presets have not received the same service validation.

Use `auto` as the rollback:

```nix
kvCacheDtype = "auto";
```

Continue performance work with [the native XPU gates](kvarn-native-xpu-gates.md)
and the durable artifacts under `benchmark-results/kvarn/`.
