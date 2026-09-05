# KVarN XPU beta

KVarN is an opt-in beta for the Brutus Intel Arc Pro B70 service. It provides a
3.74x smaller raw full-attention cache page and has remained coherent at long
context where the previously tested FP8 KV cache did not. Native decode parity
with `auto` is demonstrated in the retained exploratory measurements. This is
not a claim of formal statistical parity: KVarN prefill remains slower, so the
retained B1/65K aggregate result is about 96% of `auto`.

## Enable it

Change only the cache dtype:

```nix
kvCacheDtype = "kvarn_k4v4_g128_compact";
```

No `KVARN_*` environment variables are required. On XPU, this dtype binds the
validated native Xe2 qlen=1 reader automatically. The native fast path targets
Hq24/Hkv4/D256, K4V4/G128, eager execution, batch sizes through 12, and no
sliding-window attention. Its `xe2_dpas` writer/reader layout is an immutable
cache ABI, so an incompatible native problem fails closed instead of reading
that cache through a natural-layout fallback. Use `kvCacheDtype = "auto"` for
rollback; the `KVARN_*` variables are development diagnostics, not service
configuration.

The beta profile selects `q6_prefetch_record_cursor` (ID18) with the
`b70_q6_id18_v1` adaptive split policy. The policy uses 32 splits for B1 and 24
for B4. Cache layout is an engine-lifetime ABI; do not change the writer,
reader, or layout selectors after an engine has allocated its cache.

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

- The retained two-repeat exploratory 4K comparison measured KVarN at 99.0% of
  `auto` output throughput for B1 and 97.8% for B4. Request decode throughput
  was 99.1% and 98.1%, respectively. At B1/65K, decode was 99.9%, while slower
  prefill reduced aggregate output throughput to 95.9%. These are matched B70
  measurements, not the sealed eight-repeat ABBA parity gate.
- MTP, prefix caching, XPU graphs, multimodal serving, and production graph
  capture are outside the beta contract.
- Native decode is specialized. The dtype-only B70 profile fails clearly when
  its model-shape or native-operation contract is unavailable.
- The bounded recent-fp16 policy spends additional fixed pool memory and may
  reduce the automatically supported concurrency on memory-constrained models.
- K4V2 and non-compact presets have not received the same service validation.

Use `auto` as the rollback:

```nix
kvCacheDtype = "auto";
```

Continue performance work with [the native XPU gates](kvarn-native-xpu-gates.md)
and the durable artifacts under `benchmark-results/kvarn/`.

The retained beta performance evidence is:

- `benchmark-results/kvarn/round8-s1-h-id18-20260905`
- `benchmark-results/kvarn/round8-s1-h-id18-s32-65k-b1`
