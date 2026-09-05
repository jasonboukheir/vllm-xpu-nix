# Kvarn K4V4 correctness and service gate

> This is the historical correctness-first gate. The current opt-in beta
> contract and known limitations are summarized in [KVarN XPU beta](kvarn-beta.md).

This branch brings `kvarn_k4v4_g128_compact` back against the exact Brutus
chat model. Correctness and cache lifecycle are release gates; speed is only a
diagnostic until the cache is trustworthy at long context.

## Frozen first milestone

- Model: `jasonboukheir/Qwen3.8-27B-AEON-Ultimate-Uncensored-BF16-W4A16-AutoRound`
- Revision: `6b0622f4354481d5d04577d48ba0db844efc1330`
- Shape: 64 layers, 16 full attention and 48 linear attention, Hq24/Hkv4/D256
- Weight quantization: `compressed-tensors`; compute dtype: BF16
- KV cache: `kvarn_k4v4_g128_compact`
- `max_model_len=65536`, `gpu_memory_utilization=0.95`
- Start with `max_num_seqs=1`, then repeat at 4
- Eager, text-only, temperature 0
- No speculative decoding, prefix caching, or XPU graphs
- Native Xe2 decode is automatic in the beta; set `KVARN_NATIVE_XPU=0` only
  for a diagnostic rollback comparison

K4V2, MTP, multimodal requests, graphs, and performance promotion are separate
milestones. Do not add them to a failed first-profile run.

## Gate ladder

1. **Static cache ABI**

   The compact D256 K4V4 record is exactly 35,072 bytes per KV head and
   140,288 bytes for Hkv4, with no power-of-two padding. The manager tile is
   128 tokens. Current vLLM must expose the cache as `[B,H,1,35072]` and the
   Kvarn backend must normalize it to `[B,H,35072]` without a copy.

2. **CPU and allocator contracts**

   Run the Kvarn configuration/layout tests and the independent-pool lifecycle
   tests. Attention and Mamba use separate block-ID namespaces and backing
   buffers. Admission must be all-or-nothing across pools; teardown must return
   blocks to their owning pool. The repository flake check must use pytest so
   pytest-style tests are actually collected.

3. **Kernel primitive oracle**

   Compare packed store, exact compact-to-FP16 materialization, and decode with
   Torch references for BF16 and FP16 zeros, constants, outliers, nonuniform
   scales, and seeded random data. Cover token lengths 1, 127, 128, 129, 255,
   256, and 257; ragged/noncontiguous block tables; recycled IDs; repeated runs;
   and a fresh process under allocator pressure. Run the safe reader before
   enabling native decode.

4. **Persistent accumulated decode**

   Use `scripts/kvarn_forced_decode.py` from this packaging checkout, not a succession
   of fresh endpoint requests. BF16 and Kvarn runs must use identical prompt and
   forced token IDs, model/tokenizer revision, and engine arguments. Cover
   natural dialogue, code, math/reasoning, and adversarial repetition near the
   127/128/129 boundaries and at 4K, 16K, and 32K. Accumulate at least 4,096
   scored decode positions. Record top-1/top-5/tie-aware agreement,
   selected-token delta, MAE/RMSE/p50/p95/p99/max, and drift by context.

5. **Foreground service, B1 then B4**

   Build from all three local feature branches and start a foreground service
   equivalent to Brutus, but with the frozen first-milestone switches above.
   Record startup pool bytes, physical/usable blocks, KV GiB, token capacity,
   and maximum concurrency. A cold build may leave compiler/driver residency;
   restart once with the warmed cache before declaring a capacity failure.

6. **Visible corruption and lifecycle**

   Every fixture must generate at least 2,048 greedy tokens. Repeat in the same
   process and after restart. Fail on an API error, timeout, missing tokens,
   NaN/device fault in logs, nondeterminism, a 16-token span repeated three
   consecutive times, or a period of at most 32 repeated for at least 128
   tokens. Exercise teardown/reuse, a cancelled stream followed immediately by
   a replacement, B4 mixed context lengths, and cross-request isolation. After
   each phase, running/waiting requests and KV-cache usage must return to zero.
   During each B4 wave, poll the live service metrics and require the running
   gauge to reach the wave width; submitting concurrent clients is not by
   itself evidence that requests were simultaneously resident.

7. **Prefix caching only after uncached acceptance**

   Enable prefix caching while retaining eager/no-MTP/no-graph operation.
   Re-run shared prefixes at 127, 128, 129, and 4096, multi-turn conversation,
   sustained generation, cancellation, and teardown gates.

## Durable evidence

Write raw artifacts under `benchmark-results/kvarn/<UTC timestamp>/`, never
only under `/tmp`. The manifest must contain the packaging, vLLM, XPU-kernels,
and external Nix configuration revisions and dirty states, model and tokenizer
revision, rendered command/environment, prompt and token hashes, full outputs,
metrics snapshots, engine log, timestamps, and checksums. Keep BF16 and Kvarn
artifacts paired. The initial milestone passes only when the uncached Kvarn
service completes every functional gate; an isolated kernel pass or a capacity
estimate is not an end-to-end pass.
