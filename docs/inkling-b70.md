# Inkling 2 Mini on one Arc Pro B70

The local experimental branch builds three vLLM environments on the Brutus Nix stack. All three use upstream vLLM `0085facc58`, the Inkling XPU plugin `677db18d`, and the Brutus BMG build of vllm-xpu-kernels `65eb2861`. The CUTLASS environment adds a SYCL expert kernel. A third long-context variant adds the sparse convolution-history patches from upstream PR #69444. The plugin depends on the post-0.34 model implementation and per-layer KV configuration API. The v0.34 Qwen service has its own release pin.

## Checkpoint

On Brutus the model belongs in the Nix-configured shared Hugging Face cache. The host configuration declares the following retention root under `services.hf-cache.roots`. The installed `hf-cache-gc` reads the manifests of retained NixOS generations. A newly declared root becomes visible to the default collector once its generation is registered. The collector defaults to a dry run.

```nix
{
  source = "brutus.inkling2-mini-trial";
  type = "model";
  repo = "thinkingmachines/Inkling-2-Mini";
  revision = "ed40e7da625f594fc9c4990f5f66be24873c22d7";
}
```

To populate that declaration, run the normal Hub downloader against the shared cache. Snapshot links and the bundled drafter share this cache. Brutus already has the verified snapshot.

```sh
HF_HOME=/var/cache/huggingface hf download thinkingmachines/Inkling-2-Mini \
  --revision ed40e7da625f594fc9c4990f5f66be24873c22d7
```

The pinned snapshot has 14 files totalling 20,191,643,294 bytes. The target shard accounts for 19,526,019,168 bytes and `mtp.safetensors` for 635,801,416 bytes. The vendor uses NVFP4 and MXFP8 routed-expert weights with BF16 dense operations.

## Single-card trial

Use an idle GPU. Build either variant, then start one foreground server. The launcher serves only on loopback, resolves the pinned snapshot locally, and starts at 8,192 context tokens and one concurrent sequence.

```sh
nix build --no-link .#vllm-xpu-inkling2 .#vllm-xpu-inkling2-cutlass
./experiments/inkling-b70/serve-inkling-b70.sh eager
```

In a second terminal, send the same prompts and tool schema you will use for the Qwen INT4 control. The output directory must be new. Each run records the HTTP responses, token counts, first-token and end-to-end times, and speculative-decoding counters. Counts include reasoning tokens. Read the generated answers alongside the two exact-match scores and tool-call score.

```sh
./experiments/inkling-b70/probe_inkling_b70.py inkling-trial /tmp/inkling-eager-results
```

Stop the foreground server between variants. `graph` enables decode graphs, `dspark` adds the bundled seven-token drafter, and `cutlass` also loads the compiled expert kernel. Add `--expect-draft` to the probe for the last two variants. Add `--vision` to the launcher to load the vision encoder. Use the production service alias and URL for an INT4 Qwen comparison during an authorized test window.

For a finished-answer comparison, run the following against each model using its served name and a new output file. The six cases use identical messages, low reasoning effort and a 4,096-token limit. The client records exact-match grades, time until the first visible answer, end-to-end latency and finish reason.

```sh
python experiments/inkling-b70/quality.py --self-test
python experiments/inkling-b70/quality.py --base-url http://127.0.0.1:18001 \
  --model inkling-trial --output /tmp/inkling-quality.json
```

The `sparse` profile uses FP8 KV, sparse convolution history, decode graphs, CUTLASS, hybrid prefix prefill and the seven-token drafter. It uses the `vllm-xpu-inkling2-sparse` package.

```sh
INKLING_CONTEXT=141312 INKLING_GPU_FRACTION=0.95 ./experiments/inkling-b70/serve-inkling-b70.sh sparse
./experiments/inkling-b70/probe_inkling_b70.py inkling-trial /tmp/inkling-long --expect-draft --only-long 140000
```

Add `--varied-records` to bury both target seals among distinct ledger entries at 20% and 80% of the prompt.

For other long-context checks, set `INKLING_CONTEXT`, `INKLING_KV_DTYPE=fp8`, `INKLING_GPU_FRACTION` and `INKLING_BATCH_TOKENS` as needed. The probe's `--only-long N` calibrates a two-tag retrieval prompt using the server's tokenizer and sends it through the chat API. Allow room for chat formatting and generation above N. A successful startup alone only verifies the configured limit.

## Measurements on Brutus

The card ran at its configured 230 W limit. On an identical short chat prompt with a 1,024-token output budget, the two-run median decode rates were 21.0 tok/s eager, 55.2 with full decode graphs, 68.8 with graphs and DSpark, and 86.5 with CUTLASS and DSpark. The production Qwen INT4 stack with K4V2 and DFlash2 measured 53.7 tok/s. Both models spent this particular budget on reasoning. Each model served one client during these measurements.

On the separate six-task low-effort screen with a 4,096-token limit, Inkling completed five correct JSON answers in 56.6 seconds total. Qwen took 136.1 seconds, returned two correct bare JSON answers, and returned two additional correct answers inside Markdown code fences. Both hit the generation cap on scheduling. The FP8 sparse configuration retrieved both tags from a 140,010-token prompt and both buried seals from a 245,781-token ledger. The latter took 129.8 seconds to first token and 135.9 seconds through the final answer. The 8K multimodal CUTLASS profile passed single- and two-image requests.

Use the six-case client and logits comparison with the same cache format, prefix policy and generation parameters when reproducing compression results.

## KVarN port

The stock plugin implements BF16 and FP8 KV. Its attention code creates distinct caches for loop passes, writes KV during its projection/convolution prologue, and reads these caches in both prefill and decode. The target also computes a learned relative position bias and a gate/sink. The Qwen KVarN release kernels specialize in 256-dimensional heads with four KV heads; Mini uses head dimension 128, eight KV heads, and 24 query heads.

A first experiment can pack the oldest pages of the eight full-attention layer visits. Keep the recent relative-bias tail, 40 windowed visits, convolution state and DSpark drafter in formats already supported by the plugin. The allocator must reserve the packed page size. The store must encode by physical slot and pass, and the attention readers must combine old and recent history with the same softmax normalization. Graph replay and the seven-token verification path need their own tests.

For 128-token pages, one KV head's packed K4V2 payload is 13,824 bytes. Eight KV heads times eight full-attention visits gives 6,912 bytes per additional history token, versus 16,384 for FP8. At 131,072 tokens the maximum saving is 1.15625 GiB before the retained tail, allocator padding and scratch. K4V4 gives 0.90625 GiB on the same basis. Measure live allocations to establish any additional usable context. Qualify long retrieval, teacher-forced logits, draft acceptance and decode rate against Inkling FP8 before using a packed-cache candidate for a service.

The CPU reference exercises K4V4 and K4V2 with shuffled pages for one-token decode, eight-token verification and chunked prefill. Use a checkout of the released vLLM fork at `b18eee6e90c77a08b05d687cc9b47ebdd08cfc3e`.

```sh
nix shell .#vllm-xpu-inkling2 -c python experiments/inkling-b70/inkling_kvarn_reference.py --fork /path/to/vllm
```

[Upstream Mini implementation](https://github.com/vllm-project/vllm/pull/69028) · [Pinned B70 plugin and measurements](https://github.com/mgaruccio/vllm-inkling2-xpu/tree/677db18d0129703b529270fe75052272f0e2834a) · [Checkpoint](https://huggingface.co/thinkingmachines/Inkling-2-Mini/tree/ed40e7da625f594fc9c4990f5f66be24873c22d7)
