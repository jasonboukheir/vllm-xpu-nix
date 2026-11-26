# Inkling 2 Mini on one Arc Pro B70

The local experimental branch builds two vLLM environments on the Brutus Nix stack. Both use upstream vLLM `0085facc58`, the Inkling XPU plugin `677db18d`, and the Brutus BMG build of vllm-xpu-kernels `65eb2861`. The second environment adds a SYCL CUTLASS expert kernel. The plugin depends on the post-0.34 model implementation and per-layer KV configuration API. The v0.34 Qwen service has its own release pin.

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
./scripts/serve-inkling-b70.sh eager
```

In a second terminal, send the same prompts and tool schema you will use for the Qwen INT4 control. The output directory must be new. Each run records the HTTP responses, token counts, first-token and end-to-end times, and speculative-decoding counters. Counts include reasoning tokens. Read the generated answers alongside the two exact-match scores and tool-call score.

```sh
./scripts/probe_inkling_b70.py inkling-trial /tmp/inkling-eager-results
```

Stop the foreground server between variants. `graph` enables decode graphs, `dspark` adds the bundled seven-token drafter, and `cutlass` also loads the compiled expert kernel. Add `--expect-draft` to the probe for the last two variants. Add `--vision` to the launcher to load the vision encoder. Use the production service alias and URL for an INT4 Qwen comparison during an authorized test window.

For long-context checks, set `INKLING_CONTEXT`, `INKLING_KV_DTYPE=fp8`, `INKLING_GPU_FRACTION` and `INKLING_BATCH_TOKENS` as needed. The probe's `--only-long N` calibrates a two-tag retrieval prompt using the server's tokenizer and sends it through the chat API. Allow room for chat formatting and generation above N. A successful startup alone only verifies the configured limit.

## KVarN port

The stock plugin implements BF16 and FP8 KV. Its attention code creates distinct caches for loop passes, writes KV during its projection/convolution prologue, and reads these caches in both prefill and decode. The target also computes a learned relative position bias and a gate/sink. The Qwen KVarN release kernels specialize in 256-dimensional heads with four KV heads; Mini uses head dimension 128, eight KV heads, and 24 query heads.

A first experiment can pack the oldest pages of the eight full-attention layer visits. Keep the recent relative-bias tail, 40 windowed visits, convolution state and DSpark drafter in formats already supported by the plugin. The allocator must reserve the packed page size. The store must encode by physical slot and pass, and the attention readers must combine old and recent history with the same softmax normalization. Graph replay and the seven-token verification path need their own tests.

For 128-token pages, one KV head's packed K4V2 payload is 13,824 bytes. Eight KV heads times eight full-attention visits gives 6,912 bytes per additional history token, versus 16,384 for FP8. At 131,072 tokens the maximum saving is 1.15625 GiB before the retained tail, allocator padding and scratch. K4V4 gives 0.90625 GiB on the same basis. Measure live allocations to establish any additional usable context. Qualify long retrieval, teacher-forced logits, draft acceptance and decode rate against Inkling FP8 before using a packed-cache candidate for a service.

[Upstream Mini implementation](https://github.com/vllm-project/vllm/pull/69028) · [Pinned B70 plugin and measurements](https://github.com/mgaruccio/vllm-inkling2-xpu/tree/677db18d0129703b529270fe75052272f0e2834a) · [Checkpoint](https://huggingface.co/thinkingmachines/Inkling-2-Mini/tree/ed40e7da625f594fc9c4990f5f66be24873c22d7)
