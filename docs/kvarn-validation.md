# KVarN XPU validation

This checklist validates the experimental KVarN port against the deployed
BF16-KV profile. It does not change `nix/chat-profile.nix` or define a release
gate from CUDA-only claims.

## Reproducibility contract

Every run records the vLLM, kernel, packaging, model, and tokenizer revisions;
the rendered chat template; engine arguments; XPU/PyTorch/Triton versions; seed;
prompt hash; output; token count; finish reason; latency; and raw engine log.
BF16-KV, K4V4, and K4V2 use the same
`Lorbus/Qwen3.6-27B-int4-AutoRound` weights. A BF16-weight model is an optional
control for weight-quantization error, not the KVarN baseline.

Use seeds 0, 1, and 2. For the paper-comparable reasoning runs use temperature
0.6, top-p 0.95, top-k 20, and report Avg@3. Greedy runs are a separate kernel
and cache-lifecycle check.

## Test ladder

1. CPU contracts: preset parsing, malformed overrides, exact byte offsets,
   head-dimension-256 alignment, hybrid full-attention layer counting, and
   KVarN-only fixed block-size behavior during page unification.
2. XPU kernels: compare Sinkhorn, packed store, dequantization, decode, and
   speculative verify with Torch references. Cover zeros, constants, outliers,
   BF16/FP16, every preset, and token counts 127, 128, 129, 255, 256, and 257.
3. Eager engine: prefix caching, MTP, and graphs disabled. Exercise batch 1 and
   4, request teardown/reuse, noncontiguous block tables, partial tails, and
   cancellation.
4. Restore prefix caching, two-token MTP, XPU graph sizes `[3, 6]`, and the
   co-resident embedding service one at a time.
5. Run the full paired accuracy, capacity, and throughput matrix.

## Accuracy matrix

- MATH500: 500 prompts, up to 8192 output tokens.
- AIME24: 30 prompts, up to 16384 output tokens, Avg@3.
- HumanEval: 164 prompts, up to 16384 output tokens, 30-second execution limit.
- Line retrieval: 100 examples at each 100-line through 600-line bin; exact
  10-character key match.
- Long-context retrieval and multi-turn cases at 4K, 32K, and near 114688
  tokens, including repeated-prefix requests.

Store one JSONL row per item and a summary JSON. Use paired bootstrap confidence
intervals. Provisional non-inferiority margins, to be frozen before the full
run, are:

| Mode | MATH500 | HumanEval | Retrieval aggregate | AIME24 Avg@3 |
| --- | ---: | ---: | ---: | ---: |
| K4V4 | -2 pp | -2 pp | -1.5 pp | -3.3 pp |
| K4V2 | -3.5 pp | -3 pp | -3 pp | -6.7 pp |

No retrieval length bin may regress by more than 7 pp. K4V4 must not be
materially worse than K4V2. Report the upstream paper's Qwen3-4B results only as
a methodology anchor: it does not publish a Qwen3.6-27B K4V2 result.

## Decode-aware logit drift

Replay one fixed teacher-forced continuation through vLLM so every mode sees
identical tokens while its own KV cache fills. Capture logits at 128, 256, 512,
1024, 2048, 4096, and 8192 tokens. Report KL(BF16-KV || mode), Jensen-Shannon
divergence, top-1/top-5 agreement, target-token log-probability delta, p50/p95/
p99, and drift slope versus log2 context. K4V4's median and p95 KL must not
exceed K4V2's; the last-quarter mean must be at most four times the first
post-warmup quarter until pilot data supports a tighter envelope.

The existing `scripts/kl_eval.py` compares BF16 and AutoRound model weights and
must not be used as evidence for KV-cache accuracy.

Use `scripts/kvarn_endpoint_eval.py` for the paired KV-cache comparison. Start
BF16-KV and KVarN servers with the same weights, tokenizer, engine arguments,
and deterministic settings, then provide exact teacher-forced token sequences:

```console
python scripts/kvarn_endpoint_eval.py \
  --dataset validation-tokens.jsonl \
  --bf16-url http://127.0.0.1:8000 \
  --kvarn-url http://127.0.0.1:8001 \
  --output results/checkpoints.jsonl \
  --summary results/summary.json \
  --metadata-json engine-metadata.json
```

Each input row has `token_ids`, or `prompt_token_ids` and
`continuation_token_ids`, plus an optional `id`. The runner uses echoed prompt
logprobs to force both caches through identical tokens. Endpoint logprobs expose
only top-k probabilities, so its KL and Jensen-Shannon values use the shared
top-k tokens plus a residual bucket. They are reproducible, coarsened drift
signals—not substitutes for the full-vocabulary logits required by the final
accuracy gate. Set `--top-logprobs -1` for full logprobs only when the server and
available output storage can safely support them.

## Hybrid page-accounting gate

Record all four values below for each mode:

1. Raw bytes for one full-attention cache block.
2. Page bytes after full-attention/GDN hybrid unification and padding.
3. Tail-pool and graph-workspace bytes outside paged KV storage.
4. Engine-reported KV memory and token capacity at startup.

The predicted allocation must agree with the engine within one block. A KVarN
candidate fails even when its raw compression looks good if hybrid page padding
or the FP16 tail pool erases the usable capacity gain. Compare the 16 compressed
full-attention layers and 48 unchanged GDN layers explicitly; never apply a
dense-transformer compression ratio to this model.

## Performance and feature gates

Benchmark 4K, 16K, and 32K prompts with 128- and 1024-token decodes at
concurrency 1 and 4, after warmup, for at least 30 measured iterations. Record
TTFT, p50/p95 inter-token latency, output tokens/s, request throughput, and peak
memory. K4V2's throughput lower confidence bound must be at least 0.90x BF16-KV
and p95 ITL at most 1.15x; K4V4 uses 0.95x and 1.10x.

MTP greedy output must match MTP-disabled greedy output, rejected draft tokens
must not remain in the cache, and acceptance may fall by no more than five
percentage points from the paired BF16-KV run. Prefix-cached and uncached greedy
outputs/log probabilities must match at shared-prefix lengths 127, 128, 129,
and 4096. Any NaN, device fault, hang, cache corruption, or cross-request data
leak is an automatic failure.
