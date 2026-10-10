# Brutus performance review for v0.31.0

Read-only review on 2026-10-10. No GPU jobs, service operations, model downloads,
or source/config changes were performed. Recommendations below are experiments,
not measured speedups or release qualifications.

## Frozen inputs and actual workload

- vLLM old upstream `ced6857afa0ea7b2e3f0846a62e1394e90f15607`, old fork
  `466b9d5abe84f13c29b38c686e4d77fc09e09aa4`, target v0.31.0
  `db9527a46873454610df6dbedf79a36d6bf1a7f6`.
- Kernels old upstream `1c7cbeee1cd0c1481d48f5031f679a47f3f0ef45`, old fork
  `193cee080d654ff67c4f8bfaf924201bc88b9cf5`, target released wheel 0.1.15.4
  source `ddf336d86e3c8602888572a3502f951abd51df12` on `release/0.1.15.4`.
- Host profile: `~/.config/nix/hosts/brutus/services/vllm-xpu/models/RedHatAI__Qwen3.8-27B-INT4.nix`.
  Single Intel B70, normally one or two users; `max_num_seqs=4`, 262144-token
  ceiling, 2048 batched tokens, 0.96 GPU memory utilization, BF16 compute,
  compressed-tensors INT4 weights, compact K4V2 target **and bundled MTP2**,
  eager V1, no prefix cache, image limit 2, video 0, fixed 200704-pixel image budget.
- Cached target config at revision `bf08f3dbd9a324e53956920aad378a1f1b6dd24a`
  confirms architecture `Qwen3_5ForConditionalGeneration`, `model_type=qwen3_5`,
  64 layers, hidden size 5120, vocabulary 248320, full-attention head dimension
  256, four KV heads, and interleaved MRoPE `[11,11,10]` with rotary factor 0.25.
  The checkpoint embeds an FP8 KV scheme. Explicit BF16, rather than `auto`,
  is needed when an experiment intends unquantized KV.
- The preceding release records correctness/capacity tests but expressly lacks
  fresh throughput or latency comparisons. They cannot establish a v0.31 speedup.

## Recommendation

Qualify the existing profile first. The refresh already enables an applicable
Qwen attention fusion. The smallest useful measurements are MTP1 versus MTP2
and 1024 versus 2048 prefill tokens, with concurrency one and two. Preserve the
existing profile unless measurements demonstrate an improvement.

V2/DFlash2 is a worthwhile separate prototype, but cannot be enabled on current
KVarN by changing a flag. A BF16-cache, shorter-context V2 prototype can answer
whether a KVarN port is worth its maintenance cost. It is not a replacement for
the requested 262K/vision release qualification.

## What the frozen upstream delta actually brings

| Change | Brutus applicability |
| --- | --- |
| `ccbaed5b91e09f528180db363af7ebb9d4fafe32`, XPU `Qwen3NextAttention.use_fused_qk_norm_rope_gate` | **Applicable without a flag.** `qwen3_5.py:Qwen3_5DecoderLayer` constructs this attention class. Its fusion admits XPU, BF16, output gating and the target's interleaved MRoPE layout; `_project_qkv_gate` fuses split, Q/K normalization, partial RoPE and gate copy. Check actual selected path and text/image correctness. No measured gain yet. |
| `73c59d365b9e1e722fbea5e897e8efb646e3e513`, `ApplyRotaryEmb.forward_xpu` | New SYCL rotary dispatch for callers using this helper. Do not attribute a whole-model gain without a profile showing those calls; the fused full-attention route above bypasses separate Q/K RoPE. |
| `6654bcbfca86a8c17c8a377f6813c7a86e64cd6d`, V2 XPU sampler | Useful to a V2 prototype; current production is explicitly V1. |
| `dd3bc9c72f`, V2 shared Mamba/GDN metadata; `bf13ecc2f7`, shared token/request mappings | Potential lower CPU/metadata overhead for V2 hybrid models; no benefit while using the V1 KVarN runner. |
| `975dca5bb5db302077674cfa9afe851ee700ad73`, GDN mixed speculative output scatter | Not evidence of a native XPU improvement: it changes generic `_forward_core`, while `forward_xpu` invokes `torch.ops.vllm.gdn_attention_core_xpu`. |
| `dcfc17e0b1cb4a4bb4eeca9a75c82b82d776f25e`, XPU graphs default on | `--enforce-eager` still disables graphs. `VLLM_XPU_ENABLE_XPU_GRAPH` was removed; setting that old environment variable is no longer an effective opt-out. Keep eager/explicit graph NONE for KVarN. |
| `5c3b61e9c4`, eager disables JIT warmup | Startup/first-request behavior changes. Warm requests before measuring steady-state latency; measure readiness and first generation separately. Do not claim a faster cold start is faster generation. |
| `f730a93d2bf2cad3002578238d5a1035ee1f4177`, `max_num_active_seqs` | Can cap running admission at 2 while leaving allocation capacity 4. This can protect latency during bursts, but does nothing when only one or two requests are active and queues additional users longer. Optional, not a default improvement. |
| `6697a7dd30ceebbe69a9d37e4012d11d40f79685`, adaptive long-prefill threshold | Helps tuning multiple waiting prefills; current profile does not set this threshold. The existing 2048 total-token ceiling still applies. Avoid simultaneous scheduler changes in baseline qualification. |

The kernel delta contains only four commits: ragged speculative GDN traversal
`f4b729ae7fb3eb22c564606ec8b4d184c7c8bc4a`, PyTorch 2.14 integration
`d09702a637cfea2a0c02909af82bd789d26ea319`, added paged-decode preset
`885b120ffa082312b7d93c2be7a1e3b3dd79672f`, and the MoE
`topk_softplus_sqrt` interface change at `ddf336d8`. The GDN fix is relevant
correctness coverage, not a new measured speed optimization. Dense Qwen does
not use the MoE router. The added preset is `16,128,64,false,true,false`, not
the target's dimension-256 KVarN decode. No kernel-side performance toggle
in this delta is an obvious improvement to the current profile.

## V2, DFlash2, hybrid attention and vision

The user's likely drafter is
[incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2).
Its official card identifies a 2B BF16 draft for Qwen3.8-27B and recommends
seven proposed tokens (`method=dflash`). Its H200/SGLang results motivate a
low-concurrency experiment, but do not predict B70/vLLM/INT4 performance.

The [draft config](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2/raw/main/config.json)
has `DFlash2DraftModel`, five noncausal sliding-attention layers, 2048 window,
head dimension 128, eight KV heads, target layers `[5,19,33,47,61]`, block size 8,
and matching target hidden size/vocabulary. That structural match does not
establish acceptance rates against the RedHat INT4 target. Roughly 4 GB of raw
BF16 draft weights is an order-of-magnitude estimate before sharing and runtime
buffers; measure actual memory.

At frozen v0.31:

- `vllm/v1/worker/xpu_worker.py` selects `XPUModelRunnerV2`; its implementation
  wraps the shared GPU runner with XPU stream/graph APIs. XPU is not categorically
  excluded. `model_states/__init__.py:resolve_model_state_cls` selects
  `MambaHybridModelState` for hybrid models; `mamba_hybrid.py` has GDN metadata
  and speculative state handling. The shared runner includes multimodal encoder
  caching and input embeddings. These are implementation paths, not proof that
  this entire combination passes on B70.
- `ef42093a36` adds **ordinary draft-model** V2 speculation; this is distinct
  from DFlash2. `spec_decode/__init__.py:init_speculator` selects
  `DFlash2Speculator` by architecture.
- `config/vllm.py:_get_v1_model_runner_unsupported_features` rejects candidate-head
  DFlash drafts. Therefore **Inco DFlash2 requires V2**. Plain DFlash1 also has
  a V1 proposer; saying all DFlash requires V2 would be inaccurate.
- `48f663c37f8c77803d9281778d49ec0727abcacf` fixes XPU context-key normalization
  in `DFlashQwen3Model._project_context_kv`; `0961bbae28` enables asynchronous
  DFlash scheduling. `04730e8270` captures context precompute with the draft
  graph, useful only after graph execution is qualified.
- `db7a24c230` adds fused grouped convolution, but
  `qwen3_dflash2.py:_grouped_conv` selects it only for `hidden_states.is_cuda`.
  XPU executes the PyTorch fallback, so do not promise this new fusion on B70.
- V2 Intel-specific CI was removed by `362a64b21d5038691c929521478b63dcb4828e58`.
  This is reduced coverage, not evidence that all V2 use is broken.

The actual hard blocker is our fork: old-tip
`platforms/xpu.py:_check_kvarn_beta_unsupported_config` rejects V2, graphs,
prefix caching, and speculation other than bundled Qwen MTP1/2 with the
qualified cache formats. `config/vllm.py` also identifies KVarN as V1-only.
V2 has no port of our compact-record allocation, committed-history metadata,
target/draft ownership, materialization and reset lifecycle. Removing the
guards alone supplies none of those missing behaviors.

KVarN+DFlash2 additionally needs a separately configured draft backend with
noncausal sliding attention and suitable native presets or Triton fallback;
the current KVarN/MTP path cannot be assumed to provide it. Supporting the
target's vision encoder in V2 does not by itself prove image-conditioned draft
acceptance or cache correctness. Retain image tests when evaluating any prototype.

Upstream [issue #60174](https://github.com/vllm-project/vllm/issues/60174)
reports Qwen3.8 NVFP4 target corruption with DFlash2/DSpark after prefix hits
on 0.30/0.31. That report is **not** proof of the same failure for our INT4/XPU
profile, but makes repeated-prefix tests necessary before enabling caching in
a future prototype. Keep it off in the first experiment.

## Minimal controlled experiments for the parent

All require the parent to finish build and correctness qualification first.
Use the exact model revision, same prompts/seeds/sampling, identical image
budget, and a freshly started server per changed configuration. Separate
warm-up from measured requests and alternate baseline/candidate order.

1. **No serving change:** record the refreshed existing profile at concurrency
   1 and 2, short chat/code prompts and a long-context prompt, 512 generated
   tokens. Record TTFT, per-request output tokens/s, p50/p95 inter-token latency,
   total time, accepted draft tokens/proposed tokens, memory and preemptions.
   Repeat cells sufficiently to distinguish differences from noise. Check the
   fused Q/K/RoPE route and compare against recorded old-stack data if available;
   no valid old data means no before/after speedup claim.
2. **One-line parameter candidate:** change `num_speculative_tokens=2` to `1`,
   retaining all other profile settings. Compare decode latency and acceptance
   at concurrency 1/2, plus no-spec as a diagnostic if affordable. More proposals
   can cost more than they save. Promote only a measured win with correct text,
   tools, images, long context and mixed request lengths.
3. **One-line scheduling candidate:** keep winning speculation and compare
   `--max-num-batched-tokens 1024` with existing `2048`. Both respect KVarN's
   admission limit and GDN's 64-token grid. Test an ongoing decode overlapping a
   long prefill: lower chunks may reduce decode stalls but worsen long-prompt
   TTFT. Do not raise above 2048 under the present KVarN/MTP contract.
4. **Isolated V2 feasibility experiment:** same weights and image budget;
   explicit `bfloat16` target/draft KV, initial 32768 context and two sequences,
   eager, prefix off. First measure this profile on V1+MTP2, then change only
   `VLLM_USE_V2_MODEL_RUNNER=1`. If successful, replace MTP with a pinned Inco
   DFlash2 revision, `method=dflash`, seven speculative tokens, explicit BF16
   draft KV and, if necessary, `attention_backend=TRITON_ATTN` for the draft.
   The shorter-context setting is a feasibility choice, not an estimated B70
   maximum. Check memory before increasing context or trying graphs. A V2 speed
   result is meaningful only against the same BF16/context baseline, not the
   262K KVarN profile with different memory use.

For one/two users, optimize visible per-request latency first. Distributed
parallelism, MoE load balancing and large-batch throughput features do not
address this dense, single-device deployment. Do not enable prefix caching,
graphs or V2 in the release profile without implementing and qualifying the
required KVarN lifecycle support, or explicitly changing the product's cache
and context requirements.
