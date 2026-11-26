# Inkling 2 Mini on one Arc Pro B70

The `vllm-xpu-inkling2-kvarn` package runs Mini with its bundled seven-token DSpark drafter, KVarN K4V4 long-history KV, an 8 GiB RAM prefix cache, and extensible device memory. The Brutus recipe enables four concurrent sequences, two images per request, streaming, native reasoning and tool calls. Its 262,144-token per-request limit includes generation. These results were collected on one 30.3 GiB B70 at its configured 230 W.

## Results on Brutus

| Single-card workload | Brutus Qwen3.8-27B-INT4 + DFlash2 | Inkling 2 Mini + DSpark + K4V4 |
| --- | ---: | ---: |
| Short decode, median of two 1,024-token runs | 53.74 tok/s | 82.93 tok/s |
| 32 public math and Python questions, low effort | 32/32 in 99.05 s | 31/32 in 58.92 s |
| Six practical tasks, low effort, exact requested JSON | 2/6 in 136.1 s | 5/6 in 68.3 s |
| Peak resident device memory sampled during the 32 public questions | 28.11 GiB | 26.83 GiB |
| Configured context, including output | 131,072 | 262,144 |

Qwen used its released 0.34 stack, K4V2 history and an 8 GiB RAM cache. Inkling used the vendor's mixed NVFP4/MXFP8 checkpoint. The short decode probe used the same advice prompt and both models spent its entire output budget on reasoning. The public panel used the same 20 GSM8K and 12 CRUXEval-O questions and allowed 4,096 output tokens per question. Both models finished all 32. Inkling missed one arithmetic question. On the six practical tasks, Qwen answered two additional cases correctly inside Markdown fences. Inkling's incorrect answer was a constrained schedule.

The device memory figures come from Intel xe resident-memory counters sampled every 0.5 seconds. The engine's separate model-load measurement was 18.73 GiB for Qwen and 19.57 GiB for Inkling with vision. Tokenizers differ. The same 659,752-byte ledger took 245,781 tokens including chat formatting for Inkling and 362,148 user-content tokens before formatting for Qwen.

- **Long retrieval.** The full Inkling server recovered the two codes planted 20% and 80% through the 245,781-token ledger. Time to first token was 152.80 s and the answer finished in 162.94 s, with 31.86 decode tok/s. The separate FP8 text-only, single-slot Inkling control took 129.83 s to first token and decoded at 52.10 tok/s.
- **RAM prefix cache.** The four-slot suite passed 140 checks, including partial-page restores after another request overwrote the staging slot. An independent 32,768-token replay loaded 538,116,096 bytes from RAM, hit 32,767 host tokens with zero device hits, and reproduced all six cold output tokens. Time to first token fell from 9.81 s to 0.60 s.
- **Serving.** Four simultaneous requests each returned their own hidden code and stopped naturally. The image tests read one image and two distinct images in order. Automatic tool selection emitted a parsed `lookup_ticket("P-17")` call.
- **Compression.** On matched single-slot short prompts, FP8 KV decoded at 88.38 tok/s and K4V4 at 82.91 tok/s. In 36 teacher-forced positions the two caches agreed on 33 top tokens. All 36 FP8-preferred tokens were in K4V4's top five. On the six practical tasks the matched FP8 profile scored 6/6.

Another single-B70 implementation reported 109.1 short-chat tok/s and 92.3 tok/s with 30K additional context at 275 W using FP8 KV. Its 765.7 tok/s figure is aggregate output across 32 concurrent clients. That implementation's near-limit needle test recovered three codes at 388,513 tokens. [Published configuration and workload definitions](https://github.com/mgaruccio/b70-inference/blob/3ef8256acc07bb02f7060432d45d39317f9d57ff/results/20261125-inkling2-draft-length-schedule/README.md)

## Run the qualified profile

The pinned model is `thinkingmachines/Inkling-2-Mini@ed40e7da625f594fc9c4990f5f66be24873c22d7`. Brutus's Nix configuration declares this exact revision in `services.hf-cache.roots` with source `brutus.inkling2-mini-trial`. Every snapshot file was verified in the shared Hub cache. The 14 files occupy 20,191,643,294 bytes in total, including 19,526,019,168 bytes of target weights and 635,801,416 bytes of bundled drafter weights. The normal host rebuild registers the new retention manifest with the collector. [Pinned checkpoint](https://huggingface.co/thinkingmachines/Inkling-2-Mini/tree/ed40e7da625f594fc9c4990f5f66be24873c22d7)

Brutus already has the snapshot. To populate another identically configured host, use the same cache root and revision.

```sh
HF_HOME=/var/cache/huggingface hf download thinkingmachines/Inkling-2-Mini \
  --revision ed40e7da625f594fc9c4990f5f66be24873c22d7
```

From this checkout on an idle B70, build and launch the full foreground server.

```sh
nix build --no-link .#vllm-xpu-inkling2-kvarn
INKLING_CONTEXT=262144 INKLING_GPU_FRACTION=0.95 INKLING_SEQUENCES=4 \
  INKLING_EXTENSIBLE=1 INK2_KVARN_FLASH_PREFILL=1 \
  ./experiments/inkling-b70/serve-inkling-b70.sh kvarn --vision
```

The foreground launcher binds to `127.0.0.1:18001` and serves the alias `inkling-trial`. It resolves the pinned snapshot offline. For the system service, the Nix recipe preserves the host's `sunny-chat` alias, pins the target and the independent DSpark model lookups to the same revision, and uses Inkling's native chat template. Given this branch as a flake input named `inkling`, import the Brutus module below. Re-selecting the previous Brutus backend restores the Qwen configuration. [Nix recipe](../experiments/inkling-b70/brutus-inkling2.nix)

```nix
{ inputs, ... }:
let
  recipe = import "${inputs.inkling}/experiments/inkling-b70/brutus-inkling2.nix" {
    runtime = inputs.inkling.packages.x86_64-linux.vllm-xpu-inkling2-kvarn;
  };
in
{
  imports = [ recipe.brutusModule ];
}
```

## Reproduce the tests

Use new output paths for each comparison. The two quality commands take the served model name and URL, so the same tests work with the Qwen endpoint. The deep offload client explicitly drops the device cache. Start the foreground test server with the additional environment setting `VLLM_SERVER_DEV_MODE=1` to enable that operation.

```sh
scripts=experiments/inkling-b70
python "$scripts/quality.py" --model inkling-trial --base-url http://127.0.0.1:18001 --output /tmp/inkling-six.json
python "$scripts/standard_quality.py" --model inkling-trial --base-url http://127.0.0.1:18001 --output /tmp/inkling-public.json
python "$scripts/probe_inkling_b70.py" inkling-trial /tmp/inkling-long --expect-draft --only-long 245760 --varied-records
python "$scripts/concurrent_clients.py" inkling-trial /tmp/inkling-concurrent
python "$scripts/offload_replay.py" --model inkling-trial --port 18001 --four --output /tmp/inkling-ram-four
python "$scripts/offload_replay.py" --model inkling-trial --port 18001 --depth 32768 --output /tmp/inkling-ram-deep
nix shell .#vllm-xpu-inkling2-kvarn -c python "$scripts/test_packed_flash_prefill.py"
```

## What the fork adds

The experimental core pins post-0.34 vLLM `0085facc58` for Mini and per-layer KV configuration. The plugin pin is `677db18d`. The sparse variants include the two upstream commits for convolution-history allocation. The Xe2 CUTLASS MoE path reuses the Brutus vllm-xpu-kernels build at `65eb2861`. The release stays on vLLM 0.34 with the Qwen port. For isolated controls the flake also provides `vllm-xpu-inkling2`, `vllm-xpu-inkling2-cutlass` and `vllm-xpu-inkling2-sparse`. [Upstream Mini support](https://github.com/vllm-project/vllm/pull/69028) · [Sparse convolution history](https://github.com/vllm-project/vllm/pull/69444) · [Pinned plugin](https://github.com/mgaruccio/vllm-inkling2-xpu/tree/677db18d0129703b529270fe75052272f0e2834a)

KVarN packs sealed 128-token pages of the eight full target-attention layer visits using four bits per key and four per value. Mini has 128-dimensional heads, eight KV heads, and 24 query heads. A sealed head page occupies 17,920 bytes including scales and offsets. The 40 sliding-window visits and DSpark keep FP8 KV, convolution state uses sparse history, and the unfinished target page uses a bounded BF16 staging ring. The graph-compatible Triton writer and reader handle grouped-query attention, learned relative bias, gate/sink, speculation rollback, and partially filled pages restored from RAM. Each request admission has a generation number so a reused slot cannot inherit the previous request's staging contents.

Block-outermost physical layout (`BLNHC`) lets the allocator reserve packed page sizes. At 141,312 tokens the matched one-sequence planner reports 1.7965 GiB for K4V4 and 2.8103 GiB for FP8 before bounded staging. The Xe2 virtual-memory backend supports the extensible allocator. For long prefill the full recipe enables `INK2_KVARN_FLASH_PREFILL=1`: it gathers bounded chunks for native flash attention and merges older and recent history using their log-sum-exp. The reference checks cover relative-only, single-window and multiple-window prefill.

The host connector uses vLLM's `CPUOffloadingSpec` with active-only GPU retention. The RAM cache lives for the serving process. The replay tests exercise a live engine across requests and use the host-transfer and device-hit counters to establish where the prefix came from.
