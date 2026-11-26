# Inkling 2 Mini on one Arc Pro B70

The `vllm-xpu-inkling2-kvarn` package runs Mini with its bundled seven-token DSpark drafter, KVarN K4V4 long-history KV, an 8 GiB RAM prefix cache, and extensible device memory. The Brutus recipe enables four concurrent sequences, two images per request, streaming, native reasoning and tool calls. Its 262,144-token per-request limit includes generation. The xpu-v2.1.0 Brutus recipe lets the extensible KV allocator size the cache from free VRAM at startup. The comparisons below used an explicit 0.93 GPU-memory setting on both models unless specified. These results were collected on one 30.3 GiB B70 at its configured 230 W.

## Results on Brutus

The autosized xpu-v2.1.0 service reported 328,134 shared tokens and decoded at 78.22 tok/s on the short probe. On the 245,760-token replay, time to first token fell from 129.25 s cold to 1.07 s after restoring 245,759 tokens from RAM with zero GPU-cache hits. Output token IDs matched. Four simultaneous clients, tool use and both vision probes also passed. The following table reports the fixed-budget comparison. [Release qualification](updates/inkling-xpu-v210-20261126/qualification.json)

| Single-card workload | Brutus Qwen3.8-27B-INT4 + DFlash2 | Inkling 2 Mini + DSpark + K4V4 |
| --- | ---: | ---: |
| Short decode, median of two 1,024-token runs | 53.74 tok/s | 78.25 tok/s |
| 32 public math and Python questions, low effort | 32/32 in 99.05 s | 31/32 in 67.31 s |
| Same 32 public questions, medium effort | 31/32 in 118.62 s | 31/32 in 105.68 s |
| 16 MMLU-Pro technical questions, medium effort | 12/16 in 471.50 s | 14/16 in 187.94 s |
| Six practical tasks, low effort, exact requested JSON | 2/6 in 136.1 s | 6/6 in 55.5 s |
| Peak resident device memory sampled during the 32 public questions | 28.11 GiB | 26.18 GiB |
| Configured context, including output | 131,072 | 262,144 |

Qwen used its released 0.34 stack, K4V2 history and an 8 GiB RAM cache. Inkling used the vendor's mixed NVFP4/MXFP8 checkpoint. The short decode probe used the same advice prompt and both models spent its entire output budget on reasoning. On two short questions that finished naturally, Inkling decoded at 137.9 and 140.6 tok/s and Qwen at 64.7 and 64.8 tok/s. The public panel used the same 20 GSM8K and 12 CRUXEval-O questions and allowed 4,096 output tokens per question. At low effort Inkling missed one arithmetic question. At medium effort Qwen missed that arithmetic question and Inkling hit its output cap on one Python case. Across other runs of Inkling at 0.95, the low-effort score ranged from 31 to 32 and a medium-effort run scored 32/32 in 77.96 s. General accuracy remains uncertain on a sample this size. On the six practical tasks Qwen answered two additional cases correctly inside Markdown fences.

Every naturally finished MMLU-Pro answer on both arms matched the official answer. Four Qwen requests and two Inkling requests exhausted the shared 4,096-token budget before giving a final answer. Eleven items were correct on both, three only on Inkling and one only on Qwen. These are elapsed times through the final answer or cap. The two models tokenize and spend reasoning tokens differently.

The device memory figures come from Intel xe resident-memory counters sampled every 0.5 seconds. The configured limits share the card across admissions. At 0.93 the allocator reported capacity for 282,887 shared tokens on two cold starts. Setting the recipe argument `gpuMemoryUtilization = 0.95;` yielded roughly 309K–324K, allowing more overlapping context. At ready, the 0.93 Inkling configuration consumed 26.27 GiB and Qwen consumed 28.11 GiB. The engine's separate model-load measurement was 18.73 GiB for Qwen and 19.57 GiB for Inkling with vision. The same 659,752-byte ledger took 245,781 tokens including chat formatting for Inkling and 362,148 user-content tokens before formatting for Qwen.

- **Long retrieval.** At 0.95 with the two-phase packed reader, the full Inkling server recovered the two codes planted 20% and 80% through the 245,781-token ledger. Time to first token was 139.64 s and the answer finished in 144.61 s at 46.93 decode tok/s. The previous packed reader recovered the same codes at 31.86 tok/s. The separate FP8 text-only, single-slot control reached 52.10 tok/s with 129.83 s to first token. Each figure comes from one complete run and includes its own speculation acceptance.
- **RAM prefix cache.** On a 245,760-token varied prompt, the cold request stored 3,333,980,160 bytes in RAM. After an unrelated request reused the staging slot and the device prefix was cleared, the warm request restored 3,318,939,648 bytes and 245,759 host tokens, with zero device hits. At 0.93 first token went from 128.37 s cold to 1.06 s warm, and the complete answer from 128.62 s to 1.29 s. The 0.95 profile reproduced the same answer in 129.84 s cold and 1.32 s warm. All six output IDs matched. At 65,536 tokens the two-phase reader gave 21.80 s cold and 0.32 s warm to first token. The short four-slot suite passed 140 checks, including partial-page restores. [Timing and qualification receipts](inkling-b70-results.json)
- **Serving.** Four simultaneous requests, each with 20,497–21,120 input tokens, returned their own hidden codes and stopped naturally. A separate stream cancellation produced the live admission sequence 4→3→4. The replacement restored 1,023 tokens from RAM and returned its exact cold answer while three peers kept generating. One-image and two-image requests passed. Automatic tool selection emitted a parsed `lookup_ticket("P-17")` call and the model consumed the supplied result on the next turn.
- **Compression.** On matched single-slot short prompts, FP8 KV decoded at 88.38 tok/s and the original packed reader at 82.91 tok/s. With four slots, vision and 262K configured, the two-phase reader agreed with the FP8 control's preferred token in 32 of 36 teacher-forced positions. All 36 FP8-preferred tokens were in its top five. The six practical tasks scored 6/6 on both FP8 and the full two-phase profile.

Another single-B70 implementation reported 109.1 short-chat tok/s and 92.3 tok/s with 30K additional context at 275 W using FP8 KV. Its 765.7 tok/s figure is aggregate output across 32 concurrent clients. That implementation's near-limit needle test recovered three codes at 388,513 tokens. [Published configuration and workload definitions](https://github.com/mgaruccio/b70-inference/blob/3ef8256acc07bb02f7060432d45d39317f9d57ff/results/20261125-inkling2-draft-length-schedule/README.md)

Brutus uses fixed seven-token DSpark verification. Shortening verification to three drafts at three or four clients worked for a synthetic warmup. A subsequent natural-chat wave stalled and the xe driver logged a copy-engine reset. The cause of the reset remains open. The fixed-seven control completed all eight measured workload waves. Four-client natural chat produced 126.6 and 157.9 aggregate output tok/s on its two measured waves.

## Run the qualified profile

The pinned model is `thinkingmachines/Inkling-2-Mini@ed40e7da625f594fc9c4990f5f66be24873c22d7`. The service module registers this exact revision in `services.hf-cache.roots` as `vllm-xpu.chat`. Every snapshot file was verified in the shared Hub cache. The 14 files occupy 20,191,643,294 bytes in total, including 19,526,019,168 bytes of target weights and 635,801,416 bytes of bundled drafter weights. The normal host rebuild registers the new retention manifest with the collector. [Pinned checkpoint](https://huggingface.co/thinkingmachines/Inkling-2-Mini/tree/ed40e7da625f594fc9c4990f5f66be24873c22d7)

Brutus already has the snapshot. To populate another identically configured host, use the same cache root and revision.

```sh
HF_HOME=/var/cache/huggingface hf download thinkingmachines/Inkling-2-Mini \
  --revision ed40e7da625f594fc9c4990f5f66be24873c22d7
```

From a checkout of the permanent `releases/xpu-v2.1.0` branch on an idle B70, build and launch the autosized foreground server. Set `INKLING_GPU_FRACTION=0.93` to reproduce the measured fixed-budget profile.

```sh
nix build --no-link .#vllm-xpu-inkling2-kvarn
INKLING_CONTEXT=262144 INKLING_GPU_FRACTION=auto INKLING_SEQUENCES=4 \
  INKLING_EXTENSIBLE=1 INK2_KVARN_FLASH_PREFILL=1 INK2_KVARN_PHASED_DECODE=1 \
  INK2_KVARN_MIXED_PREFILL=1 \
  ./experiments/inkling-b70/serve-inkling-b70.sh kvarn --vision
```

The foreground launcher binds to `127.0.0.1:18001` and serves the alias `inkling-trial`. It resolves the pinned snapshot offline. For the system service, the Nix recipe preserves the host's `sunny-chat` alias, pins the target and the independent DSpark model lookups to the same revision, and uses Inkling's native chat template. Pin the packaging input to the permanent release branch in your host flake. Keep the release’s own nixpkgs pin. Re-selecting the previous Brutus backend restores the Qwen configuration. [Nix recipe](../experiments/inkling-b70/brutus-inkling2.nix)

```nix
inputs.vllm-xpu-release = {
  type = "git";
  url = "ssh://forgejo@git.sunnycareboo.com:2222/jasonbk/vllm-xpu-nix.git";
  ref = "refs/heads/releases/xpu-v2.1.0";
};
```

Import the recipe in the host module. To reproduce the 0.93 performance comparison, pass `gpuMemoryUtilization = 0.93;` alongside `runtime`.

```nix
{ inputs, ... }:
let
  recipe = import "${inputs.vllm-xpu-release}/experiments/inkling-b70/brutus-inkling2.nix" {
    runtime = inputs.vllm-xpu-release.packages.x86_64-linux.vllm-xpu-inkling2-kvarn;
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
python "$scripts/concurrent_clients.py" inkling-trial /tmp/inkling-long-clients --rows 1280 --arrival-stagger-ms 150
python "$scripts/offload_replay.py" --model inkling-trial --port 18001 --four --output /tmp/inkling-ram-four
python "$scripts/offload_replay.py" --model inkling-trial --port 18001 --depth 32768 --output /tmp/inkling-ram-deep
python "$scripts/offload_replay.py" --model inkling-trial --port 18001 --depth 245760 --output /tmp/inkling-ram-245k
python "$scripts/tool_abort_reuse.py" --model inkling-trial --port 18001 --output /tmp/inkling-lifecycle
nix shell .#vllm-xpu-inkling2-kvarn -c python "$scripts/test_packed_flash_prefill.py"
```

The additional 16-item MMLU-Pro run uses six computer science, five engineering, and five math questions from the official test split. Selection uses the category, question ID, and a fixed seed. The client checks both source files and the frozen prompt manifest. Place the pinned dataset card at `$data/README.md` and the test parquet at `$data/test.parquet`. The B70 results directory on Brutus contains these files under `mmlu-pro-data`. [Pinned dataset and license](https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/tree/b189ec765aa7ed75c8acfea42df31fdae71f97be)

```sh
nix shell .#vllm-xpu-inkling2-kvarn -c python "$scripts/mmlu_pro_quality.py" \
  --data-dir "$data" --base-url http://127.0.0.1:18001 --model inkling-trial --output /tmp/inkling-mmlu.json
```

## What the fork adds

The Inkling package in xpu-v2.1.0 pins post-0.34 vLLM `0085facc58` for Mini and per-layer KV configuration. The plugin pin is `677db18d`. The sparse variants include the two upstream commits for convolution-history allocation. The Xe2 CUTLASS MoE path reuses the Brutus vllm-xpu-kernels build at `65eb2861`. The general-purpose fork and Qwen package use the released vLLM 0.34 base. For isolated controls the flake also provides `vllm-xpu-inkling2`, `vllm-xpu-inkling2-cutlass` and `vllm-xpu-inkling2-sparse`. [Upstream Mini support](https://github.com/vllm-project/vllm/pull/69028) · [Sparse convolution history](https://github.com/vllm-project/vllm/pull/69444) · [Pinned plugin](https://github.com/mgaruccio/vllm-inkling2-xpu/tree/677db18d0129703b529270fe75052272f0e2834a)

KVarN packs sealed 128-token pages of the eight full target-attention layer visits using four bits per key and four per value. Mini has 128-dimensional heads, eight KV heads, and 24 query heads. A sealed head page occupies 17,920 bytes including scales and offsets. The 40 sliding-window visits and DSpark keep FP8 KV, convolution state uses sparse history, and the unfinished target page uses a bounded BF16 staging ring. The graph-compatible Triton writer and reader handle grouped-query attention, learned relative bias, gate/sink, speculation rollback, and partially filled pages restored from RAM. Each request admission has a generation number so a reused slot cannot inherit the previous request's staging contents.

Block-outermost physical layout (`BLNHC`) lets the allocator reserve packed page sizes. At 141,312 tokens the matched one-sequence planner reports 1.7965 GiB for K4V4 and 2.8103 GiB for FP8 before bounded staging. The Xe2 virtual-memory backend supports the extensible allocator. For long prefill the full recipe enables `INK2_KVARN_FLASH_PREFILL=1` to gather bounded chunks for native flash attention and merge older and recent history using their log-sum-exp. The reference checks cover relative-only, single-window and multiple-window prefill.

The host connector uses vLLM's `CPUOffloadingSpec` with active-only GPU retention. The xpu-v2.1.0 KVarN package submits XPU cache loads and stores on the current model stream and tracks completion with events. The fixed-budget comparisons above used the same plugin and native kernels with separate transfer streams. The RAM cache lives for the serving process. The replay tests exercise a live engine across requests and use the host-transfer and device-hit counters to establish where the prefix came from.

The two-phase decode reader (`INK2_KVARN_PHASED_DECODE=1`) scans the compact prefix in the rotated KVarN frame, changes basis once, and continues through the dense tail with the same FP32 softmax state. The B70 implementation uses 32-token tiles, eight warps and the normal split planner for queries up to eight tokens. Interleaved captured-graph measurements at 245,760 tokens reduced one full attention layer's median eight-token verification time from 9.09 to 5.37 ms. Shuffled pages and reordered request slots passed the independent dense-attention reference.

With `INK2_KVARN_MIXED_PREFILL=1`, an eager batch containing active decodes followed by one long prefill chunk uses the two-phase reader for decode and flash attention for the chunk. Four clients sent 83,234 input tokens in the same order, 150 ms apart. Wall time for all four answers fell from 100.48 s to 33.48 s on the same package. Both runs reached four active requests and returned the four correct codes. They generated 352 and 389 tokens respectively. A second comparison with simultaneous submission took 99.21 s and 38.86 s.
