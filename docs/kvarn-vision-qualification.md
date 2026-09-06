# KVarN vision qualification

This report preserves the **pre-cleanup baseline** qualification. For the
final cleaned xpu-v1.6 source identities and rerun results, see the
[release notes](releases/xpu-v1.6.md). Historical experiment build expressions
and runtime paths below describe that baseline, not additional release paths.

Qualified on B70 on 2026-09-06 UTC for the bounded image/text envelope below.
The only vision runtime change was narrowing the experimental multimodal guard
to allow `qwen3_5` images with video disabled. No attention kernel change was
needed. Production has not been deployed or modified.

Selecting `--kv-cache-dtype kvarn_k4v4_g128_compact` already selects the blessed
XPU profile. The qualified runs passed **no KVARN_ environment overrides**.
The logs confirm Xe2 layout, ID18 decode with `b70_q6_id18_v1`, the native
writer/materializer, and `fused_materialized` Sinkhorn selected by defaults.

## Results and supported envelope

Both arms used the actual AEON W4A16 AutoRound target, BF16 activations, V1,
eager mode, no prefix cache or speculation, B1, a 2,048-token prefill budget,
8,192 configured max-model-len, and 90% GPU-memory utilization.
Fixtures were 448×448 RGB PNGs, with processor min/max pixels both 200,704.
Each image produced **196 actual image tokens**; two images produced 392.

| Correctness check | Auto | KVarN |
| --- | --- | --- |
| Shape/color recognition, left/right positions, OCR | Pass | Pass |
| Changed image with identical question | Pass | Pass |
| Two images, ordered OCR | Pass | Pass |
| Image-containing multi-turn conversation | Pass | Pass |
| Counting without supplying the count in the question | Pass | Pass |
| Text-only request after image requests | Pass | Pass |
| Two changed-image controls at 6,143 prompt tokens | Pass | Pass |

All eight correctness pairs were reviewed against the actual fixtures, not
just accepted because the API returned HTTP 200. Short descriptions differed
only in wording/punctuation. Both long descriptions matched exactly across
arms. Exact token equality is not a general requirement for lossy KV storage;
the auto arm itself varied wording across repeated text-generation requests.

For both long requests, image tokens were at positions **266–461**, entirely
beyond the 128-token uncompressed sink. `/tokenize` and generation usage agreed
on 6,143 prompt tokens. Three prefill chunks were required by the 2,048-token
budget. KVarN logged 15 older blocks queued for continuation-prefill flush,
31 for decode flush, and actual fused Sinkhorn execution. Generation continued
through token position 6,144 into the next 128-token page. The largest tested
request totaled 6,173 tokens. This exercises compressed image history, not just
images retained in the recent FP16 window.

Warmed profiler-off medians, three serial samples per workload/arm, 96 output
tokens, after one workload-specific warmup:

| Workload | Auto TTFT | KVarN TTFT | Auto decode | KVarN decode |
| --- | ---: | ---: | ---: | ---: |
| Single image, 222 prompt tokens | 165.8 ms | 170.1 ms | 32.33 tok/s | 31.41 tok/s |
| Text only, 25 prompt tokens | 76.9 ms | 97.8 ms | 32.45 tok/s | 32.11 tok/s |

The two long-image TTFTs were 3.769/3.778 s for auto and 3.882/3.875 s for
KVarN. These are functional-run timings, not warmed long-context benchmarks.
Short-workload decode rates use `(completion_tokens - 1) / (total_time - TTFT)`
from client-observed SSE timing. CPU-only fdinfo sampling ran in both arms;
its overhead was not separately measured. These results do not certify
performance parity or explain the remaining small gaps. No new profiler or
counter collection was needed for the functional qualification.

## Memory and capacity

| Measurement | Auto | KVarN |
| --- | ---: | ---: |
| Model-loading memory, startup log | 17.56 GiB | 17.56 GiB |
| Available KV memory, startup log | 8.30 GiB | 7.25 GiB |
| Sampled peak resident VRAM, owned DRM clients | 27.628 GiB | 27.636 GiB |
| Samples, nominal 0.5-second interval | 179 | 170 |

Raw kernel fdinfo samples are retained. Client IDs were deduplicated per PCI
device; reported shared VRAM was zero. This is a sampled resident peak, not an
instantaneous allocator high-water mark. Both arms reserve most of the GPU
budget, so similar residency does not imply identical per-token KV cost.

Auto reported 102,715 cache tokens. KVarN's headline 8,192-token/1× figure is
limited by the recurrent-state pools reserved for `max-num-seqs=1`; it is not
the total attention-pool capacity. Its attention pool had 3,329 usable
128-token blocks = **426,112 token slots**, while each of its three recurrent
pools had one usable live-request block. The 8,192 service limit remains
configured; contexts beyond the tested 6,143-token prompts are not qualified
by this report.

## Evidence checklist

Evidence root: `benchmark-results/kvarn/vision-20260906/`.
Immutable full archive:
`/nix/store/r115jqs3w8vi8zbisp6v87j6am39096f-vision-20260906`.
Repository-local GC roots retain the archive at
`benchmark-results/kvarn/vision-evidence-20260906` and the runtime at
`benchmark-results/kvarn/vision-runtime-20260906`. The archive's
`auto-qualified-01/harness-source/` also contains the exact capture sources.

| Goal requirement | Authoritative evidence |
| --- | --- |
| 1. Actual auto baseline, pinned inputs/runtime/device | `auto-qualified-01/manifest.json`, both workload manifests, `hardware-preflight.json`, `model-assets.sha256` |
| 2. Minimal vision implementation, real KVarN path | `vision-runtime.patch`, immutable package, `kvarn-qualified-01/service.log` |
| 3. Matched functional matrix, changed images, chunk/page lifecycle | `comparison.json`, raw requests/responses/SSE, PNGs, both `long-image-*-tokenize.json` files, flush logs |
| 4. Memory/capacity, actual image tokens, bounded long context, timings | Both service logs, `memory-fdinfo.jsonl`, per-request usage/timings, `comparison.json` |
| 5. Regressions, immutable sources, commands, limitations | JUnit files, captured `harness-source`, archived final verifier/tests, commands below and limitations here |

The comparison verifier checks source/image/workload/log hashes, full request
bodies, raw SSE versus parsed output, matching runtime arguments/environment
and Nix closures, processed token counts, default-only KVarN selection, and
long-context/page coverage. It rejects changed or mismatched inputs.
`qualification-checklist.json` also records the manual semantic review.

CPU regression results: **189 packaging/harness tests**, **306 golden-source
KVarN/platform tests**, plus **11 platform tests and the one no-override dtype
default test against the pinned runtime**. JUnit results are preserved.
The pinned default test pre-imported the Nix vLLM and used pytest importlib
mode to prevent the golden source directory from shadowing the package.
Ruff checks/format checks passed for the new harness and verifier.

## Runtime and replay

- Target: `jasonboukheir/Qwen3.8-27B-AEON-Ultimate-Uncensored-BF16-W4A16-AutoRound`, revision `6b0622f4354481d5d04577d48ba0db844efc1330`.
- Service environment: `/nix/store/9pnrkj7zx5h6syplr2s6vcgsp9yz9lwm-vllm-kvarn-vision-service-env` (`/tmp/kvarn-vision-service-env`).
- Package: `/nix/store/ils5aqib7shm7g37688hrasixby56nvm-vllm-kvarn-vision-package`.
- Native kernels: `712748aa63d5ddb5ffb7d2df4fcadc3a513fbad5`, pinned in the recorded closure; Torch `2.13.0+xpu`, IGC `2.38.2`, Level Zero `1.32.0`, compute runtime `26.27.39122.11`.
- Captured harness snapshot: `/nix/store/cj7fl7q1j4pjl2j64flgr32b4r001dzk-harness-source`.
- Transport settings are encoded directly in the released harness; it has no
  dependency on the historical diagnostic-service-profile artifact.

The historical build expression was `nix/kvarn-vision-experiment.nix` in the
experiment workspace (not shipped in this release). It overlaid only the
platform file onto the accepted immutable Sinkhorn package. The mutable golden
backend contains a pre-existing optional reference-writer fallback condition
absent from that accepted package; it was deliberately not included in this
vision overlay. Default native operation was validated against the actual
package, not inferred from the mutable source tests.

Re-run the bounded matrix sequentially, using fresh output directories:

```bash
nix build .#vllm-xpu-kvarn-validation-env -o result
./result/bin/python scripts/kvarn_vision_run.py \
  --service-env ./result --qualify \
  --output benchmark-results/kvarn/vision-20260906/auto-replay
./result/bin/python scripts/kvarn_vision_run.py \
  --service-env ./result --qualify \
  --cache-dtype kvarn_k4v4_g128_compact \
  --output benchmark-results/kvarn/vision-20260906/kvarn-replay
```

These commands require the authorized escalated GPU execution path. The runner
owns/stops its service process group, binds loopback port 8017, removes inherited
KVARN overrides, and does not modify a deployed server. It omits
`--language-model-only` and uses `--limit-mm-per-prompt '{"image":2,"video":0}'`.

Verify existing evidence without GPU work:

```bash
./result/bin/python scripts/kvarn_vision_compare.py \
  --auto benchmark-results/kvarn/vision-20260906/auto-qualified-01 \
  --kvarn benchmark-results/kvarn/vision-20260906/kvarn-qualified-01 \
  --output /tmp/kvarn-vision-comparison.json
```

For byte-exact capture-source replay, restore the five captured source files
under `scripts/` in a separate checkout. The released runner also takes an
explicit immutable service environment and embeds the transport defaults;
it no longer depends on the historical diagnostic-profile artifact. New
clean-source release captures preserve their own exact harness snapshots.
Verify model assets with `sha256sum -c <evidence-root>/model-assets.sha256` from
the pinned offline model snapshot directory. Digests were collected after the
matched captures and support future replay/input-change detection.

## Limits and preserved state

This is a bounded synthetic recognition/OCR/spatial regression qualification,
not a broad vision-accuracy benchmark. Larger effective resolutions, larger
contexts, higher concurrency, arbitrary photos/documents, sampling quality,
prefix caching, graph mode, V2, video/audio and speculative decoding were not
qualified. DFlash2, MTP and tuning remain deferred.

One auto shutdown emitted a resource-tracker semaphore-cleanup warning;
all managed engines/process groups exited. A first preliminary KVarN launch
failed because the experimental harness copied inconsistent auto overrides;
its log is retained as `kvarn-smoke-01`. It is not evidence of a vision failure.

Frozen `xpu-v1.5` refs remain vLLM
`3b6bc5d08c6446ba0ce5f52614976f0f25f4d4f8` and kernels
`39a770cc2a025d7de27b5bb6ed43a025e67b5405`. Accepted Sinkhorn source SHA256
remains `bea61792b485dd0b90cb4e7ce5ee2f62728246c0acd5f0dfd6d1f633e5e59243`.
During this baseline qualification, no host-permission changes, production
deployment, external publication, release-ref edits, or unrelated kernel
changes were made. Release publication is tracked separately in its notes.
