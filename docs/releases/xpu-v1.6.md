# xpu-v1.6 — KVarN prefill and Qwen image support

Beta continuation for Intel Arc Pro B70. This release keeps the qualified
KVarN implementation and retires historical runtime experiment selectors.
Frozen xpu-v1.5 refs and their validation evidence are unchanged.

## Changes

- Improved Sinkhorn flush: the qualified FP16-provenance reload implementation
  preserves FP32 normalization and the general FP32 correctness path.
- Qwen3.5-family image inputs are enabled for the qualified AEON W4A16 model.
  The released unstable package includes torchvision.
- The B70 compact cache selects the qualified Xe2 DPAS decoder (ID18), adaptive
  split schedule, current-stream inline frontend, native writer/materializer,
  and fused-materialized Sinkhorn without tuning environment variables.
- Historical decoder variants and host-side trial implementations are removed.
  Retired environment overrides fail at startup. Necessary shape/platform
  fallbacks, memory-budget controls, and diagnostic observations remain.
- Native eager XPU startup no longer autotunes unused generic Triton decode
  paths. A sampled startup stack located the cost in Triton binary loading
  during `_warm_decode_kernels`; this is a startup cleanup, not a separately
  measured serving-throughput improvement.
- Removed the public optimization-factory app and candidate package surfaces.
  A fixed `vllm-xpu-kvarn-validation-env` supports reproducible validation.
  Historical factory documents and evidence are explicitly archived.
  The narrow validation build includes the Qwen vision encoder's noncausal
  head-dimension-96 kernel, as well as the text attention kernels.

## Measured improvement and limits

Before the cleanup, inclusive native-writer flush measurements improved by
1.36× / 1.39× / 1.72× for 1 / 4 / 16 pages. The matched 65,023-token prefill
experiment reduced the KVarN-minus-auto service gap from 2.609 s to 2.026 s
(about 0.583 s). These are workload-specific results, not a service-parity
claim. Removing switches itself has no separately established speedup.
See the [causal investigation and remaining work](https://git.sunnycareboo.com/jasonbk/vllm-xpu-nix/issues/5).

The image qualification is bounded to the actual Qwen3.5-family AEON W4A16
model, BF16 activations, B1, V1/eager execution, no prefix cache or speculation,
up to two 448×448 images, and video disabled. It includes OCR, changed-image
controls, multiple images, multi-turn requests, and compressed image history
at 6,143 prompt tokens. See [vision qualification](../kvarn-vision-qualification.md)
for the model identity, raw evidence, measurements, and limitations.
MTP, DFlash, video, other vision architectures, and broader vision concurrency
are not qualified by this release.

## Usage

Use `--kv-cache-dtype kvarn_k4v4_g128_compact` with the supported service
settings. Remove old KVarN experiment overrides from the server environment.
Use `--kv-cache-dtype auto` to return to automatic cache storage.

Build the reusable validation runtime:

```sh
nix build .#vllm-xpu-kvarn-validation-env -o result
./result/bin/python scripts/kvarn_vision_run.py --service-env ./result \
  --cache-dtype auto --qualify --output /tmp/vision-auto
./result/bin/python scripts/kvarn_vision_run.py --service-env ./result \
  --cache-dtype kvarn_k4v4_g128_compact --qualify --output /tmp/vision-kvarn
```

Run only one GPU experiment at a time. Release creation does not deploy or
restart the production server.

## Final clean-source validation

Local qualification passed on 2026-09-06 for vLLM
`f9a7a62a1ae02d2b33385663c97049172f98f4c9` and kernels
`767dc3ddf3a614f765e34b566ce788cb79bb2798`:

- 368 relevant vLLM CPU tests, including retired-override guards, fixed
  request-stability defaults, native dispatch, and block-table lifecycle.
- 117 native/static/layout regression tests, with real B70 execution against
  the explicitly identified rebuilt native library.
- Matched auto/KVarN vision runs: all eight correctness pairs passed manual
  fixture review and produced identical text; raw SSE and provenance checks
  passed. Both 6,143-token prompts exercised compressed image history and
  crossed a decode page boundary.
- No missing-kernel/reference-attention fallback warnings in either final arm.
- The earlier packaging Nix check passed 471 tests. The final remote-pin
  check remains pending SSH authentication and lock refresh.

Final warmed medians (three 96-output-token samples per arm):

| Workload | Auto TTFT | KVarN TTFT | Auto decode | KVarN decode |
| --- | ---: | ---: | ---: | ---: |
| Single image | 168.6 ms | 170.2 ms | 32.58 tok/s | 31.43 tok/s |
| Text only | 77.3 ms | 98.6 ms | 32.80 tok/s | 32.13 tok/s |

The long-image TTFTs were 3.773/3.772 s for auto and 3.875/3.880 s for KVarN.
These bounded functional-run measurements do not establish service parity
or a serving-speed improvement from removing switches.

Runtime: `/nix/store/nipzzx7r43lsi7jya27njmk9vwz4lf36-vllm-xpu-kvarn-validation-env`.
Raw evidence: `benchmark-results/kvarn/xpu-v1.6/`, using only
`vision-auto-qualified` and `vision-kvarn-qualified` as the final matched pair.
`comparison.json` records verified inputs, responses, timing, and sampled
memory; `validation/commands.md` records exact commands and limitations.
The earlier interrupted startup is diagnostic evidence, not a passed capture.
One transient `/proc` sampling error and a shutdown semaphore-cleanup warning
are retained; the managed services exited. The general multi-device package
was not rebuilt; qualification used the documented narrow B70 build.

**Publication pending:** the final branch push, lock refresh/check, coordinated
tags, and release pages must finish after the local SSH authentication agent
responds. No xpu-v1.6 tag or release page has been published. Do not present
this draft as an already published release.
