# Post-release review, 2026-10-20

This follow-up records additional checks requested by the independent release
plan and idle-delay reviews. The published `xpu-v1.13.0` tags remain unchanged.
The final release manifest and `finish-serving-validation.json` supersede the
intermediate K1 candidate's pending qualification status.

## Host recipe and cache

Host commit `ac02f115` updates both the nested release input and the complete
model recipe. Evaluated serving arguments match the qualified package and
profile after normalizing JSON and numbers and excluding the test port and
request logging. Evaluated service environments differ only in `HOME` and
`VLLM_CACHE_ROOT`; qualification also enables reset instrumentation, offline
model access, private cache/config directories, and disables usage statistics.

The launcher derives its arguments from the captured v1.12.2 unit and explicitly
applies the runner, drafter, utilization and threshold changes. Its original
`overrides` list omitted those profile changes; later manifests record them
separately. The recorded final argv, rather than that list, was compared.

Both pinned snapshots exist in the shared cache, including all weight files.
The evaluated cache roots include the target revision and
`syvai/Qwen3.8-27B-DFlash2-W4A16` at
`4d30ec736ffc6b8688dc2ae2b502d9b48bdec279`.

## Repeatability on the shipped profile

The six-request matrix was repeated on the exact released package
`c0722c9`, V2, two DFlash2 proposals, utilization 0.98 and prefill threshold 1024.
Each request used temperature zero, seed 17, 8,247 input tokens and 256 output
tokens. No other GPU workload ran during the matrix.

| Comparison | First different output token, zero-based |
| --- | --- |
| Cold 0 / cold 1 | 20 |
| Cold 0 / cold 2 | 210 |
| Cold 1 / cold 2 | 20 |
| Cold 0 / each RAM repeat | 20 |
| Cold 1 / each RAM repeat | Identical |
| Each RAM repeat / every other RAM repeat | Identical |

Every cold request computed all 8,247 tokens without an external hit. Every
RAM request restored 8,064 tokens / 314,621,984 bytes and recomputed 183 tokens.
Raw results are in
`benchmark-results/release-v032-v2-20261019/post-review-k2-01/determinism-matrix`.

Cold-repeat variation is a regression relative to the three identical stable
v1.12.2 cold controls. This result does not establish its cause, semantic
equivalence, or a general quality guarantee. The release remains suitable for
the intended interactive profile based on its separate structured-output,
retrieval, cache-lifecycle and performance checks, with deterministic replay
explicitly outside its guarantees. Keep v1.12.2 for workflows that depend on
its observed repeatability. Kernel determinism patches do not guarantee
determinism of the complete speculative model execution.

## Shared GPU check

Two synthetic 4K HEVC to 1080p H.264 QSV transcodes using the installed
Jellyfin FFmpeg and configured media-driver/VPL stack ran concurrently with
four-way chat decode and RAM restore. All chat checks passed, including matching
restored tokens, positive draft acceptance and no preemptions. Both transcodes
maintained approximately real-time progress and used about 247 MiB of VRAM each.

Restarting the released profile with both transcodes already running failed:
29.45 GiB was free, below its 29.69 GiB startup requirement. Starting chat first
therefore does not qualify arbitrary restart order. The host needs a separate
GPU for media or an explicit operational restriction before concurrent service
use. These synthetic SDR checks do not cover HDR, subtitles or all codecs.

The spare AMD GPU passed short hardware decode/encode and HDR10-to-SDR
VA-API/Vulkan probes. Two simultaneous synthetic 4K HDR10/30 fps to 1080p
H.264 tone-mapped streams sustained about 24 fps each, below real time for
that input. vLLM passed device initialization while those jobs used the other
GPU and finished startup after the media probes had been stopped.
Jellyfin's host configuration has not been moved: choosing media throughput,
chat capacity, or a restriction on restarting chat during transcodes needs an
operator decision. Raw media commands, progress and logs are retained under
`build-dev/release-v032-v2-20261019/review-round-01/`.

## Idle delay and remaining measurements

Historical journals place most observed stalls in slow RAM-tier transfers,
with multi-gigabyte service swap peaks. They support memory reclaim as the
leading hypothesis, but do not identify which pages faulted at each event.
Host commit `eb028ada` prevents ordinary swap in the chat cgroup and provides
passive before/after snapshots of process and cgroup faults, swap, ARC and GPU
power counters. It does not claim that all driver allocations are protected or
that overnight behavior has been fixed.

One matched cold sample per threshold measured a 16,435-token prompt arriving
after a separate 1,024-output-token decode had started. Both used the released
package, K2, utilization 0.98, fixed seed, and identical request bodies apart
from cache isolation salts. No transcodes ran during these samples, and neither
sample had external cache hits or preemptions.

| Measurement | Threshold 0 | Threshold 1024 |
| --- | ---: | ---: |
| Long prompt time to first content | 11.890 s | 12.440 s |
| Competing decode maximum content gap | 1.483 s | 0.846 s |
| Decode content events during long prefill | 12 | 19 |
| Competing decode completion time | 33.949 s | 34.635 s |

The observed cost was 0.549 s (+4.6%) to the long prompt's first content, with
a smaller maximum decode gap. This is one sample per configuration in
separate server sessions, not a statistical performance guarantee. The
threshold remains enabled for the loaded-progress behavior established by
the release tests. Raw results are in
`benchmark-results/release-v032-v2-20261019/post-review-threshold-{0,1024}`.
