Keep the qualified **V1 + compact K4V2 + bundled MTP2** profile. On this short
Brutus screen, the v0.31 candidate delivered **9–15% higher median decode
throughput** than the preserved v0.30 release. MTP1 shortened first-token wait
slightly but made the full responses slower. These are screening measurements,
with one old-release arm and visible variation between the two candidate arms.

The [machine-readable results](performance-results.json) preserve exact package
and runtime identities, arguments, per-request metrics, output-token hashes,
comparisons and SHA-256 hashes of the local evidence. The
[feature review](performance-review.md) records the source-level V2/DFlash
findings; no V2 or external drafter was enabled for these measurements.

Runs on 2026-10-10 UTC used one Intel Arc Pro B70 and the actual Brutus model,
`RedHatAI/Qwen3.8-27B-INT4` at
`bf08f3dbd9a324e53956920aad378a1f1b6dd24a`. The service profile retained BF16
compute, W4A16 weights, compact K4V2 target and MTP caches, 262144 context,
2048 batched tokens, four maximum sequences, 0.96 memory utilization, eager
execution and disabled prefix caching. Test overrides were the candidate/old
executable, localhost port 18000, writable runtime cache and request logging.
The B arm changed only the number of speculative tokens from two to one.

Each concurrency level received two 64-token warmups, then four measured
requests: two repetitions each of a synthetic English library guide and simple
library arithmetic. Prompt lengths were 73 and 133 tokens. Measured requests
used temperature 0, seed 17, thinking disabled and `min_tokens=max_tokens=192`.
Concurrency two launched one request of each type together; every batch had
overlapping decode delivery. Arms ran in order A, B, C, A2; the first three used
concurrency 1 then 2, and A2 reversed that order.

Values below are medians over four measured requests per cell. Decode TPS is
per request, including each request in the concurrency-two case. Delivery
ms/token divides the first-to-last token delivery duration by tokens after the
first chunk; it is a client delivery estimate. Speculative acceptance is the
phase-wide accepted/proposed draft-token ratio.

| Arm | Profile | Concurrent requests | Decode tokens/s | First token, ms | Delivery ms/token | Request duration, s | Draft acceptance |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | v0.31 MTP2 | 1 | 46.41 | 213 | 21.58 | 4.326 | 70.00% |
| A | v0.31 MTP2 | 2 | 45.34 | 341 | 22.08 | 4.559 | 69.25% |
| B | v0.31 MTP1 | 1 | 42.43 | 195 | 23.57 | 4.690 | 82.38% |
| B | v0.31 MTP1 | 2 | 39.13 | 319 | 25.56 | 5.201 | 82.38% |
| C | v0.30 MTP2 | 1 | 42.45 | 216 | 23.60 | 4.727 | 68.63% |
| C | v0.30 MTP2 | 2 | 41.33 | 340 | 24.26 | 4.975 | 68.21% |
| A2 | v0.31 MTP2 return | 1 | 48.71 | 210 | 20.57 | 4.136 | 70.00% |
| A2 | v0.31 MTP2 return | 2 | 47.44 | 333 | 21.10 | 4.364 | 70.00% |

A and A2 exceeded C by 9.3–14.7% at concurrency one and 9.7–14.8% at
concurrency two. First-token latency was similar: candidate medians were
210–213 ms versus 216 ms at concurrency one, and 333–341 ms versus 340 ms at
concurrency two. A2 itself was 4.9% faster than A at concurrency one and 4.6%
faster at concurrency two. Order, warm state and other possible causes of that
variation were not isolated; the observed old/new difference should not be
presented as a precise expected gain for every workload.

MTP1 reduced first-token latency by 17–22 ms relative to A, but reduced median
decode throughput by 8.6% at concurrency one and 13.7% at concurrency two.
Median request duration increased by 0.36 and 0.64 seconds respectively. Its
higher acceptance fraction did not compensate for fewer proposed tokens per
step: each measured cell took 420 draft steps, versus 320–322 for A and 320 for
A2. The conventional `1 + accepted_tokens / draft_steps` was 1.824 for MTP1 and
2.385–2.400 for candidate MTP2. This supports retaining MTP2; the MTP1 screen
does not justify replacing the fully qualified release profile.

All 48 streams, including warmups, were independently checked against the raw
SSE captures. All 32 measured responses returned exactly 192 token IDs with
matching usage, normal length termination and valid timing. Raw metric
differences matched the recorded counters: each measured cell generated 768
tokens across four successful requests, with no counter resets, missing
samples or preemptions. Request bodies and prompt-token hashes matched across
arms. Raw SSE, request/result JSON, metrics and copied execution inputs remain
under the ignored local directory
`benchmark-results/release-v031-20261010/performance/`, in `mtp2-01`, `mtp1-01`,
`baseline-mtp2-01` and `mtp2-02`.

Output sequences were not universally identical despite temperature zero.
Seven of eight A/C pairs and all eight A2/C pairs differed; initial divergence
was at token index 25 for English or 117 for arithmetic. Seven of eight A/B
pairs matched, all eight B/A2 pairs matched, and seven of eight A/A2 pairs
matched. This is a comparison of matched prompts and fixed output lengths,
with output-dependent work and acceptance included. It does not establish
identical-output model behavior or measure answer quality.

The candidate source commits were vLLM
`ef92dc5600af64103c01f97e3300b81c3a1083ee` and kernels
`dbb684f4c9b38a8b3bbf52bda0c490ee47049221`. A, B and A2 used the same built
package, kernel and runtime outputs recorded in
[candidate-derivations.json](candidate-derivations.json); only descriptive
qualification-status text changed in the embedded A2 metadata. C used the
preserved `xpu-v1.11.0` package, vLLM
`466b9d5abe84f13c29b38c686e4d77fc09e09aa4` and kernels
`193cee080d654ff67c4f8bfaf924201bc88b9cf5`. Its explicit OpenCL ICD was exactly
the new wrapper's pinned default. C therefore compares the old package in the
current environment with matching working discovery; it does not reconstruct
the original failing host service.

The upstream Qwen Q/K normalization, RoPE and gate fusion in
`ccbaed5b91e09f528180db363af7ebb9d4fafe32` is applicable to this target according
to source inspection. This experiment measures complete package stacks and
does not isolate that fusion's contribution. V2 plus the Inco DFlash2 drafter
still requires separate work on the compact-cache lifecycle and qualification,
as detailed in the feature review.

The sample is too small for confidence intervals or meaningful tail-latency
claims: the harness's nearest-rank p95 is simply the largest of four requests.
These short text requests do not measure image performance, long-context
prefill/decode, natural stopping, enabled thinking or saturation throughput.
Speculative tokens arrive in bursts, so SSE inter-chunk gaps and normalized
per-token delivery estimates do not reveal individual GPU token times. The
server's `inter_token_latency_seconds` histogram measures engine output-event
intervals under MTP; the JSON labels it accordingly. Separate release
qualification covers operational correctness and capacity.
