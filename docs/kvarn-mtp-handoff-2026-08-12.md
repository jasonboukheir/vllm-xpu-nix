# KVarN compact K4V4 + two-token MTP handoff

Status: paused by request on 2026-08-12. Brutus vLLM remains disabled. Do not
enable the compact profile or treat these branches as production-ready.

## What is proven

- Two-token MTP means qlen=3: one target position and two speculative
  positions.
- Independent attention and Mamba pools reserve one null page per physical
  pool. Mamba capacity is sized independently for four resident states per
  request (committed, transition, and two candidates), rather than multiplying
  maximum-context attention capacity by MTP width.
- Partial-prefix copy-on-write operations are pool-qualified. The old merged
  block-ID stream could apply a Mamba-local block ID to attention storage and
  interpret pools using the wrong block count.
- The null/sentinel page is initialized once before warmup/graph capture;
  admission no longer zeroes all blocks in a new request because those block
  tables may contain shared prefix pages.
- Focused pool-copy and partial-prefix tests passed: 20 tests.
- Exact direct-vLLM compact graph prefix checks passed at 127, 128, 129, and
  4096 shared tokens. Greedy output was identical; maximum logprob delta was
  0.187 (gate 0.5).
- The qlen=3 compact graph lifecycle passed with deterministic replay, all
  acceptance lengths, consecutive-rejection recovery, 112/160 accepted draft
  tokens (70%), and token SHA
  `4df19b280e89ec1f5c48b4bd11c5a6c6856ada83d3d58070bb73d1b97964af3d`.
- A real 114,688-token request and independent recurrent capacity were proven
  in earlier artifacts described in `docs/kvarn-native-xpu-checklist.md`.
- Native compact-to-FP16 materialization removed the pathological cumulative
  Triton history reconstruction during long chunked prefill.
- Native Hadamard/scatter inputs are recorded on the active XPU stream, fixing
  allocator reuse corruption. Its B12 device median is about 57 us and host
  enqueue about 4 us, so it is not the remaining performance bottleneck.

## Why this is not shippable

The latest matched direct-vLLM B4/6000-input/512-output comparison fails both
performance gates:

| mode | output tok/s | mean TPOT |
| --- | ---: | ---: |
| native BF16 KV | 79.11 | 31.32 ms |
| compact K4V4 | 59.00 | 38.93 ms |

Compact reaches 74.6% of BF16 throughput (required at least 95%) and 1.243x
BF16 TPOT (required at most 1.10x). Functional/capacity results must not be
used to override this failure.

## Performance diagnosis and rejected approaches

The current native verifier treats B4 x qlen=3 as twelve virtual decode rows.
This is correct and fast in isolation (~0.79 ms at 6K) but reconstructs the
same request prefix independently for all three temporal positions.

- Direct reuse of the shared-prefix cached/chunk-prefill kernel is accurate
  (maximum difference ~0.000672) but takes ~23.44 ms because it has no split-K
  scheduling.
- An eight-query-row chunk specialization regressed to ~132.7 ms with severe
  tail latency. The 256-row/32-subgroup cooperative reconstruction is
  essential; reducing the tile is not a solution. The specialization was
  removed.
- Hoisting the virtual block-table expansion regressed serving performance and
  was removed.
- Cross-layer packing, persistent scratch as default, smaller native split
  counts, shared-dequant verification, and several reducer/scatter variants
  were previously measured and rejected; see the main checklist before
  reviving any of them.
- A 72-logical-head packed split-K prototype was started but did not compile
  cleanly and was completely removed before this handoff. No unproven dispatch
  from that attempt remains.

## Recommended continuation

Build a dedicated qlen=3 split-K verifier. One request/KV-head/split workgroup
should reconstruct each compact 64-token K/V tile once and score all three
temporal positions and six GQA heads. Emit per-position partial output,
exp-sum, and maximum-logit data in the established split-32 reducer layout.
Keep scratch caller-owned and graph-stable. Validate it first against the
current virtual-row oracle on randomized compact/tail mixtures and causal
boundaries, then run the full lifecycle before serving benchmarks.

Do not push configuration changes, enable Brutus, or run Open WebUI/LiteLLM
acceptance until the direct-vLLM BF16/K4V4 pair passes. The known-good
non-KVarN profile remains the rollback.

## Useful artifacts

- `/tmp/bf16-graph-poolcopy-fixed-all-prefix.json`
- `/tmp/kvarn-compact-graph-poolcopy-fixed-all-prefix.json`
- `/tmp/kvarn-compact-graph-poolcopy-fixed-mtp.json`
- `/tmp/kvarn-native-materialize-b4-6k-o512.json`
- `/tmp/bf16-local-matched-b4-6k-o512.json`
- `/tmp/kvarn-native-materialize-b4-6k-o512-perf-gate.json`
- `/tmp/kvarn-recordstream-scatter-b12-host.json`

The much longer chronological experiment record is
`docs/kvarn-native-xpu-checklist.md`.
