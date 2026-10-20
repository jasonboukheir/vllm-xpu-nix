# v07 — Bounded RAM storage for KVarN hybrid prefixes

**Disposition: ADAPT.** Retain the bounded byte arena, pool-qualified transfer
fences, actual per-pool worker views, KVarN page materialization/sink sidecars,
restore invalidation, and `active_only` policy. Upstream v0.32 does not replace
these requirements. Port their lifecycle to V2 and extend the group accounting
to the selected DFlash draft. Fold the surviving change and v08's completeness
repair into the coherent KVarN feature, coordinated with v05/v06.

## Frozen inputs and scope

| Item | Identity |
| --- | --- |
| Repository | `/home/jasonbk/Projects/vllm` |
| Assigned commit | `a76fbb1ce06d50159978786f70daf44c27847ddd` |
| Parent | `81dc387cf9a80e88468b7f77f7df8dfb71939921` |
| Subject | `Cache idle KVarN hybrid prefixes in bounded host memory` |
| Patch ID | `9644dd93acbbfb190393ddeca564abe94b08422f` |
| Old upstream | `db9527a46873454610df6dbedf79a36d6bf1a7f6` |
| Final old stack | `dcba4007ef6af8ba143d2794c194c36b58397339` |
| Target upstream | `5df63a7203cdb9328d824a27b682e77c7b9928be`, v0.32.0 |
| Kernel old/new base | `ddf336d86e3c8602888572a3502f951abd51df12` |
| Kernel tip / native KVarN feature | `dbb684f4c9b38a8b3bbf52bda0c490ee47049221` |

Read source `AGENTS.md`, the runbook's `references/commit-audit.md`, this run's
immutable inventory and v05/performance reports, and the stable release
manifest. All source evidence below comes from these frozen Git objects.
Only this report was written. No source, Git state, lock, service, or external
issue was changed; no runtime test was executed.

## Requirement and effective old behavior

The feature keeps reusable idle Qwen3.8 hybrid prefixes in an 8 GiB host
arena while leaving active state on the XPU. It avoids padding each compact
attention page to the much larger recurrent-state page size. The stable
profile is Arc Pro B70, RedHat Qwen3.8-27B INT4, compact K4V2 target/MTP2,
262144 combined context, four scheduler slots, images, and `active_only` GPU
retention. Four slots do not imply four simultaneous maximum-length contexts.

`BytePoolOffloadingManager` allocates aligned variable-size pages transactionally,
pins outstanding stores/loads against eviction, and uses bounded LRU eviction.
`IndependentCPUOffloadingWorker` copies real page byte ranges with the existing
batched DMA operator, synchronizes before returning, and preserves FP16 sink
sidecars. KVarN's `prepare_cpu_offload` flushes only complete committed resident
pages; `prepare_cpu_reload` clears stale destination ownership and queues the
sink representation for the next metadata build. The connector qualifies
reuse fences by physical pool, while worker/canonical views use
`num_blocks_for_group`, not the legacy minimum capacity.

`active_only` evicts GPU hashes when the last reference is released and omits
the extra idle recurrent checkpoint reserve. Preserve null pages and the
legacy implicit zero-capacity metadata configuration; explicitly configured
physical pools must remain positive.

v08 `dcba4007ef6af8ba143d2794c194c36b58397339` repairs a real dependency:
V1 can stop proposing when the target prefill fits but speculative tokens do
not. It still must populate bundled MTP context pages before they become a
host-cache hit. Its zero-proposal first forward and exclusion of padded
no-write rows from KVarN ownership/fill decisions remain requirements; the V2
implementation must satisfy them through its actual call path.

## Upstream coverage

A frozen-target search finds no `BytePoolOffloadingManager`,
`IndependentCPUOffloadingSpec`, `prepare_cpu_offload`, `prepare_cpu_reload`, or
`_group_pool_offsets`. Generic CPU offload and V2 connector support are useful
infrastructure, but do not serialize local compact pages and resident sinks or
provide independent device-pool ownership.

Relevant old-to-new changes to preserve:

- `b027ac831253e83ff605a2ff5d4aa7c00619802c` adds XPU pinned registration
  for shared mmap offload. It does not replace the local byte arena, which
  already uses `torch.empty(..., pin_memory=True)` and a single worker.
- `2bea22002d66a6c89d208fa682b40a89538f023c` fixes shared-memory worker
  rank selection. Retain upstream paths; do not substitute them for per-group
  capacity or widen the local single-worker contract without validation.
- `ca0df1f1227cdd993835a4d2a62861dd3bd2e893` exempts precisely the blocks
  written by asynchronous loads from zeroing. Reconcile its scheduler/cache
  manager semantics with v05's pool-qualified IDs; do not revive the old
  broader zeroing exemption while replaying v07.
- `e37e51dd246cf421c554e7d6f53e178e3fd29085` adds cache-tier attribution
  and load-key ranges to the offload scheduler. Preserve them alongside the
  local physical fence IDs. `OffloadingManager.get_load_source` now defaults
  to `EXTERNAL_UNSPECIFIED`; give the byte arena the same `HOST` override as
  upstream's CPU manager so actual host hits remain observable.
- `cb6458c5ce61632524c0447d49790725c1e1d9a2` removes Mamba mode `all`.
  Keep target `align` semantics. None of this removes sidecar completeness.

## Minimum V2/DFlash adaptation and release hazards

### Run transfer fences before mutable KVarN metadata

At target `vllm/v1/worker/gpu/model_runner.py:1915`,
`model_state.prepare_attn` builds metadata. Connector `pre_forward` follows
at lines 2003/2028. In `gpu/kv_connector.py:81`, that hook first calls
`handle_preemptions`, and lines 87–89 then start synchronous loads.

This is too late for the local KVarN contract. Old-final
`kvarn_attn.py:2067–2095` explicitly requires reload after the fenced page copy
and **before the next metadata build**; it removes residency/fill state and
bumps the allocator epoch. Metadata constructed first can reserve or cache
the very mappings that reload subsequently invalidates. Moreover,
`offloading/worker.py:219–244` submits pending stores and required reuse flushes
inside `handle_preemptions`; KVarN metadata must not reclaim/repurpose their
resident tails before materialization.

Add a narrowly scoped pre-metadata transfer phase for this connector/backend,
or split the connector lifecycle so required stores/reuse fences and restored
KVarN state complete before mutable metadata. Keep the generic connector's
kwargs/async behavior intact. Retain the no-forward drain and reset paths.
Verify any earlier state-zero/copy consumers are ordered with these fences.

Do **not** move all store preparation to target-forward completion:
`model_runner.py:2046` calls `finish_forward` before speculation, but actual
OffloadingConnector stores are queued by `wait_for_save`, through
`post_forward` at line 2276, **after** `speculator.propose` at line 2239.
`offloading/worker.py:259–270` defers these jobs to the next step. That ordering
allows draft context precompute to complete before new prefixes are saved and
must survive the adaptation.

### Preserve draft context in each reusable prefix

At target `gpu/spec_decode/dflash/speculator.py:383–389`, the drafter combines
only this target batch's auxiliary hidden states; at line 493 it precomputes
context KV before building draft query metadata at line 500. The inspected
DFlash/DFlash2 model/speculator paths contain no hidden-state prefix cache or
reconstruction of target rows skipped by a prefix hit.

The minimum approach is to include the drafter's **projected KV for the needed
sliding window** in host stores/loads and require complete draft-window,
target-attention and recurrent-state coverage at the same accepted boundary.
Upstream offloading already has sliding-window group lookup and
`get_offloading_group_ids` selects prefix-cacheable groups; verify the actual
draft groups participate and are registered with their actual capacity.
Then a hit need not reconstruct all skipped auxiliary history. If draft KV is
excluded or missing, bound the hit/recompute explicitly; restoring only target
KV and starting the drafter with empty old context is not a correct substitute.

Preserve rejected-context/null-page masking at `dflash/speculator.py:582–634`.
If choosing a compressed draft, add its pre-write reserve/materialization
lifecycle before context precompute; query-only metadata is too late. This is
the v05-owned KVarN draft integration dependency.

### BF16 draft groups need distinct accounting

Old-final `offloading/config.py::_group_sink_bytes` classifies every
`AttentionSpec` using global target `cache_dtype`. With K4V2 target and BF16
DFlash draft it would assign KVarN sidecars to the BF16 group.
`IndependentCPUOffloadingWorker._transfer` then calls
`KVarNAttentionImpl.prepare_cpu_offload` whenever `sink_size_bytes` is nonzero;
that method indexes `by_name[name]` for every group layer (old-final
`kvarn_attn.py:2018–2019`). A non-KVarN draft is absent from this registry.
Use actual per-group format/implementation metadata for sidecar eligibility
and bytes. A BF16 group uses plain page copies and zero KVarN sink bytes;
global target type is not an appropriate discriminator.

The old independent allocator also needs v05 adaptation:
`core/kv_cache_utils.py:1940–1945` reserves only Mamba at `max_num_seqs`, while
lines 1960–1979/2027–2028 give **every non-Mamba group** the shared full-context
token capacity. A BF16 sliding draft is therefore not automatically a small
2048-token allocation. Reserve bounded sliding groups for all four admitted
slots, including lookahead/chunk scheduling, alignment and null pages, then
allocate the remaining budget to full target attention. Reconcile admission,
override, reporting and worker views; `_independent_pool_bytes_for_requests`
currently multiplies only Mamba by `max_num_seqs`.

For the candidate five-layer D128/KV8 draft, 2048 tokens of raw BF16 K/V is
40 MiB per request, 160 MiB for four. This is arithmetic excluding scheduler
overlap, alignment/lookahead and workspaces, not a measured allocation or
evidence that the existing planner achieves it. Preserve target K4V2 and 262K.

### v08 boundary dependency: mask out-of-range draft writes

Target V2 calls the speculator even when `num_spec_tokens` is reduced, then
masks returned proposal IDs (`model_runner.py:2239–2256`). Thus blindly adding
the V1 zero-proposal fallback duplicates a call path V2 already executes.
Prove committed context is populated and preserve that execution at the ceiling.

An additional exact-code hazard is target
`dflash/speculator.py:642–670`: query block indices clamp to the last block-table
entry, query positions clamp to `max_model_len - 1`, but write-slot eligibility
only checks nonzero block ID. It does not check `query_pos < max_model_len`.
When an out-of-range query maps to a nonzero last table entry, its modulo page
offset can alias committed final-page KV. V2 sizes those tables using
`max_model_len` (`model_runner.py:616–649`). Treat this as a source-derived
boundary risk requiring an explicit no-write predicate and a regression at the
real table width; no GPU failure was reproduced in this audit. Preserve valid
context stores while padding invalid queries. Merely clamping RoPE positions
or suppressing returned draft IDs does not protect cached bytes.

## Dependencies and validation

Prerequisites are v05 `ef92dc5600af64103c01f97e3300b81c3a1083ee` (format,
independent pool schema, ownership and materialization), v06
`81dc387cf9a80e88468b7f77f7df8dfb71939921` (retained-prefix alignment),
and paired kernel k07 `dbb684f4c9b38a8b3bbf52bda0c490ee47049221`.
Keep native compact format/layout and DMA ABI matched. v04 or equivalent
native dtype handling is needed if selecting a BF16 draft. Fold v08's surviving
completeness/padded-row repair into the same feature; its owner should reconcile
the V2/DFlash ceiling behavior above.

**Observed:** frozen source/diff/history inspection and stable recorded results.
The `xpu-v1.12.2.json` manifest records 1310 installed CPU passes, two skips,
115 successful API responses, 306688 shared attention tokens, and 262144
per-request context. Installed 262015-token image RAM hits measured
2.690/1.749 seconds TTFT; 20307-token text measured 14.606 seconds cold and
0.431 seconds on RAM reuse. These are previous V1+MTP2 results, not evidence
of candidate V2 correctness/performance.

**Required:** rerun existing byte-manager, connector scheduler/worker,
independent-pool, KVarN serialization and mocked DMA roundtrip regressions.
Add/extend focused behavior checks for transfer-before-metadata ordering,
cross-pool equal numeric IDs, BF16 draft groups without KVarN hooks, complete
draft-window restore, reset/cancellation with pending stores, and invalid
ceiling queries leaving final committed bytes untouched. Assert host-budget
enforcement and pool capacities under four active requests. Preserve upstream
async-load zeroing and cache-tier metrics tests.

On Brutus, compare matched cold/host-hit/continuation outputs and timings with
V2 no-spec, V2 MTP and selected DFlash2, including two images, concurrent mixed
prefill/decode, near-limit context, eviction/reload and repeated slot reuse.
Check restored draft-window bytes/acceptance as well as final text: target
verification can conceal a damaged drafter through poor acceptance. Measure
actual host/device allocations and per-request TTFT/decode performance.

The user's long-idle latency remains a separate timing investigation. This
feature contains no daily cache expiration or model reload policy. Include
host lookup/transfer/materialization phase timings in that investigation;
neither a byte-copy microtime nor a short prompt identifies the cause of the
reported 10–15-second wait.
