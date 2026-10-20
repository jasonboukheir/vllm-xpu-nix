# v08 — complete offloaded speculative pages at the context ceiling

**Disposition: ADAPT; fold into the surviving v05 KVarN feature.** Keep the
existing V1 repair and the no-write ownership invariant. V2 already runs draft
context population without V1's context-fit gate, and its autoregressive
speculator now supports a cache-populating K=0 call. Use that implementation
where applicable. It does not prove context-ceiling write safety or KVarN/RAM
prefix completeness for V2 MTP or DFlash2; those remain explicit port and
qualification requirements.

## Frozen inputs and scope

| Item | Value |
| --- | --- |
| Repository | `/home/jasonbk/Projects/vllm` |
| Assigned commit | `dcba4007ef6af8ba143d2794c194c36b58397339` |
| Parent | `a76fbb1ce06d50159978786f70daf44c27847ddd` |
| Subject | `Keep offloaded MTP prefix pages complete at the context limit` |
| Patch ID | `5fc6315431d387490d0bb735c07334c55b1e4d51` |
| Old upstream base | `db9527a46873454610df6dbedf79a36d6bf1a7f6` |
| Final old tip | assigned commit |
| Selected upstream | `5df63a7203cdb9328d824a27b682e77c7b9928be` (`v0.32.0`) |
| Kernel old/new base | `ddf336d86e3c8602888572a3502f951abd51df12` |
| Kernel old tip | `dbb684f4c9b38a8b3bbf52bda0c490ee47049221` |

Read source `AGENTS.md`, `references/commit-audit.md`, the full paired frozen
inventory in `../inventory.json`, this run's v05/performance reports and the
`xpu-v1.12.2.json` release manifest. The v07 owner independently reported the
DFlash query-slot boundary issue described below. All source inspection used
frozen Git objects. This audit writes only this report; no source, Git, service,
lock or external issue mutation, implementation, or runtime test was performed.

## Requirement independent of runner and drafter

A reusable prefix may be published only when every cache component it claims
is complete for its represented accepted history. Populate necessary draft
context even when no additional proposals fit. A speculative row that writes
no KV must not lower an existing page's fill count, reclaim its readable
resident history, consume its pending restore, or overwrite accepted context.
Partial pages must remain partial; skipping an error or marking missing data
complete would violate this requirement.

The demonstrated old workload is Intel Arc Pro B70, RedHat Qwen3.8-27B INT4,
BF16 compute, eager V1, compact K4V2 target/bundled MTP2, prefix caching and an
8 GiB `IndependentCPUOffloadingSpec` arena with `active_only` GPU retention.
Its supported maximum context is 262144. The requirement also applies to the
requested V2/DFlash2 path, although a separate BF16 sliding-window drafter may
need reconstruction rather than serialized KVarN draft pages. The export or
restore contract must accurately describe which state is available.

v08 is the final old commit, so no later repair supersedes it. Its changes are:

- In `gpu_model_runner.py`, when the target fits but MTP proposals do not,
  `_should_populate_offloaded_mtp_cache` enables a first draft forward using a
  copied scheduler output with K=0. The condition is deliberately restricted to
  compact K4V2, bundled MTP and the independent CPU offloading connector. The
  returned Bx0 tensor is discarded; normal no-draft placeholders and valid-token
  bookkeeping remain intact. The original scheduler output is not mutated.
- The existing `llm_base_proposer.py:617-626` runs the first forward before
  returning an empty tensor for K=0. v08 uses that available behavior; it does
  not add a second bespoke model-forward implementation.
- `kvarn_attn.py:1199-1211` identifies the V1 masked one-token MTP row
  (`seq_len=1`, query length 1, negative write slot). Subsequent builder logic
  skips its fill update, protects its resident blocks from reclaim/flush and
  leaves its pending sink restore untouched. The completed pages remain
  available for export.
- `prepare_cpu_offload` retains fail-closed rejection of incomplete pages and
  adds layer, physical block, fill count and block-size diagnostics.

These repair v07 RAM offload on top of v05/v06. A boundary request can otherwise
either leave an MTP page unwritten or falsely downgrade a completed sink to one
filled token; attempting export then raises `Cannot offload an incomplete KVarN
page`, or prematurely reclaimed data is unavailable for correct reuse.

## Upstream coverage and residual work

All line numbers below refer to the frozen target unless explicitly labeled
old-tip. Searching that target's `vllm` and `tests` finds no KVarN,
`IndependentCPUOffloadingSpec`, or `_should_populate_offloaded_mtp_cache`.

### V1: no replacement for the repair

`vllm/v1/worker/gpu_model_runner.py:4288-4301` still requires
`max_seq_len + num_drafter_query_tokens <= effective_drafter_max_model_len`.
At `:4408-4426`, failed fit skips the draft model forward, doing only valid-token
bookkeeping and any necessary DP dummy run. `:4454-4465` publishes zero draft
placeholders. There is no cache-only fallback. Thus preserving V1 support
requires keeping v08's scoped fallback, bookkeeping and tests when porting v05.

The upstream zero-draft recurrent fix
`0eac15270710ddb7f5296d7244fcd0496c2daa0f` routes speculation-capable recurrent
rows consistently; it does not populate this skipped draft forward or maintain
KVarN page ownership. Keep its new behavior as well, rather than replacing it
with an older runner implementation.

### V2 MTP: upstream covers first-forward ordering, not the whole invariant

`gpu/model_runner.py:2225-2257` calls `speculator.propose` without the V1 fit
gate, passes the scheduler's K, and masks unused returned draft columns.
`gpu/spec_decode/target_dependent_ar/speculator.py:276-299` performs draft
prefill first, then for K=0 fills the draft result with -1 and returns. This
explicit dynamic-K behavior was added by
`74c5cbcd7b6fa07fa5568ba7f7dfbfedc502964e` (#57053); the subsequent rename to
the current path is `6ee9be6b3c2dd5557ad72cf714b9d8fe3c2009d4` (#60335).
This is real partial coverage and should be reused, not duplicated by copying
V1's private guard into V2.

However, scheduler `num_spec_tokens_to_schedule` is the global configured or
dynamic batch K (`core/sched/scheduler.py:1485-1490`), not evidence that it was
reduced per request at the context ceiling. Standard MTP advances positions
(`target_dependent_ar/speculator.py:73-80`), and its decode preparation and
updates clamp extra positions to `max_model_len - 1` (`:839-847`, `:937-947`).
`gpu/block_table.py:341-352` maps those positions to physical write slots. A
clamped position can consequently still name a real final-page slot. The old
KVarN `seq_len=1/slot=-1` recognition does not identify this case.

This is a source-level boundary risk, not a measured corruption result. The
port must prove that extra speculative work cannot overwrite an accepted
final-page position or make fill/commit metadata claim data that was not
written. Reducing only the published proposal count is insufficient if the
forward has already written to committed history. Keep cache population and
proposal/write validity separate, including mixed near-limit and ordinary rows.

### V2 DFlash2: context population exists; query bounds require attention

`gpu/spec_decode/dflash/speculator.py:477-493` populates context KV before draft
query execution (eager mode does so at `:493`). Context slots at `:616-634`
already mask rejected suffixes and null/evicted blocks. The new metadata
shortcut `36f397281295f5b58852a7eaf429b90348161079` only skips full metadata
building for qualified full-graph backends; it supplies no KVarN lifecycle
replacement.

For draft queries, `:637-670` computes an unbounded `query_pos`, clamps its
block-table index to the final entry, accepts a write slot whenever that entry
is nonzero, and separately clamps the model position to `max_model_len - 1`.
The write-validity predicate lacks `query_pos < max_model_len`. For example,
with CP=1, block size 128, max length 512 and a four-entry table, query position
512 selects table entry 3 and offset 0: it aliases the page holding position
384 if that table entry is a real page. This counterexample is static reasoning;
actual table allocation/lookahead and execution must be tested on the port.

DFlash accepts a `num_speculative_tokens` argument at `:369` but uses its fixed
`num_query_per_req` at `:373-376` and throughout the forward. Therefore passing
K=0 alone is not a cache-only DFlash execution contract. Mask out-of-range query
writes before insertion while still populating valid committed context, and
make the metadata/lifecycle behavior agree with those masks. Existing null-block
masking and rejection masking must remain intact. If the selected BF16 draft
backend avoids compact draft metadata, validate its write bounds too: accepted
draft context is still needed for a correct prefix continuation.

V2 `gpu/model_runner.py:2275-2277` invokes connector `post_forward` after draft
proposal; `gpu/kv_connector.py:131-150` waits for saves and reports transfer
results. This provides an appropriate later fence for completed draft context,
but does not prove local deferred stores/materialization are correctly ordered
or that a target prefix hit supplies missing DFlash hidden/context states.
Resolve that jointly with v07's connector/restore port.

### Offloading changes are complementary

Relevant old-to-new upstream changes include XPU host-memory registration
`b027ac831253e83ff605a2ff5d4aa7c00619802c`, worker-rank slot selection
`2bea22002d66a6c89d208fa682b40a89538f023c`, and precise async-load zeroing
exemptions `ca0df1f1227cdd993835a4d2a62861dd3bd2e893`. Their changed paths are
the generic CPU offload worker/spec/registration and scheduler/cache manager.
They improve transport and load safety, but none fills skipped MTP context,
implements KVarN sink sidecars, or protects its resident fill/ownership state.
They are not grounds to drop v08.

## Minimum residual change and dependencies

1. Preserve the narrow V1 cache-only call, normal output bookkeeping, diagnostic
   and no-write metadata semantics. Keep the existing failure when a page truly
   is incomplete.
2. Port the no-write invariant using each V2 path's real write/commit masks and
   exact accepted history. Do not treat upper-bound lengths or clamped positions
   as proof of writes. Retain pending restore/ownership for masked rows and
   ensure active rows in the same batch still progress and reclaim normally.
3. Reuse V2 AR's existing initial forward/K=0 behavior. Ensure the chosen
   ceiling policy prevents extra writes, including ordinary K>0 and mixed
   batches. For DFlash, validate/mask query-slot bounds independently from
   context precompute and proposal publication.
4. Establish prefix completeness at the actual v07 store/restore fence. Keep
   independently owned target/draft metadata and KVarN groups as required by
   v05; a target cache hit cannot stand in for absent draft context.

Prerequisites are v05 `ef92dc5600af64103c01f97e3300b81c3a1083ee`, v06
`81dc387cf9a80e88468b7f77f7df8dfb71939921` and v07
`a76fbb1ce06d50159978786f70daf44c27847ddd`. Fold v08's surviving implementation
and tests into the v05 feature after reconciling those audits. v08 adds no
kernel symbol, ABI or packed-record format; paired KVarN kernel feature k07
`dbb684f4c9b38a8b3bbf52bda0c490ee47049221` remains needed through v05. There
is no v08-only native operator to delete. Coordinate new boundary protection
with v07 rather than introducing two independent fixes for the same writes.

## Validation: observed versus required

**Observed historical evidence, not rerun in this audit:** v08 adds nine
parameterized V1 runner checks in
`tests/v1/worker/test_gpu_model_runner.py::test_sample_tokens_populates_offloaded_mtp_cache_at_context_limit`
and two lifecycle cases in
`tests/quantization/test_kvarn.py::test_padded_mtp_step_preserves_completed_pages_for_cpu_offload`.
The runner cases cover 4094/4095/4096/4097 with max length 4096 and K=2,
connector/dtype/drafter scope, preserved scheduler K, discarded cache-only
output and valid-token/connector bookkeeping. The lifecycle cases cover single
and mixed padded/active rows, unchanged complete-page fill, retained ownership,
normal active-row flush and successful offload preparation.

The xpu-v1.12.2 manifest records 1310 installed CPU passes and two skips
(1299 focused, nine boundary runner, two mocked DMA cases), separate identical
source-overlay results, and installed GPU qualification. Its exact-ceiling
case ran 262143 input + 1 output twice with identical IDs, RAM reuse and
successful deferred-store flush checks. These establish why this is a supported
requirement; they are not v0.32/V2 evidence.

**Required on the port:**

- Rerun the existing V1 runner/lifecycle regressions, including the control
  where removal of the fallback leaves MTP unpopulated and the control where
  removal of padded-row protection changes fill/ownership. Verify a truly
  incomplete page continues to reject export.
- Extend the existing V2 target-dependent AR tests and DFlash input-preparation
  tests with just-below, exactly-at and beyond-ceiling query positions; K=0,
  K=1, normal K; mixed short/near-limit rows; accepted/rejected suffixes;
  full/partial final pages; and a real nonzero final table entry. Assert physical
  write slots and unchanged committed-page bytes, not merely clamped RoPE
  positions or output shapes. `test_dflash_prepare_inputs.py` currently uses
  CUDA-only fixtures, so adapt the existing device fixture for an actual XPU
  run; a CUDA skip is not validation on Brutus.
- At the connector fence, byte-check all exported complete pages and sink
  sidecars, restore to different/recycled physical IDs, and compare continuation
  against recomputation. Include cancellation/finalization and next-step
  deferred-store flush, so success before a delayed failing store cannot pass.
- On Brutus, retain the released exact-context scenario and exercise V2 MTP
  and the chosen DFlash2 draft with RAM offload/reuse, concurrent normal rows,
  recurrent rollback and prefix hits that skip target hidden-state production.
  Check returned tokens, completion/limits, offload errors, accepted history
  and resident/host memory. Prefix reconstruction must be correct before any
  performance comparison is meaningful.

The fix is correctness-driven. Measure the added work only at affected boundary
requests and confirm ordinary short-query latency/throughput does not regress;
no new speedup is claimed. GPU execution, V2 metadata ownership, actual DFlash
lookahead allocation and the final restore design remain unverified here.
