# v13 — ADAPT: preserve recurrent reservations under attention overrides

Owner: `/root/audit_v13`; individual read-only audit, 2026-09-20. Read the
issue, manifest, complete inventory, compatibility/port maps, commit-audit
contract, applicable source instructions and v05/v08/v12 reports. Only this
report was written; no source/Git mutation, build, GPU or service operation.

## Immutable inputs and requirement

Assigned release-only commit `d6cb2e6cb2de2321fd28eea6e0aada93fb8bd736`,
parent `6519bf709dbb354926f1ac3908e2c7917e9abd23`, independently verified
patch ID `8e784853e97529503c1c198063b3dea3441679d9`. Compared against
released vLLM `2fdd9d71f6895379b0940a8f755b4c9095840558` and frozen
upstream `e378275a8f20eac9e92212e4a1fee84ec5a31dc7`; paired kernels target
`da16a5595c105605bcd44b558c7933b934a05f85`. Old bases, rolling/release tips
and full paired inventories remain in [manifest.json](../manifest.json) and
[commit-map.json](../commit-map.json).

The released KVarN allocator reserves recurrent state for every
`scheduler_config.max_num_seqs` slot before spending remaining bytes on
attention. Before v13, the independent-pool `num_gpu_blocks_override` path
multiplied one-request attention **and recurrent** bytes by the override.
An override of one full-context attention request with four scheduler slots
therefore underfunded recurrent state and failed startup. Conversely, two
attention requests with one slot overfunded recurrent bytes, inflating the
resulting attention capacity. The override denotes resident full-context
attention requests only in this independent-pool branch; ordinary shared-pool
overrides still denote physical blocks.

This is allocation correctness, not a measured kernel optimization. The
historical workload is Brutus/B70 Qwen3.8-27B W4A16/BF16, four slots, MTP2,
262,144-token request ceiling and K4V2 target/draft caches with vision retained.
Shared attention capacity and the per-request ceiling are separate quantities.

## Exact behavior and upstream coverage

In `vllm/v1/core/kv_cache_utils.py`, v13 replaces
`_independent_pool_bytes_per_request` with
`_independent_pool_bytes_for_requests`. Each group's cost is its actual layer
count × page bytes × `_physical_blocks_per_request`; Mamba costs multiply by
`max_num_seqs`, other attention costs by the requested override. The latter
helper derives resident pages from each spec's `max_memory_usage_bytes`.
It must retain `MambaSpec`'s speculative/checkpoint/cache-mode calculations,
not hardcode the historical three states per request.

`get_kv_cache_configs` adds `_null_block_bytes` to that effective budget.
Admission/auto-fit subtract null bytes once; allocation receives the full
budget and assigns one null block per independent namespace. With MTP2 and
four slots, recurrent pools therefore retain thirteen physical blocks per
layer regardless of attention override. Matching and mixed formats use their
own physical page sizes and compatible layer groups, preserving v12's grouping.
One null block per pool includes storage for every layer in that pool.

v08's `_max_memory_usage_bytes_from_groups` already implements the same
recurrent policy for one-request admission and auto-fit; it duplicates the
one-request case of v13's helper. They complement each other. The production
file is byte-identical from v13 through final release. Later v15
`2e010408b9713dca3de690f339ef46fbe72f090f` expands the regression matrix
with explicit BF16 draft controls without superseding this fix.

Frozen upstream has no KVarN allocation branch. Ordinary
`get_kv_cache_config_from_groups`/`may_override_num_blocks` retain shared-device
block accounting; `_max_memory_usage_bytes_from_groups` estimates shared-pool
demand. This differs from the fork's all-slot recurrent reservation, so upstream
does not cover the requirement.

Upstream `d43bb2f37f87a63d3a9a299c97050443cf6a2520` adds HiSparse;
`e19a3e172ecc1d1ece8ecce9dfd91bf811a814a0` shares its host cache across TP.
`hisparse/layout.py:get_hisparse_gpu_memory_usage` and
`get_hisparse_kv_cache_config` separate sparse-MLA host/device storage and use
ordinary device-block overrides. The estimator can probe that allocator.
These are not independent KVarN attention/Mamba pools. Preserve their dispatch,
host budget, worker-memory reconciliation and placement metadata during replay.

Upstream `f730a93d2bf2cad3002578238d5a1035ee1f4177` introduces
`max_num_active_seqs` for RUNNING admission. `Scheduler.max_num_active_reqs`
uses it, but V2 `GPUModelRunner.__init__` still sizes `max_num_reqs` from
`max_num_seqs`; the scheduler configuration explicitly distinguishes those
dimensions. Retain runner-slot recurrent reservations consistently. Substituting
the lower active limit only in budgeting recreates the mismatch.

## Minimum residual and dependencies

**ADAPT; fold with v08 `3b6c822e8a4f56dc3904c7fce388a76dff4c7bdd` into
v05 `fcab69494f1719e91e416aace17ef6ad501fe900`.** Consolidate admission,
auto-fit, override and allocation around one group-aware reservation policy,
with null bytes charged exactly once. Preserve unmodified profiled-memory and
ordinary/HiSparse behavior. Retain v12's actual group formats and layer counts;
global target dtype cannot supply draft page geometry.

For DFlash2, the existing allocator's equal attention-token quanta are
insufficient: they allocate every attention pool to full target-history
capacity. Merely applying v13's non-Mamba multiplier to a sliding spec also
under-reserves independent concurrent draft windows when one target-capacity
unit serves several shorter requests. Define a separate bounded concurrent
draft reservation, including in-flight/context-insertion demand, alongside
growing shared target capacity. Upstream
`SlidingWindowSpec.max_admission_blocks_per_request` accounts for retained
tokens, in-flight tokens and block alignment; use its semantics where applicable
and coordinate scheduler/allocation/lifecycle changes. Report target tokens
separately from draft-window storage. Both caches remain K4V2; no BF16 serving
milestone is needed.

There is no new operator/schema/ABI change in v13. Paired feature dependencies
remain native k08 `9bb1e2d428b8a25b262c0cd6704614bdf6ee3ad8` and compact
k09 `f26ef2323399ab410299db5c8caec41c8d63215a`; ownership and bounded
materialization remain separate v11/v14 requirements.

## Observed checks and pending qualification

Using the existing `.venv/bin/python -B` and standard-library inspection, this
audit verified default-path AST identity after excluding the changed helper
and override branch, final-release production-file identity, and archived
parent/fix source hashes. Five archived manifest/JUnit/service artifacts match
`benchmark-results/kvarn-k4v2/20260913T051252Z/pool-override-source-prepared.json`.
Raw JUnit records **9 failures/18 cases before; 29 passes afterward**, including
18 override cases and 11 retained controls. Failures cover four-slot overrides
one/two and one-slot override two across three format combinations. Thus the
tests discriminate underfunding and excess capacity, including null/page-byte
assertions. These are verified historical results, not newly executed pytest.

Pending on the port: rerun that matrix and v08 auto-fit tests; add four runner
slots/one active slot, different speculative/checkpoint counts, insufficient
reservation/null boundaries, per-worker projection, and ordinary/HiSparse
controls. Add bounded D128 draft-window allocation/admission tests. Qualify
live V2 K4V2 target/draft capacity, cancellation/reuse, MTP and vision after
all syvai weights, context buffers and scratch are loaded. No frozen-pair
runtime or performance qualification is claimed.
