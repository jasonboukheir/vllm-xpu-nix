# v13 — ADAPT: preserve recurrent reservations with capacity overrides

Owner: `/root/r30_v13`; independent read-only audit, 2026-09-22 PDT.
Only this report was written. Read the commit-audit contract, source AGENTS,
run manifest/inventory, prior v13 report and new v05/v08 reports. No source,
ref, lock, issue or service changes, builds, pytest or GPU operations.

Assigned commit `d6cb2e6cb2de2321fd28eea6e0aada93fb8bd736`, parent
`6519bf709dbb354926f1ac3908e2c7917e9abd23`, subject
`fix(kvarn): preserve recurrent reservations with capacity overrides`.
Independently recomputed stable patch ID:
`8e784853e97529503c1c198063b3dea3441679d9`.
Targets: vLLM v0.30.0 `ced6857afa0ea7b2e3f0846a62e1394e90f15607` and
kernels v0.1.15 `1c7cbeee1cd0c1481d48f5031f679a47f3f0ef45`.
Complete frozen bases, main/release tips and inventories are in the
[manifest](../manifest.json) and [commit map](../commit-map.json).
The unchanged [prior report](../../issue17-20260920-snapshot-audit/audits/v13-d6cb2e6cb2de.md)
hash matches the assigned SHA256
`16b9fc78129079c6d93579c177f23bd9269a00523d0d38f05f212f3af77bb477`.

## Requirement and final released behavior

Before v13, an independent-pool `num_gpu_blocks_override` multiplied both
attention and recurrent demand by the requested attention capacity. One
full-context attention request with four scheduler slots underfunded recurrent
state and failed startup; two attention requests with one slot overfunded
recurrent state, inflating attention capacity. Ordinary shared-pool overrides
continue to mean physical blocks.

The supported workload remains Brutus/Arc Pro B70, RedHat Qwen3.8-27B W4A16,
BF16 compute, eager V1, compact K4V2 target/draft, bundled MTP2, vision,
four slots and a 262,144-token per-request ceiling. Preserve previously
supported matching/mixed formats and BF16 draft controls. This repair supplies
allocation correctness, not a kernel performance improvement.

In released `vllm/v1/core/kv_cache_utils.py`,
`_independent_pool_bytes_for_requests` charges actual layer count × physical
page bytes × `_physical_blocks_per_request`, multiplied by `max_num_seqs`
for Mamba and requested attention capacity otherwise. The physical-block
helper uses each spec's `max_memory_usage_bytes`; do not hardcode MTP2's
three recurrent states. `get_kv_cache_configs` adds `_null_block_bytes`;
admission/auto-fit subtract it once while allocation receives the full budget.
The independent allocator first reserves recurrent states and one null per
pool, then spends remaining memory on equal attention-token quanta. MTP2
with four slots therefore retains thirteen recurrent blocks per layer for
either override. Null storage includes every layer in its pool.

The production allocator is byte-identical from v13 through released tip
`2fdd9d71f6895379b0940a8f755b4c9095840558`. Later v15
`2e010408b9713dca3de690f339ef46fbe72f090f` expands the test matrix with
BF16 drafts; it does not supersede the fix.

## Exact target coverage

Neither target contains KVarN. At `ced6857`, `kv_cache_utils.py` retains
shared-device accounting in `get_kv_cache_config_from_groups` (line 1648),
`_max_memory_usage_bytes_from_groups` (2405), and the override branch in
`get_kv_cache_configs` (2703). The latter multiplies physical blocks by shared
pool bytes; its null reservation subtracts one shared block. These paths do
not implement independently sized KVarN attention/all-slot recurrent pools.
This is a missing fork requirement, not an upstream shared-pool defect.

Included upstream changes `d43bb2f37f87a63d3a9a299c97050443cf6a2520` and
`e19a3e172ecc1d1ece8ecce9dfd91bf811a814a0` add HiSparse and shared TP
host storage. Target `hisparse/layout.py:get_hisparse_gpu_memory_usage`
(115) and `get_hisparse_kv_cache_config` (302) account for sparse-MLA
host/device storage, retaining ordinary device-block overrides. They do not
replace this reservation policy. Preserve their dispatch, host budgets and
per-worker memory reconciliation. Included sliding-window sizing change
`d29c88f162a3ec17ddb0afdf5f4f3b3ca6b84d59` also supplies no equivalent.

Target `MambaSpec.max_memory_usage_bytes` (interface line 1034) has the same
AST as the old base: `none` includes speculative states, `align` additionally
funds checkpoints, and `all` depends on context. Target V1 runner line 551
sizes `max_num_reqs` from `max_num_seqs`. Snapshot active-limit commit
`f730a93d2bf2cad3002578238d5a1035ee1f4177` is not an ancestor of this
release; its configuration/tests must not be imported from the prior audit.

## Disposition, dependencies and validation

**ADAPT; fold into v05 `fcab69494f1719e91e416aace17ef6ad501fe900` with
v08 `3b6c822e8a4f56dc3904c7fce388a76dff4c7bdd`.** As the
[v05](v05-fcab69494f17.md) and [v08](v08-3b6c822e8a4f.md) audits recommend,
consolidate v08's duplicated one-request estimator with v13's helper.
Admission, auto-fit, overrides and allocation must share the policy, with
null storage charged once. Preserve the default profiled-memory path and
ordinary/HiSparse semantics. Retain v12
`6519bf709dbb354926f1ac3908e2c7917e9abd23` grouping and actual draft
page geometry, plus v07 concurrent admission. v13 changes no native ABI;
the owning feature still needs k08 `9bb1e2d428b8a25b262c0cd6704614bdf6ee3ad8`
and k09 `f26ef2323399ab410299db5c8caec41c8d63215a`.
Issue17 stays paused: no V2, DFlash or draft-window allocation extension.

Observed checks used read-only Git and the existing `.venv/bin/python -B`
with standard-library code. Verified patch/prior-report hashes, production
file identity, Mamba AST identity, target ancestry and negative searches.
Parent/fix ASTs match after removing only the changed helper and override
branch, confirming unchanged default-path code.

Rehashed five retained manifest/JUnit/service artifacts against
[pool-override-source-prepared.json](../../../../benchmark-results/kvarn-k4v2/20260913T051252Z/pool-override-source-prepared.json).
Raw JUnit records **9 failures among 18 cases before; 29 passes afterward**
(18 override cases plus 11 controls). Exact block counts and tensor-byte
assertions distinguish underfunding from excess capacity. These are verified
historical results, not execution against the new pair.

Pending after replay: rerun the final one/four-slot × default/one/two-capacity
matrix across matching/mixed formats including BF16 controls; v08 auto-fit;
speculative/checkpoint counts, insufficient/null boundaries, projected-worker
groups and ordinary/HiSparse controls. Qualify installed-pair usable memory
and four-slot V1 K4V2/MTP2/vision startup, concurrent admission and
cancellation/reuse. No new-pair runtime qualification is claimed.
