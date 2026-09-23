# Final source replay review

This is the historical source-stage checkpoint. Final native qualification and
user-selected release acceptance are in `release-acceptance.json` and `CURRENT.md`;
its pending-work descriptions below are superseded by those records.

Reviewed pair (native qualification pending):

- vLLM `466b9d5abe84f13c29b38c686e4d77fc09e09aa4`, five commits on frozen
  v0.30.0 `ced6857afa0ea7b2e3f0846a62e1394e90f15607`.
- kernels `193cee080d654ff67c4f8bfaf924201bc88b9cf5`, eight commits on frozen
  v0.1.15 `1c7cbeee1cd0c1481d48f5031f679a47f3f0ef45`.

Both worktrees are clean. The selected targets are ancestors, commit counts
match reconciliation, effective diffs pass `git diff --check`, and all 101
changed Python files compile. Original main/release range-diffs and effective
diffs are retained privately under this run's artifact directory. All 27 report
hashes and reconciled decisions were rechecked; `commit-map.json` maps every
original SHA. v15 maps to both the native cache-dtype test commit and the KVarN
feature. The registry workaround/revert contributes no residual source delta.

The vLLM delta has 87 paths, +21,773/-393 lines; kernels 45 paths,
+10,230/-106. The large feature contains the retained implementation, native
variants, reference paths and regression coverage, folded once per repository.
Follow-up fixes are not separate repair commits. Removed bounded dead paths are
listed in reconciliation. No source support for Model Runner V2 or DFlash was
added. Common metadata forwarding in their existing files preserves a shared
API and does not qualify those configurations.

## Adaptations reviewed

- GDN padding, paired scheduling and native mixed dispatch retain upstream
  checkpoint behavior. v04 restores upstream CUDA residual RMSNorm while
  retaining functional XPU behavior. v09 retains explicit FP16/BF16 translation
  and now tests strided cache views with independent causal references and
  native-fallback rejection.
- KVarN snapshots exact device lengths once per metadata builder because v0.30
  removed exact CPU shadows. Optimistic bounds never decide permanent flushes.
  GDN takes an exact snapshot only for the scoped request-stability profile.
  Draft upper bounds are updated out of place. The exposed synchronization cost
  remains a matched-workload performance gate.
- Preserved HiSparse role/host/transfer fields, host block counts, raw allocations
  for specs without layer views, and ordinary shared-pool reconciliation.
  Independent workers must agree on pool topology and capacity. Draft-only
  attention groups participate in physical zeroing without becoming target
  metadata owners. KV transfer still rejects independent physical pools.
- Cache-only reset releases per-layer owners, shared scratch and Hadamard cache
  while retaining live layer registration and immutable selections. Shutdown
  clears the model registry. CPU weak-reference tests cover target and draft
  allocations; native lifetime/memory qualification is still required.
- Kernel k04 is reduced to the fresh-state barrier described in its audit.
  k07 checks independent CPU numerical results before replay and forbids
  fallback. KVarN resident alignment tests include fused split-16 FP16/BF16.
  A final trailing blank-line repair was folded into k07; its old local tip
  remains in `refs/recovery/release-v030-20260922/pre-final-format`.
- Eleven paired local native schema declarations (ten KVarN plus GDN) exactly
  match the released pair, including trailing width/default arguments and mixed
  dispatch. This is a source-interface check, not built-operator validation.

## Validation and limits

| Suite | Result |
| --- | --- |
| GDN padding | 5 passed; frozen-upstream negative control has 4 expected failures and 1 pass |
| GDN scheduling/native-dispatch mocks | 65 passed |
| RMSNorm non-XPU dispatch control | 1 passed |
| Cache/metadata/HiSparse/pools/draft overrides | 242 passed, 14 CUDA skips |
| KVarN contracts, scoped operators and platform guards | 352 passed |
| Worker ownership, reset and divergent-runner guards | 19 passed |
| Shared prefix pools and connector invalid-block recovery | 185 passed |
| Kernel host layout/record/address contracts | 33 passed |
| Applicable source hooks | Passed; vLLM includes mypy; kernel upstream mypy script invokes no checker |
| Final-source Nix patch application | Both patch sequences pass with `--fuzz=0` |

These are source-only CPU/mock checks with retained Nix dependencies. The private
`run-cpu-tests.py` records explicit CPU device, disabled pinned staging and hybrid
capability stubs, plus the two unchanged upstream logging fixtures extracted
from root conftest. Native operations were not qualified by those stubs. Early
network/device/fixture errors and stale test-API failures remain in raw logs,
alongside repaired passing runs. Type/format checks required ordinary annotations,
explicit tensor guards and the upstream tagged Hub API; hooks were not bypassed.

No service state or host activation changed. Build, native tests, real serving,
vision/MTP, memory/capacity, lifecycle, measured regression and provisional
optimization gates remain in `qualification-status.json`. Both source mains
have since been published with the recorded exact leases and verified against
the backup, checkpoint and stable refs. Stable publication remains pending.

Issue #17 remains paused. Its checkpoint branch and draft PR #18 are separate
preservation records and are not release inputs. xpu-v1.10.0 remains operational.
